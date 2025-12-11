"""
AMI Health Worker V4 - single-file worker.py

Features:
- Polls Supabase for reports with ai_status='pending'
- Downloads PDF
- Detects digital vs scanned PDF
- If scanned: converts pages to images via pdf2image
- OCR: prefer local pytesseract, fallback to OpenAI Vision (base64 JPEG)
- Robust regex parser for CBC & chemistry
- Canonical mapping into keys: Hb, MCV, MCH, WBC, Neutrophils, Platelets, Creatinine, CRP, etc.
- Route Engine V4: patterns → routes → next_steps, differential diagnoses, severity score (1-5), urgency flags
- Age & sex-aware interpretation and routes
- Colour-coded flags (hex + tailwind class) in output for frontend
- Saves ai_results into Supabase and sets ai_status='completed'

Requirements (from you):
- pypdf==4.3.1
- supabase==2.7.4
- python-dotenv==1.0.1
- requests==2.31.0
- openai==1.51.0
- pdf2image==1.17.0
- pillow==10.2.0
- pytesseract==0.3.10 (optional, preferred)

NOTE: fill .env with SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY, PAGES_TO_RENDER, POLL_INTERVAL

"""

import os
import io
import re
import time
import json
import base64
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

# optional imports - handle runtime absence
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except Exception:
    HAS_SUPABASE = False

try:
    import openai
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

# ---------- config & logging ----------
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '8'))  # seconds between polls
PDF_RENDER_DPI = int(os.getenv('PDF_RENDER_DPI', '200'))
TEXT_LENGTH_THRESHOLD = int(os.getenv('TEXT_LENGTH_THRESHOLD', '80'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ami-worker')

if HAS_OPENAI:
    openai.api_key = OPENAI_API_KEY

if HAS_SUPABASE:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# ---------- helpers: PDF read / scanned detection ----------

def download_pdf_from_supabase(report_record: Dict[str, Any]) -> bytes:
    """Download PDF binary for a report. Assumes 'file_path' or 'pdf_url' present in record."""
    # This is an example. Adapt to your Supabase file storage schema.
    if 'pdf_url' in report_record and report_record['pdf_url']:
        url = report_record['pdf_url']
        import requests
        r = requests.get(url)
        r.raise_for_status()
        return r.content
    elif 'file_path' in report_record and supabase:
        # fetch from supabase storage
        bucket = os.getenv('SUPABASE_BUCKET', 'reports')
        path = report_record['file_path']
        res = supabase.storage.from_(bucket).download(path)
        return res
    else:
        raise ValueError('No pdf_url or file_path in report record')


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Attempt to extract text using pypdf. If result is short, assume scanned."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text_parts = []
        for p in reader.pages:
            try:
                text = p.extract_text() or ''
            except Exception:
                text = ''
            text_parts.append(text)
        text_joined = '\n'.join(text_parts)
        return text_joined
    except Exception as e:
        logger.warning('pypdf failed to extract text: %s', e)
        return ''


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    text = extract_text_from_pdf_bytes(pdf_bytes)
    if len(text.strip()) < TEXT_LENGTH_THRESHOLD:
        logger.info('PDF appears scanned (text length %d)', len(text))
        return True
    return False


# ---------- OCR ----------

def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
    return imgs


def ocr_image_with_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError('pytesseract not available')
    # pre-process: convert to grayscale, threshold
    gray = img.convert('L')
    # small resize to improve OCR if tiny
    w, h = gray.size
    if max(w, h) < 1600:
        factor = max(1, int(1600 / max(w, h)))
        new_size = (w * factor, h * factor)
        gray = gray.resize(new_size, Image.LANCZOS)
    text = pytesseract.image_to_string(gray, lang='eng')
    return text


def ocr_with_openai_vision(img: Image.Image, max_width=1200) -> str:
    """Fallback OCR using OpenAI Vision via image-to-text. Requires OPENAI_API_KEY.
    This function base64-encodes a resized JPEG and calls OpenAI 'gpt-4o-mini' style vision endpoint.
    Adjust to the specific OpenAI API you have available. Here we use chat completions as conceptual fallback.
    """
    if not HAS_OPENAI:
        raise RuntimeError('OpenAI not configured')
    # resize
    w, h = img.size
    if w > max_width:
        ratio = max_width / float(w)
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
    buffered = io.BytesIO()
    img.convert('RGB').save(buffered, format='JPEG', quality=80)
    encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # The concrete API call may differ; here's a placeholder using openai.ChatCompletion
    # Replace 'gpt-4o-mini' / content keys with your actual image-capable model and method.
    prompt = (
        "You are an OCR assistant. Extract the full plain text from the following base64-encoded JPEG. "
        "Return only the extracted text, no commentary.\nIMAGE_BASE64:\n" + encoded
    )
    try:
        resp = openai.ChatCompletion.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=3000,
        )
        txt = resp['choices'][0]['message']['content']
        return txt
    except Exception as e:
        logger.exception('OpenAI OCR failed: %s', e)
        return ''


def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    """Top-level OCR: prefer pytesseract; fallback to OpenAI vision. Returns concatenated text for all pages."""
    try:
        images = pdf_to_images(pdf_bytes)
    except Exception as e:
        logger.exception('Failed to render PDF to images: %s', e)
        raise

    texts = []
    for i, img in enumerate(images):
        page_text = ''
        if HAS_PYTESSERACT:
            try:
                page_text = ocr_image_with_pytesseract(img)
                logger.info('OCR page %d with pytesseract length %d', i, len(page_text))
            except Exception as e:
                logger.warning('pytesseract failed on page %d: %s', i, e)
                page_text = ''
        if not page_text:
            try:
                page_text = ocr_with_openai_vision(img)
                logger.info('OCR page %d with OpenAI length %d', i, len(page_text))
            except Exception as e:
                logger.exception('OpenAI OCR failed on page %d: %s', i, e)
                page_text = ''
        texts.append(page_text)
    return '\n\n---PAGE_BREAK---\n\n'.join(texts)


# ---------- Parsing: robust regex extraction ----------

# We will look for common lab value patterns like: Hb: 11.6 g/dL (ref: 12.4-16.7)
# and many variants: 'HAEMOGLOBIN 11.6 g/dL (ref: 12.4-16.7)', 'Hb 11.6 g/dL'

VALUE_RE = r'(-?\d+\.?\d*)'
UNIT_RE = r'([a-zA-Z/%^0-9\-\s]+)?'
REF_RE = r'(?:ref[:]?\s*\(?([\d\.\-–to,\s]+)\)?)?'

COMMON_KEYS = {
    'hb': ['hb', 'haemoglobin', 'hemoglobin'],
    'rbc': ['erythrocyte count', 'erythrocyte', 'rbc'],
    'hct': ['haematocrit', 'hematocrit', 'hct'],
    'mcv': ['mcv', 'mean corpuscular volume'],
    'mch': ['mch', 'mean corpuscular hemoglobin'],
    'mchc': ['mchc'],
    'rdw': ['rdw'],
    'wbc': ['wbc', 'leukocyte', 'leucocyte', 'white cell count', 'leucocyte count'],
    'neutrophils_pc': ['neutrophils %', 'neutrophils', 'neutrophil%'],
    'neutrophils_abs': ['neutrophils x10', 'neutrophils absolute', 'neutrophil count'],
    'lymphocytes_pc': ['lymphocytes %', 'lymphocytes'],
    'monocytes_pc': ['monocytes %', 'monocytes'],
    'eosinophils_pc': ['eosinophils %', 'eosinophils'],
    'basophils_pc': ['basophils %', 'basophils'],
    'platelets': ['platelets', 'thrombocytes', 'platelet count'],
    'crp': ['crp', 'c-reactive protein', 'c reactive protein'],
    'creatinine': ['creatinine'],
    'sodium': ['sodium', 'na'],
    'potassium': ['potassium', 'k'],
    'chloride': ['chloride', 'cl'],
    'urea': ['urea'],
    # add more as needed
}

# reverse lookup to key by label
LABEL_TO_KEY = {}
for k, labels in COMMON_KEYS.items():
    for lbl in labels:
        LABEL_TO_KEY[lbl.lower()] = k


def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    lines = [l.strip() for l in re.split(r'\n|\r', text) if l.strip()]

    # try direct patterns first: label + value
    for line in lines:
        # normalize
        line_n = re.sub(r'\s+', ' ', line.lower())
        # simple label-value regex: label ... number
        for label, key in LABEL_TO_KEY.items():
            if label in line_n:
                # attempt extract number near label
                pat = re.compile(rf'{re.escape(label)}[^\d\-\+\n\r]{{0,25}}({VALUE_RE})')
                m = pat.search(line_n)
                if m:
                    val = float(m.group(1))
                    unit_match = re.search(rf'{re.escape(label)}[^\n]*?([a-zA-Z/%\d]+)\b', line, re.IGNORECASE)
                    unit = unit_match.group(1).strip() if unit_match else None
                    results[key] = {'value': val, 'unit': unit, 'raw_line': line}
                    continue
        # fallback: generic "Hb 11.6 g/dL (ref: 12.4-16.7)" style
        generic = re.findall(rf'([A-Za-z\- ]{{2,20}})\s*[:]?\s*{VALUE_RE}\s*{UNIT_RE}\s*(?:\(ref[:]?\s*([^\)]*)\))?', line)
        for g in generic:
            label_raw = g[0].strip()
            val = g[1]
            unit = g[2].strip() if g[2] else None
            ref = g[3].strip() if g[3] else None
            key = find_key_for_label(label_raw)
            if key:
                try:
                    value = float(val)
                    results[key] = {'value': value, 'unit': unit, 'ref': ref, 'raw_line': line}
                except Exception:
                    pass
    # special handling: percentages and absolute neutrophils like "Neutrophils: 88.0% 17.08 x10^9/L"
    for line in lines:
        if 'neutrophil' in line.lower():
            percent = re.search(r'([0-9]{1,3}\.\d|[0-9]{1,3})\s*%', line)
            absval = re.search(r'([0-9]{1,3}\.\d{1,3})\s*x?10\^?\-?\d*\/?L|([0-9]{1,3}\.\d{1,3})\s*x?10\^?\d?\/L', line)
            # simpler absolute extraction
            absval2 = re.search(r'([0-9]{1,3}\.\d{1,3})\s*x10\^?\d?\/?L', line)
            if percent:
                try:
                    results.setdefault('neutrophils_pc', {})['value'] = float(percent.group(1))
                except:
                    pass
            # absolute neutrophils
            abs_match = re.search(r'([0-9]{1,3}\.\d{1,3})\s*x10\^?\d?\/?L', line)
            if abs_match:
                try:
                    results.setdefault('neutrophils_abs', {})['value'] = float(abs_match.group(1))
                except:
                    pass

    # final light heuristic: look for simple "Hb 11.6 g/dL" across entire text
    fallback_matches = re.findall(rf'\b(hb|haemoglobin|haemoglobin)\b[^\d\n\r]{{0,20}}{VALUE_RE}', text, flags=re.IGNORECASE)
    for m in fallback_matches:
        try:
            results.setdefault('hb', {})['value'] = float(m[1])
        except:
            pass

    return results


def find_key_for_label(label: str) -> Optional[str]:
    l = re.sub(r'[^a-z0-9 ]', '', label.lower())
    l = l.strip()
    # try direct
    if l in LABEL_TO_KEY:
        return LABEL_TO_KEY[l]
    # try partial match
    for k_label, key in LABEL_TO_KEY.items():
        if k_label in l or l in k_label:
            return key
    return None


# ---------- Canonical mapping ----------

CANONICAL_KEYS = ['Hb', 'MCV', 'MCH', 'MCHC', 'RDW', 'WBC', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils', 'NLR', 'Platelets', 'Creatinine', 'CRP', 'Sodium', 'Potassium', 'Chloride', 'Urea', 'RBC', 'HCT']


def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    # mapping rules from parsed keys to canonical
    map_rules = {
        'hb': 'Hb',
        'mcv': 'MCV',
        'mch': 'MCH',
        'mchc': 'MCHC',
        'rdw': 'RDW',
        'wbc': 'WBC',
        'neutrophils_pc': 'Neutrophils',
        'neutrophils_abs': 'Neutrophils',
        'lymphocytes_pc': 'Lymphocytes',
        'monocytes_pc': 'Monocytes',
        'eosinophils_pc': 'Eosinophils',
        'basophils_pc': 'Basophils',
        'platelets': 'Platelets',
        'creatinine': 'Creatinine',
        'crp': 'CRP',
        'sodium': 'Sodium',
        'potassium': 'Potassium',
        'chloride': 'Chloride',
        'urea': 'Urea',
        'rbc': 'RBC',
        'hct': 'HCT',
    }

    for k, v in parsed.items():
        if k in map_rules:
            canon = map_rules[k]
            # merge multiple sources (percentage + absolute for neutrophils)
            if canon not in out:
                out[canon] = {}
            # prefer numeric value
            try:
                out[canon]['value'] = float(v.get('value'))
            except Exception:
                out[canon]['value'] = None
            if v.get('unit'):
                out[canon]['unit'] = v.get('unit')
            if v.get('ref'):
                out[canon]['ref'] = v.get('ref')
            out[canon]['raw'] = v.get('raw_line')
    # calculate NLR if possible
    if 'Neutrophils' in out and 'Lymphocytes' in out:
        try:
            n = float(out['Neutrophils']['value'])
            l = float(out['Lymphocytes']['value'])
            if l > 0:
                out['NLR'] = {'value': round(n / l, 2), 'unit': None}
        except Exception:
            pass
    return out


# ---------- Route Engine V4 ----------

# Severity scoring helper: 1 (mild) - 5 (critical)

def score_severity_for_abnormality(key: str, value: float, age_group: str, sex: str) -> int:
    """Age and sex aware severity scoring heuristics. These are clinical heuristics for triage."""
    # default mild
    score = 1
    # example rules -- expand as needed
    if key == 'Hb':
        # thresholds vary by sex & age group
        if sex.lower() == 'female':
            low_cut = 12.0 if age_group not in ['neonate', 'infant'] else 14.0
        else:
            low_cut = 13.0
        if value is None:
            return 1
        if value < low_cut - 3:
            score = 5
        elif value < low_cut - 1.5:
            score = 4
        elif value < low_cut:
            score = 3
        else:
            score = 1
    elif key == 'WBC':
        if value is None:
            return 1
        if value > 25:
            score = 5
        elif value > 15:
            score = 4
        elif value > 11:
            score = 3
    elif key == 'CRP':
        if value is None:
            return 1
        if value > 200:
            score = 5
        elif value > 100:
            score = 4
        elif value > 50:
            score = 3
        elif value > 10:
            score = 2
    elif key == 'Creatinine':
        if value is None:
            return 1
        # very rough AKI thresholds
        if value > 354:
            score = 5
        elif value > 200:
            score = 4
        elif value > 120:
            score = 3
    elif key == 'Platelets':
        if value is None:
            return 1
        if value < 20:
            score = 5
        elif value < 50:
            score = 4
        elif value < 100:
            score = 3
    else:
        score = 1
    return score


COLOR_MAP = {
    5: {'label': 'critical', 'color': '#b91c1c', 'tw': 'bg-red-600', 'urgency': 'high'},
    4: {'label': 'severe', 'color': '#f97316', 'tw': 'bg-orange-500', 'urgency': 'high'},
    3: {'label': 'moderate', 'color': '#f59e0b', 'tw': 'bg-yellow-400', 'urgency': 'medium'},
    2: {'label': 'borderline', 'color': '#facc15', 'tw': 'bg-yellow-300', 'urgency': 'low'},
    1: {'label': 'normal', 'color': '#10b981', 'tw': 'bg-green-500', 'urgency': 'low'},
}


def age_group_from_age(age: Optional[float]) -> str:
    if age is None:
        return 'adult'
    try:
        a = float(age)
    except:
        return 'adult'
    if a < (1/12):
        return 'neonate'
    if a < 1:
        return 'infant'
    if a < 13:
        return 'child'
    if a < 18:
        return 'teen'
    if a < 65:
        return 'adult'
    return 'elderly'


# full route engine: produce patterns, routes, next steps, ddx, severity and urgency

def route_engine_v4(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    age = patient_meta.get('age')
    sex = patient_meta.get('sex', 'unknown')
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    severity_scores = []
    urgency_flags = []

    # helper to add
    def add_find(name, reason, score):
        patterns.append({'pattern': name, 'reason': reason})
        severity_scores.append(score)

    # Anemia patterns
    hb = canonical.get('Hb', {}).get('value')
    mcv = canonical.get('MCV', {}).get('value')
    mch = canonical.get('MCH', {}).get('value')
    rdw = canonical.get('RDW', {}).get('value')

    if hb is not None:
        hb_score = score_severity_for_abnormality('Hb', hb, ag, sex)
        if hb_score > 1:
            add_find('anemia', f'Hb {hb} g/dL (age_group={ag}, sex={sex})', hb_score)
            # microcytic
            if mcv is not None and mcv < 80:
                add_find('microcytic anemia', f'MCV {mcv} fL', max(3, hb_score))
                routes.append('Iron deficiency route')
                ddx.extend(['Iron deficiency anemia', 'Thalassemia trait', 'Chronic blood loss'])
                # age/sex modifiers
                if ag == 'teen' and sex.lower() == 'female':
                    next_steps.append('Evaluate menstrual blood loss; order ferritin and reticulocyte count')
                else:
                    next_steps.append('Order ferritin and reticulocyte count; check stool occult blood if adult')
            elif mcv is not None and mcv > 100:
                add_find('macrocytic anemia', f'MCV {mcv} fL', max(3, hb_score))
                routes.append('Macrocytic anemia route')
                ddx.extend(['Vitamin B12 deficiency', 'Folate deficiency', 'Alcohol-related', 'Myelodysplasia'])
                next_steps.append('Order B12, folate, and reticulocyte count; review medications')
            else:
                add_find('normocytic anemia', 'MCV normal or missing', max(2, hb_score))
                routes.append('Normocytic anemia route')
                ddx.extend(['Acute blood loss', 'Hemolysis', 'Chronic disease'])
                next_steps.append('Order reticulocyte count, LDH, peripheral smear')

    # Infection / inflammation patterns
    wbc = canonical.get('WBC', {}).get('value')
    crp = canonical.get('CRP', {}).get('value')
    neut = canonical.get('Neutrophils', {}).get('value')
    nlr = canonical.get('NLR', {}).get('value')

    if wbc is not None and wbc > 11:
        wbc_score = score_severity_for_abnormality('WBC', wbc, ag, sex)
        add_find('leukocytosis', f'WBC {wbc} x10^9/L', wbc_score)
        if neut is not None and neut > 70:
            add_find('neutrophilic predominance', f'Neutrophils {neut}%', max(3, wbc_score))
            routes.append('Bacterial infection route')
            ddx.extend(['Bacterial infection', 'Acute inflammation'])
            next_steps.append('Assess clinically for source of infection; consider empiric antibiotics if unstable')
    if crp is not None and crp > 10:
        crp_score = score_severity_for_abnormality('CRP', crp, ag, sex)
        add_find('elevated inflammatory marker', f'CRP {crp} mg/L', crp_score)
        if crp > 50:
            routes.append('Significant inflammatory response')
            ddx.extend(['Bacterial infection', 'Severe inflammatory condition'])
            next_steps.append('Consider urgent clinical review, blood cultures if febrile, start sepsis workup if clinically indicated')

    # NLR
    if nlr is not None and nlr > 5:
        add_find('high NLR', f'NLR {nlr}', 4 if nlr > 10 else 3)
        routes.append('High NLR: likely bacterial/critical inflammatory response')
        next_steps.append('Urgent clinical assessment for sepsis/invasive infection')

    # Platelets
    plate = canonical.get('Platelets', {}).get('value')
    if plate is not None:
        plate_score = score_severity_for_abnormality('Platelets', plate, ag, sex)
        if plate < 150:
            add_find('thrombocytopenia', f'Platelets {plate} x10^9/L', plate_score)
            ddx.extend(['Immune thrombocytopenia', 'DIC', 'Bone marrow problem'])
            next_steps.append('Check peripheral smear, repeat platelet count, evaluate for bleeding')
        elif plate > 450:
            add_find('thrombocytosis', f'Platelets {plate} x10^9/L', 2)
            ddx.extend(['Reactive thrombocytosis', 'Myeloproliferative disorder'])
            next_steps.append('Consider inflammatory/reactive cause; repeat and consider further haematology testing if persistent')

    # Creatinine / AKI
    creat = canonical.get('Creatinine', {}).get('value')
    if creat is not None:
        creat_score = score_severity_for_abnormality('Creatinine', creat, ag, sex)
        if creat_score >= 3:
            add_find('elevated creatinine', f'Creatinine {creat} umol/L', creat_score)
            routes.append('AKI route')
            ddx.extend(['Acute kidney injury', 'Chronic kidney disease'])
            next_steps.append('Assess urine output, review medications, check electrolytes and repeat creatinine urgently')

    # Combined patterns
    # anaemia + high CRP + leukocytosis -> consider bone marrow suppression vs acute infection causing anemia of inflammation
    if hb is not None and crp is not None and wbc is not None:
        if hb < 12 and crp > 20 and wbc > 11:
            add_find('anemia with inflammatory response', 'Low Hb with high CRP and leukocytosis', 4)
            routes.append('Infection with anemia route')
            ddx.extend(['Inflammation causing anemia', 'Concurrent iron deficiency with infection'])
            next_steps.append('Treat source of infection; reassess Hb after infection controlled; do ferritin when CRP falls')

    # Age-specific modifications & final urgency
    combined_score = max(severity_scores) if severity_scores else 1
    color_entry = COLOR_MAP.get(combined_score, COLOR_MAP[1])
    urgency = color_entry['urgency']

    # build human-friendly summary for ER
    summary_lines = []
    # concise bullet lines
    if patterns:
        summary_lines.append('Patterns detected: ' + '; '.join([p['pattern'] for p in patterns]))
    if routes:
        summary_lines.append('Primary route(s): ' + '; '.join(routes))
    if ddx:
        summary_lines.append('Top differential diagnoses: ' + ', '.join(dict.fromkeys(ddx)))
    if next_steps:
        summary_lines.append('Immediate suggested actions: ' + ' | '.join(next_steps))

    # age-aware note
    age_note = ''
    if ag == 'teen' and sex.lower() == 'female':
        age_note = 'Teenage female — consider menstrual blood loss and iron deficiency as high-likelihood causes.'
    elif ag == 'elderly':
        age_note = 'Elderly patient — broaden differential to include chronic disease and malignancy.'

    final = {
        'patterns': patterns,
        'routes': routes,
        'next_steps': next_steps,
        'differential': list(dict.fromkeys(ddx)),
        'severity_score': combined_score,
        'urgency_flag': urgency,
        'color': color_entry['color'],
        'tw_class': color_entry['tw'],
        'age_group': ag,
        'age_note': age_note,
        'summary': '\n'.join(summary_lines[:6]) if summary_lines else 'No significant abnormalities detected.'
    }

    return final


# ---------- Trend analysis (if previous results exist) ----------

def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {'trend': 'no_previous'}
    diffs = {}
    for k, v in current.items():
        prev_val = previous.get(k, {}).get('value') if previous else None
        cur_val = v.get('value')
        if prev_val is None or cur_val is None:
            continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {'previous': prev_val, 'current': cur_val, 'delta': delta, 'pct_change': pct}
        except Exception:
            pass
    return {'trend': diffs}


# ---------- Supabase update ----------

def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        logger.warning('Supabase client not available; skipping save.')
        return
    table = os.getenv('SUPABASE_TABLE', 'reports')
    update_payload = {'ai_status': 'completed', 'ai_results': ai_results}
    res = supabase.table(table).update(update_payload).eq('id', report_id).execute()
    logger.info('Saved ai_results for report %s status: %s', report_id, res.status_code if hasattr(res, 'status_code') else 'ok')


# ---------- Full worker loop ----------

def process_report(report_record: Dict[str, Any]) -> None:
    report_id = report_record.get('id') or report_record.get('report_id')
    logger.info('Processing report %s', report_id)
    try:
        pdf_bytes = download_pdf_from_supabase(report_record)
    except Exception as e:
        logger.exception('Failed to download PDF for report %s: %s', report_id, e)
        return

    # detect
    scanned = is_scanned_pdf(pdf_bytes)
    if not scanned:
        # try to extract digital text
        text = extract_text_from_pdf_bytes(pdf_bytes)
    else:
        text = do_ocr_on_pdf(pdf_bytes)

    # parse
    parsed = find_values_in_text(text)
    canonical = canonical_map(parsed)

    # fetch previous results if available for trend analysis
    previous = None
    if supabase:
        try:
            tbl = os.getenv('SUPABASE_TABLE', 'reports')
            prev_q = supabase.table(tbl).select('ai_results').eq('patient_id', report_record.get('patient_id')).order('created_at', desc=True).limit(1).execute()
            rows = prev_q.data if hasattr(prev_q, 'data') else prev_q
            if rows:
                previous = rows[0].get('ai_results')
        except Exception as e:
            logger.debug('Failed to fetch previous results: %s', e)

    trends = trend_analysis(canonical, previous)

    # route engine
    patient_meta = {'age': report_record.get('age'), 'sex': report_record.get('sex')}
    routes = route_engine_v4(canonical, patient_meta, previous)

    # build ai_results
    ai_results = {
        'canonical': canonical,
        'routes': routes,
        'trends': trends,
        'raw_text_excerpt': text[:4000],
        'scanned': scanned,
        'processed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    # add per-key decorated flags
    decorated = {}
    for key in canonical:
        val = canonical[key].get('value')
        # map severity for each key
        score = score_severity_for_abnormality(key, val, age_group_from_age(report_record.get('age')), report_record.get('sex', 'unknown'))
        cmap = COLOR_MAP.get(score, COLOR_MAP[1])
        decorated[key] = {
            'raw': canonical[key],
            'decorated': {
                'severity': score,
                'urgency': cmap['urgency'],
                'color': cmap['color'],
                'tw_class': cmap['tw']
            }
        }
    ai_results['decorated'] = decorated

    # save
    try:
        save_ai_results_to_supabase(report_id, ai_results)
    except Exception as e:
        logger.exception('Failed to save ai_results for report %s: %s', report_id, e)


def poll_and_process():
    if not supabase:
        logger.error('Supabase client not configured. Exiting poll loop.')
        return
    table = os.getenv('SUPABASE_TABLE', 'reports')
    logger.info('Starting poll loop (interval %ds) against table %s', POLL_INTERVAL, table)
    while True:
        try:
            res = supabase.table(table).select('*').eq('ai_status', 'pending').limit(10).execute()
            rows = res.data if hasattr(res, 'data') else res
            if rows:
                for r in rows:
                    try:
                        process_report(r)
                    except Exception as e:
                        logger.exception('Processing failed for record %s: %s', r.get('id'), e)
            else:
                logger.debug('No pending reports found.')
        except Exception as e:
            logger.exception('Polling error: %s', e)
        time.sleep(POLL_INTERVAL)


if __name__ == '__main__':
    # quick CLI: process a single local PDF for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-pdf', help='Path to PDF to process locally')
    parser.add_argument('--once', action='store_true', help='Poll once then exit')
    args = parser.parse_args()
    if args.test_pdf:
        with open(args.test_pdf, 'rb') as f:
            pdfb = f.read()
        dummy_report = {'id': 'local-test', 'age': 17, 'sex': 'female', 'patient_id': 'local-test-patient', 'pdf_url': None}
        # override download function for local file
        def download_pdf_from_supabase_override(rr):
            return pdfb
        globals()['download_pdf_from_supabase'] = download_pdf_from_supabase_override
        process_report(dummy_report)
        print('Test run complete.\n')
    else:
        if args.once:
            poll_and_process()
        else:
            poll_and_process()

