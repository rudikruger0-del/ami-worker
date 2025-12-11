# worker.py
"""
AMI Health Worker V4 (pypdf + pytesseract OCR only) - single-file worker.py

Notes:
- Uses pypdf (PdfReader) for digital PDFs.
- Uses pdf2image -> pytesseract for scanned PDFs.
- No OpenAI calls for OCR (option A chosen).
- Saves ai_results into Supabase table (ai_status -> 'completed').
- Run with: python worker.py --test-pdf sample.pdf
"""

import os
import io
import re
import time
import json
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps

# optional imports
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    pytesseract = None
    HAS_PYTESSERACT = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except Exception:
    create_client = None
    HAS_SUPABASE = False

# ---------- config & logging ----------
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_TABLE = os.getenv('SUPABASE_TABLE', 'reports')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '6'))
PDF_RENDER_DPI = int(os.getenv('PDF_RENDER_DPI', '200'))
TEXT_LENGTH_THRESHOLD = int(os.getenv('TEXT_LENGTH_THRESHOLD', '80'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ami-worker')

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    if not HAS_SUPABASE:
        logger.warning("supabase package not available - saving disabled.")
    else:
        logger.warning("SUPABASE_URL or SUPABASE_KEY missing - saving disabled.")

# ---------- PDF reading & scanned detection ----------

def download_pdf_from_record(record: Dict[str, Any]) -> bytes:
    """Download PDF binary. Supports 'pdf_url' or Supabase storage 'file_path'."""
    if 'pdf_url' in record and record['pdf_url']:
        import requests
        url = record['pdf_url']
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    elif 'file_path' in record and supabase:
        bucket = os.getenv('SUPABASE_BUCKET', 'reports')
        path = record['file_path']
        res = supabase.storage.from_(bucket).download(path)
        # supabase-python returns a StorageFileResponse-like; try .data or raw bytes
        if hasattr(res, 'data'):
            return res.data
        return res
    else:
        raise ValueError("No 'pdf_url' or 'file_path' in record, or supabase not configured.")


def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    """Extract text using pypdf (digital text)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as e:
        logger.debug("PdfReader failed: %s", e)
        return ''
    parts = []
    for i, p in enumerate(reader.pages):
        try:
            txt = p.extract_text() or ''
            parts.append(txt)
        except Exception as e:
            logger.debug("pypdf page extract failed %d: %s", i, e)
            parts.append('')
    joined = '\n'.join(parts)
    return joined


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    txt = extract_text_with_pypdf(pdf_bytes)
    if len(txt.strip()) < TEXT_LENGTH_THRESHOLD:
        logger.info("PDF appears scanned (text len %d)", len(txt))
        return True
    return False


# ---------- OCR via pytesseract ----------

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Simple preprocessing to improve OCR: convert to grayscale, increase contrast, median filter."""
    try:
        # convert to L (grayscale)
        img = img.convert('L')
        # increase contrast if needed
        img = ImageOps.autocontrast(img, cutoff=2)
        # slight sharpen and reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        # optionally resize small images up
        w, h = img.size
        if max(w, h) < 1200:
            factor = int(1200 / max(1, max(w, h)))
            new_size = (w * factor, h * factor)
            img = img.resize(new_size, Image.LANCZOS)
        return img
    except Exception:
        return img


def ocr_image_pytesseract(img: Image.Image, lang: str = 'eng') -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract is not available in this environment.")
    img2 = preprocess_image_for_ocr(img)
    # Use OEM/LSTM and psm modes if desired
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img2, lang=lang, config=custom_oem_psm_config)
    return text


def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
    return imgs


def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    """Render scanned PDF to images and OCR each page with pytesseract, returning concatenated text."""
    try:
        images = pdf_to_images(pdf_bytes)
    except Exception as e:
        logger.exception("Failed to render PDF to images: %s", e)
        raise

    texts = []
    for i, img in enumerate(images):
        page_text = ''
        try:
            page_text = ocr_image_pytesseract(img)
            logger.info("OCR page %d length %d", i, len(page_text))
        except Exception as e:
            logger.exception("pytesseract failed on page %d: %s", i, e)
            page_text = ''
        texts.append(page_text)
    return '\n\n---PAGE_BREAK---\n\n'.join(texts)


# ---------- Parsing: extract lab values robustly ----------

# pattern search helpers
VALUE_RE = r'(-?\d+\.\d+|-?\d+)'
UNIT_RE = r'([a-zA-Z/%\^\d\-\s\.\u00b3\u00b2]*)'  # extended to include superscript chars
REF_RE = r'(?:\(?(?:ref|reference|range)[:\s]*([^\)]*)\)?)?'

COMMON_KEYS = {
    'hb': ['hb', 'haemoglobin', 'hemoglobin'],
    'rbc': ['rbc', 'erythrocyte count', 'erythrocyte'],
    'hct': ['hct', 'haematocrit', 'hematocrit'],
    'mcv': ['mcv', 'mean corpuscular volume'],
    'mch': ['mch', 'mean corpuscular haemoglobin', 'mean corpuscular hemoglobin'],
    'mchc': ['mchc'],
    'rdw': ['rdw'],
    'wbc': ['wbc', 'white cell count', 'leukocyte', 'leucocyte', 'leukocytes'],
    'neutrophils_pc': ['neutrophils %', 'neutrophils%','neutrophils percent','neutrophil%','neutrophils'],
    'neutrophils_abs': ['neutrophils absolute','neutrophil count','neutrophil absolute'],
    'lymphocytes_pc': ['lymphocytes %','lymphocytes'],
    'monocytes_pc': ['monocytes %','monocytes'],
    'eosinophils_pc': ['eosinophils %','eosinophils'],
    'basophils_pc': ['basophils %','basophils'],
    'platelets': ['platelets','thrombocytes','platelet count'],
    'crp': ['crp','c-reactive protein','c reactive protein'],
    'creatinine': ['creatinine','creat'],
    'sodium': ['sodium','na'],
    'potassium': ['potassium','k'],
    'chloride': ['chloride','cl'],
    'urea': ['urea','bun'],
    'alt': ['alt','alanine aminotransferase'],
    'ast': ['ast','aspartate aminotransferase'],
    'ck': ['ck','creatine kinase'],
}

LABEL_TO_KEY = {}
for k, labels in COMMON_KEYS.items():
    for l in labels:
        LABEL_TO_KEY[l.lower()] = k


def find_key_for_label(label: str) -> Optional[str]:
    l = re.sub(r'[^a-z0-9 ]', '', label.lower()).strip()
    if l in LABEL_TO_KEY:
        return LABEL_TO_KEY[l]
    for lab, key in LABEL_TO_KEY.items():
        if lab in l or l in lab:
            return key
    return None


def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    # normalize whitespace; split lines
    lines = [ln.strip() for ln in re.split(r'\n|\r', text) if ln.strip()]
    # first pass: direct labelled matches
    for line in lines:
        ln = line.lower()
        for label, key in LABEL_TO_KEY.items():
            if label in ln:
                # search for numeric near label
                # allow formats like "Hb 11.6 g/dL (ref: 12.4-16.7)" or "HB: 11.6"
                # build generic pattern allowing optional units and refs
                try:
                    pat = re.compile(rf'{re.escape(label)}[^\d\n\r\-]{{0,25}}({VALUE_RE})', flags=re.IGNORECASE)
                    m = pat.search(line)
                    if m:
                        val = float(m.group(1))
                        unit_m = re.search(rf'{re.escape(label)}[^\d\n\r]*{VALUE_RE}\s*({UNIT_RE})', line, flags=re.IGNORECASE)
                        unit = unit_m.group(1).strip() if unit_m else None
                        results.setdefault(key, {})['value'] = val
                        if unit:
                            results.setdefault(key, {})['unit'] = unit
                        results.setdefault(key, {})['raw_line'] = line
                        continue
                except Exception:
                    pass
        # generic label + value extraction fallback
        gen = re.findall(rf'([A-Za-z\-/ ]{{2,30}})\s*[:\-]?\s*{VALUE_RE}\s*{UNIT_RE}', line)
        for g in gen:
            label_raw = g[0].strip()
            val_s = g[1]
            unit = g[2].strip() if g[2] else None
            key = find_key_for_label(label_raw)
            if key:
                try:
                    val = float(val_s)
                    results.setdefault(key, {})['value'] = val
                    if unit:
                        results.setdefault(key, {})['unit'] = unit
                    results.setdefault(key, {})['raw_line'] = line
                except:
                    continue

    # second pass: special handling for percentages and neutrophil absolute
    for line in lines:
        ln = line.lower()
        if 'neutrophil' in ln:
            p = re.search(r'([0-9]{1,3}\.\d+|[0-9]{1,3})\s*%+', line)
            if p:
                try:
                    results.setdefault('neutrophils_pc', {})['value'] = float(p.group(1))
                except:
                    pass
            # absolute like "17.08 x10^9/L" or "17.08 x10^9/L"
            a = re.search(r'([0-9]{1,3}\.\d+)\s*x\s*10\^?\d?\/?l', line, flags=re.IGNORECASE)
            if a:
                try:
                    results.setdefault('neutrophils_abs', {})['value'] = float(a.group(1))
                except:
                    pass
            # simpler absolute numeric
            a2 = re.search(r'([0-9]{1,3}\.\d+)\s*/?L', line, flags=re.IGNORECASE)
            if a2 and 'neutrophil' in ln and 'x' not in line:
                try:
                    results.setdefault('neutrophils_abs', {})['value'] = float(a2.group(1))
                except:
                    pass

    # fallback: search for Hb explicitly anywhere
    fh = re.findall(r'\b(hb|haemoglobin|haemoglobin)\b[^\d\n\r]{0,20}('+VALUE_RE+')', text, flags=re.IGNORECASE)
    for m in fh:
        try:
            results.setdefault('hb', {})['value'] = float(m[1])
        except:
            pass

    return results


# ---------- Canonical mapping ----------

CANONICAL_KEYS = ['Hb', 'MCV', 'MCH', 'MCHC', 'RDW', 'WBC', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils', 'NLR', 'Platelets', 'Creatinine', 'CRP', 'Sodium', 'Potassium', 'Chloride', 'Urea', 'RBC', 'HCT', 'ALT', 'AST', 'CK']

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
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
        'alt': 'ALT',
        'ast': 'AST',
        'ck': 'CK',
    }
    for k, v in parsed.items():
        if k in map_rules:
            canon = map_rules[k]
            out.setdefault(canon, {})
            try:
                out[canon]['value'] = float(v.get('value'))
            except Exception:
                out[canon]['value'] = None
            if v.get('unit'):
                out[canon]['unit'] = v.get('unit')
            if v.get('raw_line'):
                out[canon]['raw'] = v.get('raw_line')
            if v.get('ref'):
                out[canon]['ref'] = v.get('ref')

    # compute NLR if percentages or absolute provided
    try:
        n = out.get('Neutrophils', {}).get('value')
        l = out.get('Lymphocytes', {}).get('value')
        if n is not None and l is not None and l != 0:
            out['NLR'] = {'value': round(float(n) / float(l), 2)}
    except Exception:
        pass

    return out


# ---------- Route Engine V4 (rules, routes, ddx, severity) ----------

# severity color map: 1 green, 2 yellow, 3 orange, 4 red, 5 deep red
COLOR_MAP = {
    5: {'label': 'critical', 'color': '#b91c1c', 'tw': 'bg-red-700', 'urgency': 'high'},
    4: {'label': 'high', 'color': '#ef4444', 'tw': 'bg-red-500', 'urgency': 'high'},
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


def score_severity_for_abnormality(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    """Heuristic severity scoring."""
    if value is None:
        return 1
    try:
        v = float(value)
    except:
        return 1

    score = 1
    key_l = key.lower()
    if key_l == 'hb':
        # sex-aware thresholds
        low_cut = 12.0 if sex.lower() == 'female' else 13.0
        if age_group in ['neonate','infant']:
            low_cut = 14.0
        if v < low_cut - 4:
            score = 5
        elif v < low_cut - 2:
            score = 4
        elif v < low_cut:
            score = 3
    elif key_l == 'wbc':
        if v > 30:
            score = 5
        elif v > 20:
            score = 4
        elif v > 11:
            score = 3
    elif key_l == 'crp':
        if v > 200:
            score = 5
        elif v > 100:
            score = 4
        elif v > 50:
            score = 3
        elif v > 10:
            score = 2
    elif key_l == 'neutrophils' or key_l == 'nlr':
        # high NLR or neutrophilia -> significant
        if v > 10:
            score = 5
        elif v > 6:
            score = 4
        elif v > 3:
            score = 3
    elif key_l == 'creatinine':
        if v > 400:
            score = 5
        elif v > 200:
            score = 4
        elif v > 120:
            score = 3
    elif key_l == 'platelets':
        if v < 10:
            score = 5
        elif v < 30:
            score = 4
        elif v < 100:
            score = 3
    elif key_l in ['sodium', 'potassium']:
        # severe electrolyte derangement
        if key_l == 'sodium':
            if v < 120 or v > 160:
                score = 5
            elif v < 125 or v > 155:
                score = 4
            elif v < 130 or v > 150:
                score = 3
        if key_l == 'potassium':
            if v < 2.8 or v > 6.5:
                score = 5
            elif v < 3.2 or v > 6.0:
                score = 4
            elif v < 3.5 or v > 5.5:
                score = 3
    else:
        score = 1
    return score


def route_engine_v4(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    age = patient_meta.get('age')
    sex = patient_meta.get('sex', 'unknown')
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    severity_scores = []

    def add_find(name, reason, score):
        patterns.append({'pattern': name, 'reason': reason})
        severity_scores.append(score)

    # pull canonical values
    hb = canonical.get('Hb', {}).get('value')
    mcv = canonical.get('MCV', {}).get('value')
    mch = canonical.get('MCH', {}).get('value')
    rdw = canonical.get('RDW', {}).get('value')
    wbc = canonical.get('WBC', {}).get('value')
    crp = canonical.get('CRP', {}).get('value')
    neut = canonical.get('Neutrophils', {}).get('value')
    nlr = canonical.get('NLR', {}).get('value')
    plate = canonical.get('Platelets', {}).get('value')
    creat = canonical.get('Creatinine', {}).get('value')
    sodium = canonical.get('Sodium', {}).get('value')
    alt = canonical.get('ALT', {}).get('value')
    ast = canonical.get('AST', {}).get('value')
    ck = canonical.get('CK', {}).get('value')

    # ANEMIA
    if hb is not None:
        hb_score = score_severity_for_abnormality('Hb', hb, ag, sex)
        if hb_score > 1:
            add_find('anemia', f'Hb {hb} g/dL', hb_score)
            # microcytic
            if mcv is not None and mcv < 80:
                add_find('microcytic anemia', f'MCV {mcv} fL', max(3, hb_score))
                routes.append('Iron deficiency route (Ferritin + reticulocyte)')
                ddx.extend(['Iron deficiency anemia', 'Thalassaemia trait', 'Chronic blood loss'])
                next_steps.append('Order ferritin, reticulocyte count, and peripheral smear. Consider stool occult blood if adult.')
            elif mcv is not None and mcv > 100:
                add_find('macrocytic anemia', f'MCV {mcv} fL', max(3, hb_score))
                routes.append('Macrocytic route (B12/folate)')
                ddx.extend(['Vitamin B12 deficiency', 'Folate deficiency', 'Alcohol-related'])
                next_steps.append('Order B12 and folate; review meds.')
            else:
                add_find('normocytic anemia', 'MCV normal or missing', max(2, hb_score))
                routes.append('Normocytic anemia route')
                ddx.extend(['Acute blood loss', 'Hemolysis', 'Anaemia of inflammation'])
                next_steps.append('Order reticulocyte count, LDH, peripheral smear.')

    # INFECTION / INFLAMMATION
    if wbc is not None and wbc > 11:
        wbc_score = score_severity_for_abnormality('WBC', wbc, ag, sex)
        add_find('leukocytosis', f'WBC {wbc} x10^9/L', wbc_score)
        if neut is not None and neut > 70:
            add_find('neutrophilic predominance', f'Neutrophils {neut}%', max(3, wbc_score))
            routes.append('Bacterial infection / Sepsis route')
            ddx.extend(['Bacterial infection', 'Sepsis', 'Acute inflammation'])
            next_steps.append('Clinical assessment for sepsis; consider blood cultures, IV fluids, empiric antibiotics if unstable.')

    if crp is not None and crp > 10:
        crp_score = score_severity_for_abnormality('CRP', crp, ag, sex)
        add_find('elevated CRP', f'CRP {crp} mg/L', crp_score)
        if crp > 50:
            routes.append('Significant inflammatory response')
            ddx.extend(['Severe infection', 'Inflammatory disease'])
            next_steps.append('Consider urgent review; blood cultures if febrile; procalcitonin if available.')

    # NLR
    if nlr is not None:
        if nlr > 10:
            add_find('very high NLR', f'NLR {nlr}', 5)
            routes.append('High NLR / Sepsis route')
            next_steps.append('Urgent clinical review for sepsis; consider admission and sepsis pathway.')
        elif nlr > 5:
            add_find('high NLR', f'NLR {nlr}', 4)
            routes.append('High NLR route')
            next_steps.append('Assess severity and source of infection.')

    # PLATELETS
    if plate is not None:
        pscore = score_severity_for_abnormality('Platelets', plate, ag, sex)
        if plate < 150:
            add_find('thrombocytopenia', f'Platelets {plate}', pscore)
            ddx.extend(['Immune thrombocytopenia', 'DIC', 'Bone marrow suppression'])
            next_steps.append('Repeat platelet count; check smear; assess bleeding.')
        elif plate > 450:
            add_find('thrombocytosis', f'Platelets {plate}', max(2, pscore))
            next_steps.append('Consider reactive thrombocytosis; repeat and investigate inflammation/iron status.')

    # RENAL / AKI
    if creat is not None:
        cscore = score_severity_for_abnormality('Creatinine', creat, ag, sex)
        if cscore >= 3:
            add_find('elevated creatinine', f'Creatinine {creat} umol/L', cscore)
            routes.append('AKI route')
            ddx.extend(['Acute kidney injury', 'Chronic kidney disease'])
            next_steps.append('Repeat creatinine urgently; check urine output and electrolytes; review meds.')

    # ELECTROLYTE & RHabdo
    if ck is not None and ck > 1000:
        add_find('rhabdomyolysis signal', f'CK {ck}', 4)
        routes.append('Rhabdomyolysis route')
        next_steps.append('Assess for muscle pain/urine colour; check creatinine and electrolytes; consider urgent fluids.')

    # LIVER indicators
    if (alt is not None and alt > 200) or (ast is not None and ast > 200):
        add_find('transaminitis', f'ALT {alt} AST {ast}', 3)
        routes.append('Hepatic route')
        next_steps.append('Review hepatotoxins, viral hepatitis risk; consider LFT panel.')

    # COMBINED PATTERNS
    if hb is not None and crp is not None and wbc is not None:
        if hb < 12 and crp > 20 and wbc > 11:
            add_find('anemia with inflammatory/infective response', 'Low Hb + high CRP + leukocytosis', 4)
            routes.append('Infection with anemia route')
            next_steps.append('Treat source of infection; reassess Hb after control; do ferritin when CRP falls.')

    # AGE / SEX modifiers
    if ag == 'teen' and sex.lower() == 'female':
        # teenage female — high likelihood of menstrual blood loss causing microcytic anemia
        next_steps.append('Assess menstrual history; consider urgent ferritin and reticulocyte count.')

    combined_score = max(severity_scores) if severity_scores else 1
    color_entry = COLOR_MAP.get(combined_score, COLOR_MAP[1])
    urgency = color_entry['urgency']

    # build human-friendly summary
    summary_lines = []
    if patterns:
        summary_lines.append('Patterns: ' + '; '.join([p['pattern'] for p in patterns]))
    if routes:
        summary_lines.append('Primary routes: ' + '; '.join(routes))
    if ddx:
        summary_lines.append('Top differential diagnoses: ' + ', '.join(dict.fromkeys(ddx)))
    if next_steps:
        summary_lines.append('Immediate suggested actions: ' + ' | '.join(next_steps))

    age_note = ''
    if ag == 'teen' and sex.lower() == 'female':
        age_note = 'Teenage female — consider menstrual blood loss and iron deficiency.'
    elif ag == 'elderly':
        age_note = 'Elderly – broaden differential for chronic disease and malignancy.'

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
        'summary': '\n'.join(summary_lines[:10]) if summary_lines else 'No significant abnormalities detected.'
    }
    return final


# ---------- Trend analysis ----------

def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {'trend': 'no_previous'}
    diffs = {}
    for k, v in current.items():
        prev_val = None
        if isinstance(previous, dict):
            prev_val = previous.get(k, {}).get('value')
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
        logger.warning("Supabase client not available; skipping save.")
        return
    try:
        payload = {'ai_status': 'completed', 'ai_results': ai_results, 'ai_error': None}
        res = supabase.table(SUPABASE_TABLE).update(payload).eq('id', report_id).execute()
        logger.info("Saved ai_results for report %s", report_id)
    except Exception as e:
        logger.exception("Failed to save ai_results: %s", e)


# ---------- Full processing ----------

def process_report(record: Dict[str, Any]) -> None:
    report_id = record.get('id') or record.get('report_id') or '<unknown>'
    logger.info("Processing report %s", report_id)
    try:
        pdf_bytes = download_pdf_from_record(record)
    except Exception as e:
        logger.exception("Failed to download PDF: %s", e)
        if supabase:
            supabase.table(SUPABASE_TABLE).update({'ai_status': 'failed', 'ai_error': str(e)}).eq('id', report_id).execute()
        return

    scanned = is_scanned_pdf(pdf_bytes)
    if not scanned:
        text = extract_text_with_pypdf(pdf_bytes)
    else:
        if not HAS_PYTESSERACT:
            err = "Detected scanned PDF but pytesseract not available."
            logger.error(err)
            if supabase:
                supabase.table(SUPABASE_TABLE).update({'ai_status': 'failed', 'ai_error': err}).eq('id', report_id).execute()
            return
        text = do_ocr_on_pdf(pdf_bytes)

    # parse
    parsed = find_values_in_text(text)
    canonical = canonical_map(parsed)

    # fetch previous results (latest by patient_id) for trend
    previous = None
    if supabase:
        try:
            pid = record.get('patient_id')
            if pid:
                prev_q = supabase.table(SUPABASE_TABLE).select('ai_results').eq('patient_id', pid).order('created_at', desc=True).limit(1).execute()
                rows = prev_q.data if hasattr(prev_q, 'data') else prev_q
                if rows:
                    previous = rows[0].get('ai_results')
        except Exception as e:
            logger.debug("Failed fetch previous: %s", e)

    trends = trend_analysis(canonical, previous)

    patient_meta = {'age': record.get('age'), 'sex': record.get('sex', 'unknown')}
    routes = route_engine_v4(canonical, patient_meta, previous)

    ai_results = {
        'canonical': canonical,
        'parsed': parsed,
        'routes': routes,
        'trends': trends,
        'raw_text_excerpt': text[:5000],
        'scanned': scanned,
        'processed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    # add per-key decorated flags
    decorated = {}
    for key, val in canonical.items():
        value = val.get('value')
        score = score_severity_for_abnormality(key, value, age_group_from_age(record.get('age')), record.get('sex', 'unknown'))
        cmap = COLOR_MAP.get(score, COLOR_MAP[1])
        decorated[key] = {
            'raw': val,
            'decorated': {
                'severity': score,
                'urgency': cmap['urgency'],
                'color': cmap['color'],
                'tw_class': cmap['tw']
            }
        }
    ai_results['decorated'] = decorated

    try:
        save_ai_results_to_supabase(report_id, ai_results)
    except Exception as e:
        logger.exception("Failed to save results: %s", e)


# ---------- Poll loop ----------

def poll_and_process():
    if not supabase:
        logger.error("Supabase client not configured. Exiting.")
        return
    logger.info("Starting poll loop (interval %ds) table=%s", POLL_INTERVAL, SUPABASE_TABLE)
    while True:
        try:
            res = supabase.table(SUPABASE_TABLE).select('*').eq('ai_status', 'pending').limit(10).execute()
            rows = res.data if hasattr(res, 'data') else res
            if rows:
                for r in rows:
                    try:
                        # mark processing
                        supabase.table(SUPABASE_TABLE).update({'ai_status': 'processing'}).eq('id', r.get('id')).execute()
                        process_report(r)
                    except Exception as e:
                        logger.exception("Record processing failed %s: %s", r.get('id'), e)
                        try:
                            supabase.table(SUPABASE_TABLE).update({'ai_status': 'failed', 'ai_error': str(e)}).eq('id', r.get('id')).execute()
                        except:
                            pass
            else:
                logger.debug("No pending reports.")
        except Exception as e:
            logger.exception("Polling error: %s", e)
        time.sleep(POLL_INTERVAL)


# ---------- CLI / test ----------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="AMI Health Worker V4 (pypdf + pytesseract)")
    parser.add_argument('--test-pdf', help='Path to local PDF for testing')
    parser.add_argument('--once', action='store_true', help='Poll once then exit (if not test)')
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, 'rb') as f:
            pdfb = f.read()
        dummy = {
            'id': 'local-test',
            'patient_id': 'local-patient',
            'age': 17,
            'sex': 'female',
            'file_path': args.test_pdf,
        }
        # override downloader
        def dl_override(rec):
            return pdfb
        globals()['download_pdf_from_record'] = dl_override
        process_report(dummy)
        print("Test run complete.")
    else:
        if args.once:
            poll_and_process()
        else:
            poll_and_process()
