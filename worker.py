# worker.py
"""
AMI Health Worker V5 - single-file worker.py

Adds:
- Route Engine V5 with: explicit Ferritin route, stronger microcytic algorithm,
  sepsis detection, rhabdomyolysis route, renal route, platelet routes, viral vs bacterial hints,
  urgency color map (Green/Yellow/Orange/Red), and clearer summary/impression/suggested_follow_up fields.
- OCR: pypdf for digital PDFs, pdf2image + pytesseract preferred for scanned PDFs, OpenAI Vision fallback.
- Canonical mapping & decorated flags for frontend.
- Trend analysis if previous ai_results exist for same patient_id.
- Saves ai_results to Supabase table (env: SUPABASE_TABLE)
- Polls for reports with ai_status == 'pending'
"""

import os
import io
import re
import time
import json
import base64
import logging
import traceback
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

# optional libraries
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

# ---------- config ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "12000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v5")

if HAS_OPENAI and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    logger.warning("Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY in env.")

# ---------- helpers: PDF extraction ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n\n".join(pages).strip()
        return text
    except Exception as e:
        logger.debug("pypdf failed: %s", e)
        return ""

def is_scanned_pdf_by_text(pdf_bytes: bytes) -> bool:
    txt = extract_text_from_pdf_bytes(pdf_bytes)
    return len(txt.strip()) < TEXT_LENGTH_THRESHOLD

def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI):
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        return imgs
    except Exception as e:
        logger.exception("pdf2image convert failed: %s", e)
        raise

# ---------- OCR functions ----------
def ocr_image_with_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract not installed")
    gray = img.convert("L")
    w, h = gray.size
    if max(w, h) < 1600:
        factor = max(1, int(1600 / max(w, h)))
        gray = gray.resize((w * factor, h * factor), Image.LANCZOS)
    try:
        txt = pytesseract.image_to_string(gray, lang="eng")
        return txt
    except Exception as e:
        logger.exception("pytesseract error: %s", e)
        return ""

def openai_vision_ocr_image(img: Image.Image) -> str:
    """
    Fallback OCR using OpenAI chat vision approach (conceptual).
    This function will base64-encode JPEG and call a chat completion that accepts image text.
    Implementation may need adjustment to match your OpenAI client capabilities.
    """
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI not configured")
    # resize moderately
    max_w = 1200
    w, h = img.size
    if w > max_w:
        ratio = max_w / float(w)
        img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
    buff = io.BytesIO()
    img.convert("RGB").save(buff, format="JPEG", quality=80)
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant. Extract only the plain text from the provided base64 JPEG. "
        "Return the extracted text only, no commentary."
    )
    # use the Chat Completions style; adapt if your OpenAI client uses different methods
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"IMAGE_BASE64:\ndata:image/jpeg;base64,{b64}"}
            ],
            temperature=0,
            max_tokens=3000
        )
        content = resp.choices[0].message.content
        return content or ""
    except Exception as e:
        logger.exception("OpenAI OCR failed: %s", e)
        return ""

def do_ocr_on_pdf_bytes(pdf_bytes: bytes) -> str:
    images = pdf_to_images(pdf_bytes)
    texts = []
    for i, img in enumerate(images):
        page_text = ""
        if HAS_PYTESSERACT:
            try:
                page_text = ocr_image_with_pytesseract(img)
                logger.info("pytesseract page %d length %d", i, len(page_text))
            except Exception as e:
                logger.warning("pytesseract failed on page %d: %s", i, e)
                page_text = ""
        if not page_text:
            try:
                page_text = openai_vision_ocr_image(img)
                logger.info("OpenAI OCR page %d length %d", i, len(page_text))
            except Exception as e:
                logger.exception("Vision OCR failed page %d: %s", i, e)
                page_text = ""
        texts.append(page_text)
    return "\n\n---PAGE---\n\n".join(texts)

# ---------- parsing ----------

VALUE_RE = r'(-?\d+(?:\.\d+)?)'
UNIT_RE = r'([a-zA-Z/%\^0-9\-\sµμ]+)?'
REF_RE = r'(?:ref[:]?\s*\(?([\d\.\-–to,\/\s]+)\)?)?'

COMMON_KEYS = {
    'hb': ['hb', 'haemoglobin', 'hemoglobin'],
    'rbc': ['rbc', 'erythrocyte', 'erythrocyte count', 'erythrocyte count'],
    'hct': ['hct', 'haematocrit', 'hematocrit'],
    'mcv': ['mcv', 'mean corpuscular volume'],
    'mch': ['mch', 'mean corpuscular haemoglobin', 'mean corpuscular hemoglobin'],
    'mchc': ['mchc'],
    'rdw': ['rdw'],
    'wbc': ['wbc', 'white cell count', 'leucocyte', 'leukocyte'],
    'neutrophils_pc': ['neutrophils %', 'neutrophils', 'neutrophil%'],
    'neutrophils_abs': ['neutrophil absolute', 'neutrophils x10', 'neutrophil count'],
    'lymphocytes_pc': ['lymphocytes %', 'lymphocytes'],
    'monocytes_pc': ['monocytes %', 'monocytes'],
    'eosinophils_pc': ['eosinophils %', 'eosinophils'],
    'basophils_pc': ['basophils %', 'basophils'],
    'platelets': ['platelets', 'thrombocytes', 'platelet count'],
    'crp': ['crp', 'c-reactive protein'],
    'creatinine': ['creatinine'],
    'sodium': ['sodium', 'na'],
    'potassium': ['potassium', 'k'],
    'chloride': ['chloride', 'cl'],
    'urea': ['urea'],
    'alt': ['alt', 'alanine aminotransferase'],
    'ast': ['ast', 'aspartate aminotransferase'],
    'ck': ['ck', 'creatine kinase', 'creatine kinase total'],
    'ckmb': ['ck-mb', 'ck mb', 'ckmb'],
    'ferritin': ['ferritin'],
    # add others as needed
}

LABEL_TO_KEY = {}
for k, labels in COMMON_KEYS.items():
    for l in labels:
        LABEL_TO_KEY[l.lower()] = k

def find_key_for_label(label: str) -> Optional[str]:
    l = re.sub(r'[^a-z0-9 ]', '', label.lower()).strip()
    if not l:
        return None
    if l in LABEL_TO_KEY:
        return LABEL_TO_KEY[l]
    # partial matching
    for lab, key in LABEL_TO_KEY.items():
        if lab in l or l in lab:
            return key
    return None

# robust extraction scanning lines
def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    # split by lines and also by common delimiters
    lines = [ln.strip() for ln in re.split(r'\n|\r', text) if ln.strip()]
    # first pass: label + number in same line
    for line in lines:
        ln = line.lower()
        # normalized for units capture
        for label_raw, key in list(LABEL_TO_KEY.items()):
            if label_raw in ln:
                # find number after label
                # look for patterns like 'hb 11.6 g/dl (ref: 12.4-16.7)'
                m = re.search(rf'{re.escape(label_raw)}[^\d\-\+\/\n\r]{{0,25}}{VALUE_RE}', ln)
                if m:
                    try:
                        val = float(m.group(1))
                    except:
                        continue
                    # try units from original line (case preserved)
                    um = re.search(rf'{re.escape(label_raw)}[^\n]*?([a-zA-Z/%\^0-9\-\sµμ]+)\b', line, re.IGNORECASE)
                    unit = um.group(1).strip() if um else None
                    # try ref range
                    ref_m = re.search(r'\(ref[: ]*([^\)]+)\)', line, re.IGNORECASE)
                    ref = ref_m.group(1).strip() if ref_m else None
                    results.setdefault(key, {})['value'] = val
                    if unit:
                        results[key]['unit'] = unit
                    if ref:
                        results[key]['ref'] = ref
    # second pass: generic 'HB: 11.6 g/dL (12.4-16.7)' style
    generic_matches = re.findall(rf'([A-Za-z\/\-\s]{{2,30}})[:\s]*{VALUE_RE}\s*{UNIT_RE}\s*(?:\(ref[:\s]*([^\)]+)\))?', text, flags=re.IGNORECASE)
    for gm in generic_matches:
        label_raw = gm[0].strip()
        val = gm[1]
        unit = gm[2].strip() if gm[2] else None
        ref = gm[3].strip() if gm[3] else None
        key = find_key_for_label(label_raw)
        if key:
            try:
                results.setdefault(key, {})['value'] = float(val)
                if unit:
                    results[key]['unit'] = unit
                if ref:
                    results[key]['ref'] = ref
            except:
                pass
    # percentages and absolute neutrophils
    for line in lines:
        if 'neutrophil' in line.lower():
            p = re.search(r'([0-9]{1,3}(?:\.\d+)?)\s*%', line)
            if p:
                try:
                    results.setdefault('neutrophils_pc', {})['value'] = float(p.group(1))
                except:
                    pass
            abs_m = re.search(r'([0-9]{1,3}\.\d+)\s*x10\^?\d*\/?l', line.lower())
            if abs_m:
                try:
                    results.setdefault('neutrophils_abs', {})['value'] = float(abs_m.group(1))
                except:
                    pass
    # fallback simple hb
    fh = re.search(r'\b(hb|haemoglobin)\b[^\d\n\r]{0,12}'+VALUE_RE, text, flags=re.IGNORECASE)
    if fh:
        try:
            results.setdefault('hb', {})['value'] = float(fh.group(2))
        except:
            pass
    return results

# ---------- canonical mapping ----------
CANONICAL_KEYS = ['Hb','RBC','HCT','MCV','MCH','MCHC','RDW','WBC','Neutrophils','Lymphocytes','Monocytes','Eosinophils','Basophils','NLR','Platelets','Creatinine','CRP','Sodium','Potassium','Chloride','Urea','ALT','AST','CK','CKMB','Ferritin']

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    map_rules = {
        'hb': 'Hb',
        'rbc': 'RBC',
        'hct': 'HCT',
        'mcv': 'MCV',
        'mch': 'MCH',
        'mchc': 'MCHC',
        'rdw': 'RDW',
        'wbc': 'WBC',
        'neutrophils_pc': 'Neutrophils',
        'neutrophils_abs': 'Neutrophils',  # prefer percent if both
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
        'alt': 'ALT',
        'ast': 'AST',
        'ck': 'CK',
        'ckmb': 'CKMB',
        'ferritin': 'Ferritin',
    }
    for k, v in parsed.items():
        if k in map_rules:
            canon = map_rules[k]
            out.setdefault(canon, {})
            try:
                out[canon]['value'] = float(v.get('value')) if v.get('value') is not None else None
            except:
                out[canon]['value'] = None
            if v.get('unit'):
                out[canon]['units'] = v.get('unit')
            if v.get('ref'):
                # try parse simple ref like "12.4-16.7"
                ref_text = v.get('ref')
                out[canon]['ref'] = ref_text
            out[canon]['raw'] = v.get('raw_line') or v
    # compute NLR when possible
    try:
        if 'Neutrophils' in out and 'Lymphocytes' in out:
            n = float(out['Neutrophils'].get('value') or 0)
            l = float(out['Lymphocytes'].get('value') or 0)
            if l > 0:
                out['NLR'] = {'value': round(n / l, 2)}
    except Exception:
        pass
    return out

# ---------- utility helpers ----------
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

# color map with more granularity
COLOR_MAP = {
    5: {'label':'critical','color':'#b91c1c','tw':'bg-red-600','urgency':'high'},   # red
    4: {'label':'high','color':'#f97316','tw':'bg-orange-500','urgency':'high'},    # orange
    3: {'label':'medium','color':'#f59e0b','tw':'bg-yellow-400','urgency':'medium'},# yellow
    2: {'label':'borderline','color':'#facc15','tw':'bg-yellow-300','urgency':'low'},# light yellow
    1: {'label':'normal','color':'#10b981','tw':'bg-green-500','urgency':'low'}     # green
}

# ---------- Route Engine V5 ----------
def score_severity_for_abnormality(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    """Return 1..5 severity score for a single analyte"""
    if value is None:
        return 1
    try:
        v = float(value)
    except:
        return 1
    # Hb severity by sex/age (approx)
    if key == 'Hb':
        if sex and sex.lower() == 'female':
            low_cut = 12.0
        else:
            low_cut = 13.0
        if v < low_cut - 4:
            return 5
        if v < low_cut - 2:
            return 4
        if v < low_cut:
            return 3
        return 1
    if key == 'WBC':
        if v > 30:
            return 5
        if v > 20:
            return 4
        if v > 12:
            return 3
        return 1
    if key == 'CRP':
        if v > 200:
            return 5
        if v > 100:
            return 4
        if v > 50:
            return 3
        if v > 10:
            return 2
        return 1
    if key == 'Creatinine':
        if v > 354:
            return 5
        if v > 200:
            return 4
        if v > 120:
            return 3
        return 1
    if key == 'Platelets':
        if v < 10:
            return 5
        if v < 50:
            return 4
        if v < 100:
            return 3
        return 1
    if key == 'Potassium':
        if v < 2.8 or v > 6.0:
            return 5
        if v < 3.2 or v > 5.5:
            return 4
        if v < 3.5 or v > 5.1:
            return 3
        return 1
    if key == 'CK':
        if v > 50000:
            return 5
        if v > 20000:
            return 4
        if v > 5000:
            return 3
        return 1
    # default
    return 1

def route_engine_v5(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produces:
    - patterns (list of dicts{name, reason})
    - routes (list of route names)
    - next_steps (list)
    - differential (list)
    - severity_score (1-5)
    - urgency_flag (low/medium/high)
    - color, tw_class
    - age_group, age_note
    - compact summary (ER-friendly)
    - impression & suggested_follow_up (legacy-friendly fields)
    """
    age = patient_meta.get('age')
    sex = (patient_meta.get('sex') or 'unknown')
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    severity_scores = []

    def add_pattern(name, reason, score):
        patterns.append({'pattern': name, 'reason': reason})
        if score:
            severity_scores.append(score)

    # helper values
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
    ferr = canonical.get('Ferritin', {}).get('value') if canonical.get('Ferritin') else None
    ck = canonical.get('CK', {}).get('value')
    potassium = canonical.get('Potassium', {}).get('value')

    # ----- Anemia and microcytic logic -----
    if hb is not None:
        hb_score = score_severity_for_abnormality('Hb', hb, ag, sex)
        if hb_score > 1:
            add_pattern('anemia', f'Hb {hb} g/dL (age_group={ag}, sex={sex})', hb_score)
            # Microcytic detection
            if mcv is not None:
                if mcv < 75:
                    add_pattern('microcytic anemia (marked)', f'MCV {mcv} fL', max(3, hb_score))
                    routes.append('Iron deficiency evaluation')
                    ddx.extend(['Iron deficiency anemia', 'Thalassaemia trait', 'Chronic blood loss'])
                    # ferritin guidance
                    if ferr is not None:
                        if ferr < 30:
                            next_steps.append('Ferritin low — consistent with iron deficiency; consider iron replacement and GI/menstrual source evaluation')
                        else:
                            next_steps.append('Ferritin non-low — consider thalassaemia trait or inflammation; review MCV/RDW and consider haemoglobinopathy testing')
                    else:
                        next_steps.append('Order ferritin and reticulocyte count to differentiate iron deficiency vs thalassaemia')
                elif 75 <= mcv <= 85:
                    # borderline microcytic
                    add_pattern('microcytic anemia (mild)', f'MCV {mcv} fL', max(2, hb_score))
                    routes.append('Iron studies route')
                    next_steps.append('Order ferritin, CBC indices, and reticulocyte count; assess for chronic blood loss')
                    ddx.extend(['Early iron deficiency', 'Thalassaemia trait'])
                elif mcv > 100:
                    add_pattern('macrocytic anemia', f'MCV {mcv} fL', max(3, hb_score))
                    routes.append('Macrocytic evaluation')
                    ddx.extend(['Vitamin B12 deficiency','Folate deficiency','Alcohol related','Myelodysplasia'])
                    next_steps.append('Order B12, folate, reticulocyte count; review medications')
                else:
                    add_pattern('normocytic anemia', 'MCV in normal range', max(2, hb_score))
                    routes.append('Normocytic anemia route')
                    ddx.extend(['Acute blood loss','Hemolysis','Anemia of chronic disease'])
                    next_steps.append('Order reticulocyte count, peripheral smear, LDH, bilirubin')
            else:
                add_pattern('anemia (MCV missing)','MCV not provided; unable to subtype', max(2, hb_score))
                routes.append('General anemia route')
                next_steps.append('Order MCV, ferritin, reticulocyte count')

    # ----- Infection / inflammation / sepsis rules -----
    if wbc is not None and wbc > 11:
        wbc_score = score_severity_for_abnormality('WBC', wbc, ag, sex)
        add_pattern('leukocytosis', f'WBC {wbc} x10^9/L', wbc_score)
        if neut is not None:
            # neutrophil percent may be >70% (for reports that provide %)
            try:
                if float(neut) >= 70:
                    add_pattern('neutrophilic predominance', f'Neutrophils {neut}%', max(3, wbc_score))
            except:
                pass
        routes.append('Infection/inflammation route')
        ddx.extend(['Bacterial infection','Acute inflammation','Sepsis'])

    if crp is not None and crp > 10:
        crp_score = score_severity_for_abnormality('CRP', crp, ag, sex)
        add_pattern('elevated CRP', f'CRP {crp} mg/L', crp_score)
        if crp > 50:
            routes.append('Significant inflammatory response')
            ddx.extend(['Bacterial infection','Severe inflammatory condition'])
            next_steps.append('Consider blood cultures if febrile and urgent review for sepsis if clinically unwell')

    # ----- Sepsis suspicion (combined rule) -----
    sepsis_suspected = False
    # rule: high WBC (>15) OR very high CRP (>50) OR NLR high (>10) plus clinical hints -> escalate
    try:
        if (wbc is not None and wbc > 15) or (crp is not None and crp > 50) or (nlr is not None and nlr > 10):
            # escalate if neutrophilic or systemic markers present
            sepsis_suspected = True
    except:
        sepsis_suspected = False

    if sepsis_suspected:
        add_pattern('sepsis risk', 'Elevated inflammatory markers (WBC/CRP/NLR) suggest possible sepsis', 5 if (wbc and wbc>25 or crp and crp>150) else 4)
        routes.append('Sepsis evaluation route')
        ddx.insert(0, 'Sepsis')
        next_steps.append('Urgent clinical assessment for sepsis; consider blood cultures, IV fluids, early antibiotics if clinically indicated')
        # push severity up
        severity_scores.append(4)

    # ----- NLR high
    if nlr is not None:
        try:
            if float(nlr) > 5:
                add_pattern('high NLR', f'NLR {nlr}', 4 if float(nlr) > 10 else 3)
                routes.append('High NLR route')
                next_steps.append('Consider severe bacterial infection; urgent review if symptomatic')
        except:
            pass

    # ----- Platelet rules -----
    if plate is not None:
        plate_score = score_severity_for_abnormality('Platelets', plate, ag, sex)
        if plate < 150:
            add_pattern('thrombocytopenia', f'Platelets {plate} x10^9/L', plate_score)
            routes.append('Thrombocytopenia route')
            ddx.extend(['Immune thrombocytopenia','DIC','Bone marrow suppression'])
            next_steps.append('Repeat platelet count, peripheral smear; assess for bleeding and medication causes')
        elif plate > 450:
            add_pattern('thrombocytosis', f'Platelets {plate} x10^9/L', 2)
            ddx.extend(['Reactive thrombocytosis','Myeloproliferative disorder'])
            next_steps.append('Consider repeat and basic inflammatory workup; consider haematology referral if persistent')

    # ----- Creatinine / renal -----
    if creat is not None:
        creat_score = score_severity_for_abnormality('Creatinine', creat, ag, sex)
        if creat_score >= 3:
            add_pattern('elevated creatinine', f'Creatinine {creat} umol/L', creat_score)
            routes.append('Renal stress/AKI route')
            ddx.extend(['Acute kidney injury','Chronic kidney disease'])
            next_steps.append('Assess urine output, review nephrotoxic meds, repeat creatinine and electrolytes urgently')

    # ----- CK / Rhabdomyolysis route -----
    if ck is not None:
        ck_score = score_severity_for_abnormality('CK', ck, ag, sex)
        if ck_score >= 3:
            add_pattern('marked CK elevation', f'CK {ck} U/L', ck_score)
            routes.append('Rhabdomyolysis / muscle injury route')
            ddx.extend(['Rhabdomyolysis','Acute muscle injury','Severe exertion/drug-induced'])
            next_steps.append('Check creatinine, urine dipstick for myoglobin, aggressive IV fluids and urgent clinical assessment if severe')

    # ----- Potassium extremes -----
    if potassium is not None:
        pot_score = score_severity_for_abnormality('Potassium', potassium, ag, sex)
        if pot_score >= 4:
            add_pattern('critical potassium abnormality', f'K {potassium} mmol/L', pot_score)
            routes.append('Electrolyte emergency route')
            next_steps.append('Repeat electrolytes immediately; correct potassium as per local protocol')

    # ----- Ferritin explicit route (iron studies) -----
    if ferr is not None:
        try:
            if ferr < 30:
                add_pattern('low ferritin', f'Ferritin {ferr} ng/mL — consistent with iron deficiency', 3)
                routes.append('Iron deficiency route')
                ddx.insert(0, 'Iron deficiency anemia')
                next_steps.append('Treat iron deficiency; investigate source (GI bleed, heavy menses); consider iron replacement')
            elif ferr > 300:
                add_pattern('high ferritin', f'Ferritin {ferr} — acute phase reactant or iron overload', 2)
                routes.append('Ferritin elevated — consider inflammation or overload')
                next_steps.append('Interpret ferritin with CRP; consider haemochromatosis workup if clinically suspected')
        except:
            pass
    else:
        # if anemia + microcytic and ferr missing, instruct to order ferritin
        if any(p['pattern'].startswith('microcytic') for p in patterns) or any('anemia' in p['pattern'] for p in patterns):
            next_steps.append('Order ferritin to confirm iron deficiency vs thalassaemia trait')

    # ----- Combined patterns (infection + anemia) -----
    if hb is not None and crp is not None and wbc is not None:
        try:
            if hb < 12 and crp > 20 and wbc > 11:
                add_pattern('anemia with inflammatory response', 'Low Hb with high CRP and leukocytosis', 4)
                routes.append('Infection with anemia route')
                ddx.extend(['Anemia of inflammation','Concurrent iron deficiency with infection'])
                next_steps.append('Treat source of infection then reassess Hb; delay ferritin if major inflammation present')
        except:
            pass

    # ----- Age and sex notes -----
    age_note = ''
    if ag == 'teen' and sex.lower() == 'female':
        age_note = 'Teenage female — consider menstrual blood loss and iron deficiency as high-likelihood.'
    elif ag == 'child':
        age_note = 'Child — consider infection, nutritional deficiency; correlate clinically.'
    elif ag == 'elderly':
        age_note = 'Elderly patient — broaden differential to include chronic disease and malignancy.'

    # ----- final severity & urgency aggregation -----
    combined_score = max(severity_scores) if severity_scores else 1
    # ensure sepsis suspicion pushes to high
    if sepsis_suspected:
        combined_score = max(combined_score, 4)
    color_entry = COLOR_MAP.get(combined_score, COLOR_MAP[1])
    urgency = color_entry['urgency']

    # build concise ER summary
    summary_bullets = []
    if patterns:
        summary_bullets.append("Patterns: " + "; ".join([p['pattern'] for p in patterns]))
    if routes:
        summary_bullets.append("Primary routes: " + "; ".join(routes))
    if ddx:
        summary_bullets.append("Top differentials: " + ", ".join(list(dict.fromkeys(ddx))))
    if next_steps:
        summary_bullets.append("Immediate suggestions: " + " | ".join(next_steps))

    summary_text = " ".join(summary_bullets) if summary_bullets else "No significant abnormalities detected."

    # build impression & suggested follow-up for legacy consumers
    impression = f"Overall severity: {('critical' if combined_score==5 else 'high' if combined_score==4 else 'moderate' if combined_score==3 else 'mild')}. {summary_text}"
    suggested_follow_up = " | ".join(next_steps) if next_steps else "Clinical correlation advised."

    result = {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": list(dict.fromkeys(ddx)),
        "severity_score": combined_score,
        "urgency_flag": urgency,
        "color": color_entry['color'],
        "tw_class": color_entry['tw'],
        "age_group": ag,
        "age_note": age_note,
        "summary": summary_text,
        "impression": impression,
        "suggested_follow_up": suggested_follow_up
    }

    return result

# ---------- Trend analysis ----------
def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {"trend": "no_previous"}
    diffs = {}
    for k, v in current.items():
        prev_val = previous.get(k, {}).get('value') if previous else None
        cur_val = v.get('value')
        if prev_val is None or cur_val is None:
            continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {"previous": prev_val, "current": cur_val, "delta": delta, "pct_change": pct}
        except Exception:
            pass
    return {"trend": diffs}

# ---------- Save to Supabase ----------
def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        logger.warning("Supabase client not available; skipping save.")
        return
    try:
        res = supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": ai_results,
            "ai_error": None
        }).eq("id", report_id).execute()
        logger.info("Saved ai_results for %s", report_id)
    except Exception as e:
        logger.exception("Failed to save ai_results: %s", e)
        # attempt fallback: set ai_status failed
        try:
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":str(e)}).eq("id", report_id).execute()
        except:
            pass

# ---------- process single report ----------
def process_report(report_record: Dict[str, Any]) -> Dict[str, Any]:
    report_id = report_record.get("id") or report_record.get("report_id")
    logger.info("Processing report %s (path=%s)", report_id, report_record.get("file_path"))
    try:
        # download PDF bytes
        pdf_bytes_resp = None
        if report_record.get("file_path") and supabase:
            path = report_record.get("file_path")
            # fetch from storage
            res = supabase.storage.from_(BUCKET).download(path)
            # supabase returns object with .data for bytes
            if hasattr(res, 'data'):
                pdf_bytes_resp = res.data
            else:
                pdf_bytes_resp = res
        elif report_record.get("pdf_url"):
            import requests
            r = requests.get(report_record.get("pdf_url"))
            r.raise_for_status()
            pdf_bytes_resp = r.content
        else:
            raise ValueError("No file_path or pdf_url on report")

        pdf_bytes = pdf_bytes_resp

        # try digital text extraction
        text = extract_text_from_pdf_bytes(pdf_bytes)
        scanned = False
        if len(text.strip()) < TEXT_LENGTH_THRESHOLD:
            scanned = True

        if scanned:
            logger.info("Report %s detected as SCANNED — running Balanced OCR", report_id)
            ocr_text = do_ocr_on_pdf_bytes(pdf_bytes)
            merged_text = ocr_text
        else:
            logger.info("Report %s appears digital — using text extraction", report_id)
            merged_text = text

        # truncate for AI if needed
        if merged_text and len(merged_text) > MAX_TEXT_CHARS:
            logger.debug("Truncating merged_text for AI")
            merged_text = merged_text[:MAX_TEXT_CHARS]

        if not merged_text or not merged_text.strip():
            raise ValueError("No usable text extracted from PDF")

        parsed = find_values_in_text(merged_text)
        canonical = canonical_map(parsed)

        # fetch previous ai_results for trend analysis (by patient_id if available)
        previous = None
        pid = report_record.get("patient_id")
        if pid and supabase:
            try:
                qry = supabase.table(SUPABASE_TABLE).select("ai_results").eq("patient_id", pid).order("created_at", {"ascending": False}).limit(1).execute()
                rows = qry.data if hasattr(qry, 'data') else qry
                if rows:
                    previous = rows[0].get("ai_results")
            except Exception as e:
                logger.debug("Prev fetch failed: %s", e)

        trends = trend_analysis(canonical, previous)

        # Run route engine
        patient_meta = {"age": report_record.get("age"), "sex": report_record.get("sex")}
        routes = route_engine_v5(canonical, patient_meta, previous)

        # build ai_results payload
        ai_results = {
            "canonical": canonical,
            "routes": routes,
            "trends": trends,
            "raw_text_excerpt": merged_text[:4000],
            "scanned": scanned,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # legacy fields for compatibility
            "summary": {
                "impression": routes.get("impression"),
                "suggested_follow_up": routes.get("suggested_follow_up")
            }
        }

        # add per-key decorated info
        decorated = {}
        for k_name, kdata in canonical.items():
            v = kdata.get("value")
            # determine severity score for each key if applicable
            score = score_severity_for_abnormality(k_name, v, age_group_from_age(patient_meta.get("age")), patient_meta.get("sex", "unknown"))
            cmap = COLOR_MAP.get(score, COLOR_MAP[1])
            decorated[k_name] = {
                "raw": kdata,
                "decorated": {
                    "severity": score,
                    "urgency": cmap['urgency'],
                    "color": cmap['color'],
                    "tw_class": cmap['tw']
                }
            }
        ai_results["decorated"] = decorated

        # Save to Supabase
        save_ai_results_to_supabase(report_id, ai_results)

        logger.info("Report %s processed successfully", report_id)
        return {"success": True, "data": ai_results}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        logger.exception("Error processing report %s: %s", report_id, err)
        # mark failed in supabase
        try:
            if supabase:
                supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
        except Exception:
            pass
        return {"error": err}

# ---------- Poll loop ----------
def poll_and_process():
    if not supabase:
        logger.error("Supabase client not configured. Exiting.")
        return
    logger.info("AMI Worker V5 polling for pending reports...")
    while True:
        try:
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending").limit(5).execute()
            rows = res.data if hasattr(res, 'data') else res
            if rows:
                for r in rows:
                    try:
                        logger.info("Found job: %s", r.get("id"))
                        supabase.table(SUPABASE_TABLE).update({"ai_status": "processing"}).eq("id", r.get("id")).execute()
                        process_report(r)
                    except Exception as e:
                        logger.exception("Processing record failed: %s", e)
            else:
                logger.debug("No pending reports.")
        except Exception as e:
            logger.exception("Polling error: %s", e)
        time.sleep(POLL_INTERVAL)

# ---------- CLI test helper ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pdf", help="Path to local PDF for testing")
    parser.add_argument("--once", action="store_true", help="Poll once then exit")
    args = parser.parse_args()
    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdfb = f.read()
        # override download function by injecting a dummy record
        dummy = {"id":"local-test","file_path":"local","age":17,"sex":"female","patient_id":"local-p"}
        # monkeypatch: replace download step by using local bytes
        def download_override(rr):
            return pdfb
        # simulate process
        # simplified local flow: decide scanned/digital, parse, run routes
        text = extract_text_from_pdf_bytes(pdfb)
        scanned = len(text.strip()) < TEXT_LENGTH_THRESHOLD
        if scanned:
            extracted = do_ocr_on_pdf_bytes(pdfb)
            merged = extracted
        else:
            merged = text
        parsed = find_values_in_text(merged)
        canonical = canonical_map(parsed)
        routes = route_engine_v5(canonical, {"age": dummy.get("age"), "sex": dummy.get("sex")})
        ai_results = {
            "canonical": canonical,
            "routes": routes,
            "raw_text_excerpt": merged[:4000],
            "scanned": scanned,
            "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        print(json.dumps(ai_results, indent=2))
    else:
        poll_and_process()
