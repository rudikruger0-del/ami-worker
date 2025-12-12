#!/usr/bin/env python3
# worker.py — AMI Health Worker V5 (single-file, fixed OCR & parsers)
# Usage: python worker.py --test-pdf sample.pdf
# Keep .env variables (SUPABASE_URL, SUPABASE_KEY, ...)

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
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'reports')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '6'))
PDF_RENDER_DPI = int(os.getenv('PDF_RENDER_DPI', '200'))
TEXT_LENGTH_THRESHOLD = int(os.getenv('TEXT_LENGTH_THRESHOLD', '80'))
OCR_LANG = os.getenv('OCR_LANG', 'eng')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ami-worker-v5')

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
        path = record['file_path']
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
            if hasattr(res, 'data'):
                return res.data
            return res
        except Exception as e:
            logger.exception("Supabase storage download failed: %s", e)
            raise
    else:
        raise ValueError("No 'pdf_url' or 'file_path' in record, or supabase not configured.")

def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    """Extract text using pypdf (digital text). Returns concatenated pages."""
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
    """Decide whether PDF is scanned (low extracted text length)"""
    txt = extract_text_with_pypdf(pdf_bytes)
    if len(txt.strip()) < TEXT_LENGTH_THRESHOLD:
        logger.info("PDF appears scanned (text len %d)", len(txt))
        return True
    return False

# ---------- OCR via pytesseract (fixed) ----------

def preprocess_image_for_ocr(img: Image.Image, target_min_dim: int = 1600) -> Image.Image:
    """
    Preprocess to improve OCR:
    - convert to grayscale,
    - autocontrast,
    - denoise (median),
    - upscale small pages for better OCR.
    """
    try:
        img = img.convert('L')
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        w, h = img.size
        maxdim = max(w, h)
        if maxdim < target_min_dim:
            factor = max(1, int(target_min_dim / maxdim))
            img = img.resize((w * factor, h * factor), Image.LANCZOS)
        return img
    except Exception as e:
        logger.debug("preprocess_image_for_ocr failed: %s", e)
        return img

def ocr_image_pytesseract(img: Image.Image, lang: str = OCR_LANG) -> str:
    """Robust OCR with digit-merge fixes to rejoin spaced digits and small OCR splits."""
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract is not available in this environment.")
    img2 = preprocess_image_for_ocr(img)
    config = r'--oem 3 --psm 6'
    try:
        text = pytesseract.image_to_string(img2, lang=lang, config=config)
    except Exception as e:
        logger.exception("pytesseract error: %s", e)
        text = ''

    # sanitize: remove non-printables
    text = ''.join(ch if (31 < ord(ch) < 127 or ch in '\n\r\t') else ' ' for ch in text)

    # --- CORE FIX: rejoin broken numeric fragments produced by OCR ---
    # Examples fixed:
    #  "1 4 0 2 8" -> "14028"
    #  "3 8 . 3"   -> "38.3"
    #  "1 2 , 5"   -> "12.5"
    # Remove spaces between digits; also remove spaces immediately after dots if digits follow
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)        # join digit groups
    text = re.sub(r'(?<=\.)\s+(?=\d)', '', text)        # "3. 8" -> "3.8"
    text = re.sub(r'(?<=\d)\s+(?=\.?\s*\d)', '', text) # extra guard: "3 8 . 3" -> "38.3"

    return text

def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    """Render PDF to list of PIL Images using pdf2image."""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
        return imgs
    except Exception as e:
        logger.exception("convert_from_bytes failed: %s", e)
        raise

def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    """Render scanned PDF to images and OCR each page with pytesseract, returning concatenated text."""
    try:
        images = pdf_to_images(pdf_bytes)
    except Exception as e:
        logger.exception("Failed to render PDF to images: %s", e)
        raise

    texts = []
    for i, img in enumerate(images):
        try:
            page_text = ocr_image_pytesseract(img)
            logger.info("OCR page %d length %d", i, len(page_text))
        except Exception as e:
            logger.exception("pytesseract failed on page %d: %s", i, e)
            page_text = ''
        texts.append(page_text)
    return '\n\n---PAGE_BREAK---\n\n'.join(texts)

# ---------- Parsing: robust lab extraction (fixed) ----------

# regex pieces
VALUE_RE = r'(-?\d+\.\d+|-?\d+)'
PERCENT_RE = r'([0-9]{1,3}\.?\d*)\s*%'

COMMON_KEYS = {
    'hb': ['hb','haemoglobin','hemoglobin'],
    'rbc': ['rbc','erythrocyte count','erythrocyte'],
    'hct': ['hct','haematocrit','hematocrit'],
    'mcv': ['mcv','mean corpuscular volume'],
    'mch': ['mch','mean corpuscular haemoglobin','mean corpuscular hemoglobin'],
    'mchc': ['mchc'],
    'rdw': ['rdw','red cell distribution width'],
    'wbc': ['wbc','white cell count','white blood cell','leukocyte','leucocyte','leukocytes'],
    'neutrophils_pc': ['neutrophils %','neutrophils%','neutrophils percent','neutrophil%','neutrophils'],
    'neutrophils_abs': ['neutrophil absolute','neutrophil count','neutrophils absolute'],
    'lymphocytes_pc': ['lymphocytes %','lymphocytes%','lymphocytes'],
    'monocytes_pc': ['monocytes %','monocytes'],
    'eosinophils_pc': ['eosinophils %','eosinophils'],
    'basophils_pc': ['basophils %','basophils'],
    'platelets': ['platelets','thrombocytes','platelet count'],
    'crp': ['crp','c-reactive protein','c reactive protein'],
    'creatinine': ['creatinine','creat'],
    'sodium': ['sodium','na '],
    'potassium': ['potassium','k '],
    'chloride': ['chloride','cl '],
    'urea': ['urea','bun'],
    'alt': ['alt','alanine aminotransferase'],
    'ast': ['ast','aspartate aminotransferase'],
    'ck': ['ck','creatine kinase'],
    # keep CK-MB possibility as fallback label in free-text (handled via generic gen pattern)
}

LABEL_TO_KEY = {}
for k, labels in COMMON_KEYS.items():
    for l in labels:
        LABEL_TO_KEY[l.lower()] = k

def normalize_label(lbl: str) -> str:
    return re.sub(r'[^a-z0-9 ]', '', lbl.lower()).strip()

def find_key_for_label(label: str) -> Optional[str]:
    l = normalize_label(label)
    if l in LABEL_TO_KEY:
        return LABEL_TO_KEY[l]
    # fuzzy contains check
    for lab, key in LABEL_TO_KEY.items():
        if lab in l or l in lab:
            return key
    return None

def safe_float(s: str) -> Optional[float]:
    """Robust numeric parser:
     - replaces commas with dots
     - removes stray characters (keeps digits, dot, minus)
     - if multiple dots keep last (join fragments)
     - returns None if empty / invalid
    """
    if s is None:
        return None
    s = s.replace(',', '.')
    # remove everything except digits, dot, minus
    s = re.sub(r'[^0-9\.\-]', '', s)
    # if multiple dots, keep only last dot as decimal separator (join earlier fragments)
    if s.count('.') > 1:
        parts = s.split('.')
        s = ''.join(parts[:-1]) + '.' + parts[-1]
    s = s.strip()
    if s == '':
        return None
    try:
        return float(s)
    except:
        return None

def normalize_impossible_values(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Heuristic corrections for obvious OCR-caused impossible values."""
    def fix_value(key, v):
        if v is None:
            return v
        try:
            v = float(v)
        except:
            return None
        # Hemoglobin: if >50 likely missing decimal
        if key == "hb":
            if v > 50:
                return round(v / 10.0, 2)
            if v < 3:
                return None
        if key == "mcv":
            if v < 40 or v > 200:
                return None
        if key == "rdw":
            if v > 40 or v < 5:
                return None
        if key == "wbc":
            # WBC in x10^9/L; OCR 3000 -> 300.0 -> divide heuristics
            if v > 300:
                return round(v / 10.0, 2)
        if key == "platelets":
            # Platelets often reported as 376 (x10^9/L) or raw counts like 4529600
            if v > 1000000:
                return round(v / 10000.0, 1)
            if v > 10000 and v < 1000000:
                return round(v / 1000.0, 1)
        if key == "neutrophils_pc":
            if v < 0 or v > 100:
                return None
        if key == "potassium":
            # 24 -> 2.4 common OCR error
            if 20 <= v < 100:
                return round(v / 10.0, 2)
            if v > 100:
                return None
        if key == "creatinine":
            if v > 2000:
                return round(v / 10.0, 2)
        return v

    for k in list(results.keys()):
        entry = results.get(k)
        if entry and 'value' in entry:
            entry['value'] = fix_value(k, entry.get('value'))
    return results

def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parsing:
    - Merge spaced digits early (fix OCR breakages).
    - Search labeled tokens (exact & fuzzy).
    - Generic fallback (label/value/unit groups).
    - Post-process impossible values heuristically.
    """
    # CORE FIX: rejoin broken OCR numeric fragments globally BEFORE parsing
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(r'(?<=\.)\s+(?=\d)', '', text)

    results: Dict[str, Dict[str, Any]] = {}
    # normalize whitespace; split lines
    lines = [ln.strip() for ln in re.split(r'\n|\r', text) if ln.strip()]

    # first pass: direct labelled matches
    for line in lines:
        ln = line.lower()
        for label, key in LABEL_TO_KEY.items():
            if label in ln:
                # percent forms (e.g., "Neutrophils 78%")
                p = re.search(PERCENT_RE, line)
                if p:
                    val = safe_float(p.group(1))
                    if val is not None:
                        results.setdefault(key, {})['value'] = val
                        results.setdefault(key, {})['raw_line'] = line
                        continue
                # numeric near label
                try:
                    pat = re.compile(rf'{re.escape(label)}[^\d\n\r\-]{{0,40}}({VALUE_RE})', flags=re.IGNORECASE)
                    m = pat.search(line)
                    if m:
                        val = safe_float(m.group(1))
                        if val is not None:
                            results.setdefault(key, {})['value'] = val
                            um = re.search(rf'{re.escape(label)}[^\d\n\r]*{re.escape(m.group(1))}\s*([a-zA-Z/%\d\.\-]*)', line, flags=re.IGNORECASE)
                            if um and um.group(1).strip():
                                results.setdefault(key, {})['unit'] = um.group(1).strip()
                            results.setdefault(key, {})['raw_line'] = line
                            continue
                except Exception:
                    pass

        # fallback generic capture: "<label> 11.6 unit"
        gen = re.findall(r'([A-Za-z\-/ ]{2,40})\s*[:\-]?\s*(' + VALUE_RE + r')\s*([a-zA-Z/%\d\.\-]*)', line)
        for g in gen:
            label_raw = g[0].strip()
            val_s = g[1]
            unit = g[2].strip() if g[2] else None
            key = find_key_for_label(label_raw)
            if key:
                v = safe_float(val_s)
                if v is None:
                    continue
                results.setdefault(key, {})['value'] = v
                if unit:
                    results.setdefault(key, {})['unit'] = unit
                results.setdefault(key, {})['raw_line'] = line

    # second pass: neutrophils absolute / percent detection
    for line in lines:
        if 'neutrophil' in line.lower():
            p = re.search(PERCENT_RE, line)
            if p:
                v = safe_float(p.group(1))
                if v is not None:
                    results.setdefault('neutrophils_pc', {})['value'] = v
            a = re.search(r'([0-9]{1,3}\.\d+)\s*x\s*10\^?\d?/?L', line, flags=re.IGNORECASE)
            if a:
                v = safe_float(a.group(1))
                if v is not None:
                    results.setdefault('neutrophils_abs', {})['value'] = v
            a2 = re.search(r'([0-9]{1,3}\.\d+)\s*/?L', line, flags=re.IGNORECASE)
            if a2 and 'x' not in line:
                v = safe_float(a2.group(1))
                if v is not None:
                    results.setdefault('neutrophils_abs', {})['value'] = v

    # some final heuristics for platelets if present as big raw numbers handled earlier
    if 'platelets' in results:
        pl = results['platelets'].get('value')
        if pl is not None:
            if pl > 10000 and pl < 1000000:
                results['platelets']['value'] = round(pl / 1000.0, 1)
            elif pl > 1000000:
                results['platelets']['value'] = round(pl / 1000000.0, 3)

    # final cleanup of impossible values
    results = normalize_impossible_values(results)
    return results

# ---------- Canonical mapping & decoration ----------
CANONICAL_KEYS = ['Hb', 'MCV', 'MCH', 'MCHC', 'RDW', 'WBC', 'Neutrophils', 'Lymphocytes',
                  'Monocytes', 'Eosinophils', 'Basophils', 'NLR', 'Platelets', 'Creatinine',
                  'CRP', 'Sodium', 'Potassium', 'Chloride', 'Urea', 'RBC', 'HCT', 'ALT', 'AST', 'CK']

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    map_rules = {
        'hb': 'Hb', 'mcv': 'MCV', 'mch': 'MCH', 'mchc': 'MCHC', 'rdw': 'RDW',
        'wbc': 'WBC', 'neutrophils_pc': 'Neutrophils', 'neutrophils_abs': 'Neutrophils',
        'lymphocytes_pc': 'Lymphocytes', 'monocytes_pc': 'Monocytes',
        'eosinophils_pc': 'Eosinophils', 'basophils_pc': 'Basophils',
        'platelets': 'Platelets', 'creatinine': 'Creatinine', 'crp': 'CRP',
        'sodium': 'Sodium', 'potassium': 'Potassium', 'chloride': 'Chloride',
        'urea': 'Urea', 'rbc': 'RBC', 'hct': 'HCT', 'alt': 'ALT', 'ast': 'AST', 'ck': 'CK'
    }
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in parsed.items():
        canon = map_rules.get(k)
        if not canon:
            continue
        try:
            out[canon] = {'value': float(v.get('value')) if v.get('value') is not None else None}
        except Exception:
            out[canon] = {'value': None}
        if v.get('unit'):
            out[canon]['unit'] = v.get('unit')
        if v.get('raw_line'):
            out[canon]['raw'] = v.get('raw_line')

    # compute NLR if both neutrophils and lymphocytes present
    try:
        n = out.get('Neutrophils', {}).get('value')
        l = out.get('Lymphocytes', {}).get('value')
        if n is not None and l is not None and l != 0:
            out['NLR'] = {'value': round(float(n) / float(l), 2)}
    except Exception:
        pass

    return out

# ---------- Risk bar helper, flags, severity mapping (kept from V5) ----------
COLOR_MAP = {
    5: {'label': 'critical', 'color': '#b91c1c', 'tw': 'bg-red-700', 'urgency': 'high'},
    4: {'label': 'severe', 'color': '#ef4444', 'tw': 'bg-red-500', 'urgency': 'high'},
    3: {'label': 'moderate', 'color': '#f59e0b', 'tw': 'bg-yellow-400', 'urgency': 'medium'},
    2: {'label': 'mild', 'color': '#facc15', 'tw': 'bg-yellow-300', 'urgency': 'low'},
    1: {'label': 'normal', 'color': '#10b981', 'tw': 'bg-green-500', 'urgency': 'low'},
}

def risk_percentage_for_key(key: str, value: Optional[float]) -> int:
    if value is None:
        return 0
    k = key.lower()
    try:
        v = float(value)
    except:
        return 0
    if k == 'crp':
        pct = min(100, int((v / 200.0) * 100))
        return pct
    if k == 'wbc':
        if v <= 11: return int((v/11)*20)
        if v <= 20: return 30 + int(((v-11)/9) * 30)
        return min(100, 60 + int(((v-20)/30)*40))
    if k == 'neutrophils' or k == 'nlr':
        if v <= 3: return int((v/3)*10)
        if v <= 6: return 15 + int(((v-3)/3)*25)
        if v <= 10: return 40 + int(((v-6)/4)*30)
        return min(100, 70 + int(((v-10)/30)*30))
    if k == 'creatinine':
        if v <= 120: return int((v/120)*20)
        if v <= 200: return 25 + int(((v-120)/80)*30)
        return min(100, 60 + int(((v-200)/300)*40))
    if k == 'hb':
        if v >= 12: return 5
        if v >= 10: return 20
        if v >= 8: return 50
        return min(100, 70 + int(((12 - v)/12)*30))
    if k == 'platelets':
        if v >= 100 and v <= 450: return 5
        if v < 100: return min(100, 30 + int(((100 - v)/100)*70))
        if v > 450: return min(100, 20 + int(((v-450)/1000)*80))
    return min(100, int(min(abs(v), 100)))

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

def severity_text_from_score(score: int) -> str:
    try:
        s = int(score)
    except:
        s = 1
    if s <= 1:
        return 'normal'
    if s == 2:
        return 'mild'
    if s == 3:
        return 'moderate'
    if s == 4:
        return 'severe'
    return 'critical'

def flag_for_key(key: str, value: Optional[float], sex: str = 'unknown') -> Tuple[str, str]:
    if value is None:
        return 'normal', '#ffffff'
    k = key.lower()
    try:
        v = float(value)
    except:
        return 'normal', '#ffffff'
    if k == 'hb':
        if sex.lower() == 'female':
            low, high = 12.0, 15.5
        else:
            low, high = 13.0, 17.5
        if v < low: return 'low', '#f59e0b'
        if v > high: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'wbc':
        if v < 4.0: return 'low', '#f59e0b'
        if v > 11.0: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'platelets':
        if v < 150: return 'low', '#f59e0b'
        if v > 450: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'creatinine':
        if v < 45: return 'low', '#f59e0b'
        if v > 120: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'crp':
        if v <= 10: return 'normal', '#ffffff'
        if v <= 50: return 'high', '#f59e0b'
        return 'high', '#b91c1c'
    if k == 'sodium':
        if v < 135: return 'low', '#f59e0b'
        if v > 145: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'potassium':
        if v < 3.5: return 'low', '#f59e0b'
        if v > 5.1: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'mcv':
        if v < 80: return 'low', '#f59e0b'
        if v > 100: return 'high', '#b91c1c'
        return 'normal', '#ffffff'
    if k == 'nlr':
        if v > 10: return 'high', '#b91c1c'
        if v > 5: return 'high', '#f59e0b'
        return 'normal', '#ffffff'
    if k in ('alt', 'ast', 'ck'):
        if v > 200: return 'high', '#b91c1c'
        if v > 100: return 'high', '#f59e0b'
        return 'normal', '#ffffff'
    return 'normal', '#ffffff'

# ---------- Route Engine V5 (kept) ----------
# (kept mostly identical to the V5 you provided — no admission logic, severity_text only)
def score_severity_for_abnormality_v5(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    if value is None:
        return 1
    try:
        v = float(value)
    except:
        return 1
    key_l = key.lower()
    score = 1
    if key_l == 'hb':
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
        if v > 30: score = 5
        elif v > 20: score = 4
        elif v > 12: score = 3
    elif key_l == 'crp':
        if v > 250: score = 5
        elif v > 100: score = 4
        elif v > 50: score = 3
        elif v > 10: score = 2
    elif key_l in ('neutrophils', 'nlr'):
        if v > 12: score = 5
        elif v > 7: score = 4
        elif v > 3: score = 3
    elif key_l == 'creatinine':
        if v > 400: score = 5
        elif v > 200: score = 4
        elif v > 120: score = 3
    elif key_l == 'platelets':
        if v < 10: score = 5
        elif v < 30: score = 4
        elif v < 100: score = 3
        elif v > 1000: score = 4
    elif key_l in ('sodium', 'potassium'):
        if key_l == 'sodium':
            if v < 120 or v > 160: score = 5
            elif v < 125 or v > 155: score = 4
            elif v < 130 or v > 150: score = 3
        if key_l == 'potassium':
            if v < 2.8 or v > 6.5: score = 5
            elif v < 3.2 or v > 6.0: score = 4
            elif v < 3.5 or v > 5.5: score = 3
    else:
        score = 1
    return score

def generate_diagnostic_possibilities(canonical: Dict[str, Dict[str, Any]], patterns: List[Dict[str, Any]], routes: List[str]) -> List[str]:
    Hb = canonical.get("Hb", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")
    Neut = canonical.get("Neutrophils", {}).get("value")
    CRP = canonical.get("CRP", {}).get("value")
    NLR = canonical.get("NLR", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")
    K = canonical.get("Potassium", {}).get("value")
    Urea = canonical.get("Urea", {}).get("value")

    possibilities: List[str] = []
    if (WBC and WBC > 12) or (Neut and Neut > 80) or (NLR and NLR > 10) or (CRP and CRP > 20):
        reasons = []
        if WBC and WBC > 12: reasons.append(f"WBC {WBC}")
        if Neut and Neut > 80: reasons.append(f"neutrophilia {Neut}%")
        if NLR and NLR > 10: reasons.append(f"NLR {NLR}")
        if CRP and CRP > 20: reasons.append(f"CRP {CRP}")
        possibilities.append("Sepsis / bacterial infection — " + "; ".join(reasons))
    if K is not None and (K < 3.0 or K > 6.0):
        possibilities.append(f"Severe electrolyte derangement — potassium {K} mmol/L")
    if Creat is not None and Creat > 120:
        possibilities.append(f"Acute kidney injury suspected — creatinine {Creat} umol/L")
    else:
        possibilities.append("Renal function normal — no evidence of AKI")
    if Hb is not None and Hb < 11:
        possibilities.append(f"Anemia — Hb {Hb} g/dL")
    else:
        possibilities.append("No anemia — Hb normal for age/sex")
    if CRP is not None and CRP > 10:
        possibilities.append(f"Inflammatory response — CRP {CRP} mg/L")
    return possibilities

def route_engine_v5(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    age = patient_meta.get('age')
    sex = patient_meta.get('sex', 'unknown')
    ag = age_group_from_age(age)

    patterns: List[Dict[str, Any]] = []
    routes: List[str] = []
    next_steps: List[str] = []
    ddx: List[str] = []
    per_key_scores: Dict[str, int] = {}

    def add_pattern(name, reason, score):
        patterns.append({'pattern': name, 'reason': reason})
        per_key_scores[name] = max(per_key_scores.get(name, 1), score)

    hb = canonical.get('Hb', {}).get('value')
    mcv = canonical.get('MCV', {}).get('value')
    wbc = canonical.get('WBC', {}).get('value')
    crp = canonical.get('CRP', {}).get('value')
    neut = canonical.get('Neutrophils', {}).get('value')
    nlr = canonical.get('NLR', {}).get('value')
    plate = canonical.get('Platelets', {}).get('value')
    creat = canonical.get('Creatinine', {}).get('value')
    sodium = canonical.get('Sodium', {}).get('value')
    potassium = canonical.get('Potassium', {}).get('value')
    alt = canonical.get('ALT', {}).get('value')
    ast = canonical.get('AST', {}).get('value')
    ck = canonical.get('CK', {}).get('value')

    # Anemia handling
    if hb is not None:
        score_hb = score_severity_for_abnormality_v5('hb', hb, ag, sex)
        if score_hb > 1:
            add_pattern('anemia', f'Hb {hb} g/dL', score_hb)
            if mcv is not None and mcv < 80:
                add_pattern('microcytic anemia', f'MCV {mcv} fL', max(3, score_hb))
                routes.append('Iron deficiency route (Ferritin + reticulocyte)')
                ddx.extend(['Iron deficiency anemia', 'Thalassaemia trait', 'Chronic blood loss'])
                next_steps.append('Order ferritin, reticulocyte count, and peripheral smear. Consider stool occult blood if adult.')
            elif mcv is not None and mcv > 100:
                add_pattern('macrocytic anemia', f'MCV {mcv} fL', max(3, score_hb))
                routes.append('Macrocytic route (B12/folate)')
                ddx.extend(['Vitamin B12 deficiency', 'Folate deficiency', 'Alcohol-related'])
                next_steps.append('Order B12 and folate; review meds.')
            else:
                add_pattern('normocytic anemia', 'MCV normal or missing', max(2, score_hb))
                routes.append('Normocytic anemia route')
                ddx.extend(['Acute blood loss', 'Hemolysis', 'Anaemia of inflammation'])
                next_steps.append('Order reticulocyte count, LDH, peripheral smear.')

    # Infection / inflammation & sepsis signals
    sepsis_flag = False
    if wbc is not None and wbc > 11:
        s = score_severity_for_abnormality_v5('wbc', wbc, ag, sex)
        add_pattern('leukocytosis', f'WBC {wbc} x10^9/L', s)
        if neut is not None and neut >= 70:
            add_pattern('neutrophilic predominance', f'Neutrophils {neut}%', max(3, s))
            routes.append('Bacterial infection / Sepsis route')
            ddx.extend(['Bacterial infection', 'Sepsis', 'Acute inflammation'])
            next_steps.append('Clinical assessment for sepsis; consider blood cultures, IV fluids, empiric antibiotics if unstable.')
            sepsis_flag = True

    if crp is not None:
        s = score_severity_for_abnormality_v5('crp', crp, ag, sex)
        if s > 1:
            add_pattern('elevated CRP', f'CRP {crp} mg/L', s)
            if crp > 50:
                routes.append('Significant inflammatory response')
                ddx.extend(['Severe infection', 'Inflammatory disease'])
                next_steps.append('Consider urgent review; blood cultures if febrile; procalcitonin if available.')
                if crp > 150:
                    sepsis_flag = True

    if nlr is not None:
        if nlr > 10:
            add_pattern('very high NLR', f'NLR {nlr}', 5)
            routes.append('High NLR / Sepsis route')
            next_steps.append('Urgent clinical review for sepsis; consider sepsis pathway.')
            sepsis_flag = True
        elif nlr > 5:
            add_pattern('high NLR', f'NLR {nlr}', 4)
            routes.append('High NLR route')
            next_steps.append('Assess severity and source of infection.')

    # Platelets
    if plate is not None:
        pscore = score_severity_for_abnormality_v5('platelets', plate, ag, sex)
        if plate < 150:
            add_pattern('thrombocytopenia', f'Platelets {plate}', pscore)
            ddx.extend(['Immune thrombocytopenia', 'DIC', 'Bone marrow suppression'])
            next_steps.append('Repeat platelet count; check smear; assess bleeding.')
        elif plate > 450:
            add_pattern('thrombocytosis', f'Platelets {plate}', max(2, pscore))
            next_steps.append('Consider reactive thrombocytosis; repeat and investigate inflammation/iron status.')

    # Renal / AKI
    if creat is not None:
        cscore = score_severity_for_abnormality_v5('creatinine', creat, ag, sex)
        if cscore >= 3:
            add_pattern('elevated creatinine', f'Creatinine {creat} umol/L', cscore)
            routes.append('AKI route')
            ddx.extend(['Acute kidney injury', 'Chronic kidney disease'])
            next_steps.append('Repeat creatinine urgently; check urine output and electrolytes; review meds.')

    # Electrolytes
    if sodium is not None:
        s = score_severity_for_abnormality_v5('sodium', sodium, ag, sex)
        if s >= 3:
            add_pattern('sodium derangement', f'Sodium {sodium} mmol/L', s)
            next_steps.append('Correct sodium abnormalities per local protocol; check fluid status.')

    if potassium is not None:
        s = score_severity_for_abnormality_v5('potassium', potassium, ag, sex)
        if s >= 3:
            add_pattern('potassium derangement', f'Potassium {potassium} mmol/L', s)
            next_steps.append('Correct potassium urgently; monitor ECG.')

    # Rhabdomyolysis and LFT
    if ck is not None and ck > 1000:
        add_pattern('rhabdomyolysis signal', f'CK {ck}', 4)
        routes.append('Rhabdomyolysis route')
        next_steps.append('Assess for muscle pain/urine colour; check creatinine and electrolytes; consider urgent fluids.')

    if (alt is not None and alt > 200) or (ast is not None and ast > 200):
        add_pattern('transaminitis', f'ALT {alt} AST {ast}', 3)
        routes.append('Hepatic route')
        next_steps.append('Review hepatotoxins, viral hepatitis risk; consider LFT panel.')

    # Combined pattern: anemia + inflammation
    if hb is not None and crp is not None and wbc is not None:
        if hb < 12 and crp > 20 and wbc > 11:
            add_pattern('anemia with inflammatory/infective response', 'Low Hb + high CRP + leukocytosis', 4)
            routes.append('Infection with anemia route')
            next_steps.append('Treat source of infection; reassess Hb after control; do ferritin when CRP falls.')

    # Age/sex modifiers
    if ag == 'teen' and sex.lower() == 'female':
        next_steps.append('Assess menstrual history; consider urgent ferritin and reticulocyte count.')

    # Build differential ranking by simple heuristics (frequency + severity)
    ddx_rank: Dict[str, int] = {}
    for i, d in enumerate(ddx):
        ddx_rank[d] = ddx_rank.get(d, 0) + (10 - i)
    if sepsis_flag:
        for d in ('Sepsis','Bacterial infection','Severe infection'):
            ddx_rank[d] = ddx_rank.get(d, 0) + 50
    ddx_sorted = sorted(ddx_rank.items(), key=lambda x: -x[1])
    ddx_list = [d for d, _ in ddx_sorted]

    # severity aggregation
    severity_scores = list(per_key_scores.values()) if per_key_scores else [1]
    combined_score = max(severity_scores) if severity_scores else 1
    color_entry = COLOR_MAP.get(combined_score, COLOR_MAP[1])
    urgency = color_entry['urgency']

    # risk bars
    risk_bars: Dict[str, Dict[str, Any]] = {}
    for kk in CANONICAL_KEYS:
        val = canonical.get(kk, {}).get('value')
        pct = risk_percentage_for_key(kk, val)
        if pct >= 80:
            clr = '#b91c1c'
        elif pct >= 60:
            clr = '#ef4444'
        elif pct >= 40:
            clr = '#f59e0b'
        elif pct >= 20:
            clr = '#facc15'
        else:
            clr = '#10b981'
        risk_bars[kk] = {'percentage': pct, 'color': clr}

    diagnostic_possibilities = generate_diagnostic_possibilities(canonical, patterns, routes)

    summary_lines: List[str] = []
    if diagnostic_possibilities:
        summary_lines.append("Diagnostic possibilities:\n• " + "\n• ".join(diagnostic_possibilities))
    if patterns:
        summary_lines.append('Patterns: ' + '; '.join([p['pattern'] for p in patterns]))
    if routes:
        summary_lines.append('Primary routes: ' + '; '.join(routes))
    if ddx_list:
        summary_lines.append('Top differential diagnoses: ' + ', '.join(ddx_list))
    if next_steps:
        summary_lines.append('Immediate suggested actions: ' + ' | '.join(next_steps))

    age_note = ''
    if ag == 'teen' and sex.lower() == 'female':
        age_note = 'Teenage female — consider menstrual blood loss and iron deficiency.'
    elif ag == 'elderly':
        age_note = 'Elderly – broaden differential for chronic disease and malignancy.'

    severity_text = severity_text_from_score(combined_score)

    final = {
        'patterns': patterns,
        'routes': routes,
        'next_steps': next_steps,
        'differential': ddx_list,
        'severity_text': severity_text,
        'urgency_flag': urgency,
        'color': color_entry['color'],
        'tw_class': color_entry['tw'],
        'age_group': ag,
        'age_note': age_note,
        'diagnostic_possibilities': diagnostic_possibilities,
        'risk_bars': risk_bars,
        'summary': '\n'.join(summary_lines[:12]) if summary_lines else 'No significant abnormalities detected.'
    }
    return final

# ---------- Trends, save & processing ----------
def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {'trend': 'no_previous'}
    diffs: Dict[str, Any] = {}
    for k, v in current.items():
        prev_val = previous.get(k, {}).get('value') if isinstance(previous, dict) else None
        cur_val = v.get('value')
        if prev_val is None or cur_val is None:
            continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {'previous': prev_val, 'current': cur_val, 'delta': delta, 'pct_change': pct}
        except Exception:
            pass
    return {'trend': diffs or 'no_change'}

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
    routes = route_engine_v5(canonical, patient_meta, previous)

    ai_results = {
        'canonical': canonical,
        'parsed': parsed,
        'routes': routes,
        'trends': trends,
        'raw_text_excerpt': text[:5000],
        'scanned': scanned,
        'processed_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }

    # add per-key decorated flags (color, severity_text, risk bar)
    decorated: Dict[str, Any] = {}
    sex = record.get('sex', 'unknown')
    for key in CANONICAL_KEYS:
        val = canonical.get(key, {})
        value = val.get('value')
        unit = val.get('unit') if isinstance(val.get('unit'), str) else None

        s_num = score_severity_for_abnormality_v5(key, value, age_group_from_age(record.get('age')), sex)
        s_text = severity_text_from_score(s_num)
        flag, flag_color = flag_for_key(key, value, sex)
        pct = risk_percentage_for_key(key, value)
        risk_color = '#b91c1c' if pct >= 80 else ('#ef4444' if pct >= 60 else ('#f59e0b' if pct >= 40 else ('#facc15' if pct >= 20 else '#10b981')))
        decorated[key] = {
            'value': value,
            'unit': unit,
            'flag': flag,
            'color': flag_color,
            'severity_text': s_text,
            'risk_bar': {'percentage': pct, 'color': risk_color}
        }
    ai_results['decorated'] = decorated

    try:
        save_ai_results_to_supabase(report_id, ai_results)
    except Exception as e:
        logger.exception("Failed to save results: %s", e)

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
    parser = argparse.ArgumentParser(description="AMI Health Worker V5 (pypdf + pytesseract)")
    parser.add_argument('--test-pdf', help='Path to local PDF for testing')
    parser.add_argument('--once', action='store_true', help='Poll once then exit (if not test)')
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, 'rb') as f:
            pdfb = f.read()
        dummy = {
            'id': 'local-test',
            'patient_id': 'local-patient',
            'age': 45,
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
