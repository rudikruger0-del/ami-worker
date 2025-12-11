"""
AMI Health Worker V4 - single-file worker.py (Doctor‑magnet, Option 1)

This full worker implements the AMI Health Worker V4 you requested.
Key features included in this regeneration:
- Robust environment handling using SUPABASE_SERVICE_KEY or SUPABASE_KEY fallback.
- Startup diagnostics and non-silent failure messages (clear error if keys missing).
- PDF download from Supabase storage or direct pdf_url.
- Scanned detection with pypdf text extraction threshold.
- OCR: prefer local pytesseract; fallback to OpenAI Vision (base64 JPEG) when needed.
- Robust regex-based parser for CBC and chemistry analytes with broad label coverage.
- Canonical mapping to required keys (Hb, MCV, MCH, WBC, Neutrophils, Platelets, Creatinine, CRP, etc.).
- Route Engine V4: patterns → routes → next_steps, differential diagnosis, severity (1–5), urgency flags.
- Age & sex aware interpretation and specialty views for GP, ER, Paeds, Obs/Gyn, Internal Med, Nephrology, Haematology, Infectious Diseases.
- Color-coded flags (hex + Tailwind class) ready for frontend rendering.
- Per-key decorated output + overall Action Card (one-line immediate actions, timeframe, tests).
- Trend analysis when previous ai_results exist for the same patient_id.
- Safeguarded Supabase updates with explicit status transitions: pending → processing → completed/failed.
- CLI test mode (--test-pdf) for local verification.

Usage:
- Populate environment variables: SUPABASE_URL, SUPABASE_SERVICE_KEY (or SUPABASE_KEY), OPENAI_API_KEY (optional but recommended), SUPABASE_TABLE, SUPABASE_BUCKET, POLL_INTERVAL, PDF_RENDER_DPI
- Deploy to your container and run worker.py

"""

import os
import io
import re
import time
import json
import base64
import math
import logging
import traceback
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

# optional imports
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

# environment loader with multiple fallbacks
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_KEY') or os.getenv('SUPABASE_KEY') or os.getenv('SUPABASE_SERVICE_ROLE_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_KEY')
SUPABASE_TABLE = os.getenv('SUPABASE_TABLE', 'reports')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'reports')
POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '6'))
PDF_RENDER_DPI = int(os.getenv('PDF_RENDER_DPI', '200'))
TEXT_LENGTH_THRESHOLD = int(os.getenv('TEXT_LENGTH_THRESHOLD', '120'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('ami-worker')

# Diagnostic startup
logger.info('Starting AMI Health Worker V4')
logger.info('pytesseract available: %s', HAS_PYTESSERACT)
logger.info('openai available: %s', HAS_OPENAI)
logger.info('supabase package available: %s', HAS_SUPABASE)

if HAS_OPENAI and OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY

# Create supabase client, but fail gracefully with clear message if keys missing
supabase = None
if HAS_SUPABASE:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.error('Missing Supabase credentials. Expected SUPABASE_SERVICE_KEY or SUPABASE_KEY environment variable.')
        logger.error('Found SUPABASE_URL=%s SUPABASE_KEY=%s', bool(SUPABASE_URL), bool(SUPABASE_KEY))
        # We deliberately do not raise here – we allow offline/testing mode, but worker loop will be disabled.
        supabase = None
    else:
        try:
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info('Connected to Supabase')
        except Exception as e:
            logger.exception('Failed to create Supabase client: %s', e)
            supabase = None
else:
    logger.warning('Supabase python package not installed. Worker will not be able to access Supabase.')

# ---------- helpers ----------

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

# ---------- PDF reading & scanned detection ----------

def download_pdf(report: Dict[str, Any]) -> bytes:
    """Download PDF via pdf_url or Supabase storage path in report."""
    if report.get('pdf_url'):
        import requests
        r = requests.get(report['pdf_url'])
        r.raise_for_status()
        return r.content
    if supabase and report.get('file_path'):
        response = supabase.storage.from_(SUPABASE_BUCKET).download(report['file_path'])
        # supabase client may return an object with .data
        if hasattr(response, 'data'):
            return response.data
        return response
    raise ValueError('No pdf_url or file_path available for report')


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for p in reader.pages:
            try:
                parts.append(p.extract_text() or '')
            except Exception:
                parts.append('')
        return '\n'.join(parts)
    except Exception as e:
        logger.debug('pypdf failed: %s', e)
        return ''


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    text = extract_text_from_pdf_bytes(pdf_bytes)
    return len(text.strip()) < TEXT_LENGTH_THRESHOLD

# ---------- OCR ----------

def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI):
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def ocr_with_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError('pytesseract not available')
    gray = img.convert('L')
    w, h = gray.size
    if max(w, h) < 1600:
        factor = max(1, int(1600 / max(w, h)))
        gray = gray.resize((w * factor, h * factor), Image.LANCZOS)
    txt = pytesseract.image_to_string(gray, lang='eng')
    return txt


def ocr_with_openai(img: Image.Image, max_width=1200) -> str:
    if not HAS_OPENAI:
        raise RuntimeError('OpenAI not configured')
    w, h = img.size
    if w > max_width:
        ratio = max_width / float(w)
        img = img.resize((max_width, int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.convert('RGB').save(buf, format='JPEG', quality=75)
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    prompt = 'Extract plain text from this base64 JPEG. Return only the text. IMAGE_BASE64:\n' + b64
    try:
        resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0)
        return resp['choices'][0]['message']['content']
    except Exception as e:
        logger.exception('OpenAI OCR error: %s', e)
        return ''


def do_ocr(pdf_bytes: bytes) -> str:
    imgs = pdf_to_images(pdf_bytes)
    pages = []
    for i, img in enumerate(imgs):
        page_text = ''
        if HAS_PYTESSERACT:
            try:
                page_text = ocr_with_pytesseract(img)
                logger.info('OCR page %d via pytesseract len=%d', i, len(page_text))
            except Exception as e:
                logger.debug('pytesseract failed page %d: %s', i, e)
        if not page_text:
            try:
                page_text = ocr_with_openai(img)
                logger.info('OCR page %d via OpenAI len=%d', i, len(page_text))
            except Exception as e:
                logger.debug('OpenAI OCR failed page %d: %s', i, e)
        pages.append(page_text)
    return '\n\n---PAGE_BREAK---\n\n'.join(pages)

# ---------- Parsing: robust extraction ----------

# synonym sets for many analytes
LABELS = {
    'Hb': ['hb', 'haemoglobin', 'hemoglobin'],
    'RBC': ['rbc', 'erythrocyte count', 'erythrocyte'],
    'HCT': ['hct', 'haematocrit', 'hematocrit'],
    'MCV': ['mcv', 'mean corpuscular volume'],
    'MCH': ['mch', 'mean corpuscular hemoglobin', 'mean corpuscular haemoglobin'],
    'MCHC': ['mchc'],
    'RDW': ['rdw'],
    'WBC': ['wbc', 'white cell count', 'leukocyte', 'leucocyte'],
    'Neutrophils': ['neutrophils', 'neutrophil', 'neutrophil%'],
    'Lymphocytes': ['lymphocytes', 'lymphocyte'],
    'Monocytes': ['monocytes', 'monocyte'],
    'Eosinophils': ['eosinophils'],
    'Basophils': ['basophils'],
    'Platelets': ['platelets', 'thrombocytes', 'platelet count'],
    'CRP': ['crp', 'c-reactive protein'],
    'Creatinine': ['creatinine'],
    'Sodium': ['sodium', 'na'],
    'Potassium': ['potassium', 'k'],
    'Chloride': ['chloride', 'cl'],
    'Urea': ['urea'],
    'ALT': ['alt', 'alanine aminotransferase'],
    'AST': ['ast', 'aspartate aminotransferase'],
    'CK': ['ck', 'creatine kinase'],
}

# build reverse lookup
REVERSE = {}
for k, synonyms in LABELS.items():
    for s in synonyms:
        REVERSE[s.lower()] = k

# value extraction regexes
NUMBER = r'(-?\d+\.?\d*)'
PCT = r'(-?\d+\.?\d*)\s*%'


def find_values(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    lines = [l.strip() for l in re.split(r'\n|\r', text) if l.strip()]

    for line in lines:
        ln = line.lower()
        # quick check: look for any synonym
        for syn, canon in REVERSE.items():
            if syn in ln:
                # try find number after label
                m = re.search(rf'{re.escape(syn)}[^\d\n\r\%\-\+\.]{{0,30}}{NUMBER}', ln)
                if not m:
                    # percentage format
                    m = re.search(rf'{re.escape(syn)}[^\n\r]{{0,30}}{PCT}', ln)
                if m:
                    val = safe_float(m.group(1))
                    # units capture: look in original line for non-digit tokens after value
                    unit_search = re.search(rf'{re.escape(m.group(1))}\s*([a-zA-Z/%^\-0-9]+)', line)
                    units = unit_search.group(1).strip() if unit_search else None
                    results.setdefault(canon, {})['value'] = val
                    if units:
                        results[canon]['units'] = units
                    results[canon]['raw'] = line
        # fallback generic matches like 'Hb: 11.6 g/dL (12.0-16.0)'
        generic = re.findall(rf'([A-Za-z ]{1,20})\s*[:]?\s*{NUMBER}\s*([a-zA-Z/%\-\^0-9]+)?', line)
        for g in generic:
            label = g[0].strip().lower()
            val = safe_float(g[1])
            if not val:
                continue
            found_key = None
            for syn, canon in REVERSE.items():
                if syn in label or label in syn:
                    found_key = canon
                    break
            if found_key:
                results.setdefault(found_key, {})['value'] = val
                results[found_key]['raw'] = line
    # handle percentage-only lines (Neutrophils 88%) previously missed
    for line in lines:
        pct_m = re.search(r'([A-Za-z\s]{2,20})\s*[:]?\s*' + PCT, line)
        if pct_m:
            lab_label = pct_m.group(1).strip().lower()
            val = safe_float(pct_m.group(2))
            for syn, canon in REVERSE.items():
                if syn in lab_label:
                    results.setdefault(canon, {})['value'] = val
                    results[canon]['raw'] = line
    return results

# ---------- Canonical mapping ----------
CANONICAL_ORDER = ['Hb','RBC','HCT','MCV','MCH','MCHC','RDW','WBC','Neutrophils','Lymphocytes','Monocytes','Eosinophils','Basophils','Platelets','NLR','CRP','Creatinine','Sodium','Potassium','Chloride','Urea','ALT','AST','CK']


def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k, v in parsed.items():
        out[k] = {
            'value': safe_float(v.get('value')),
            'units': v.get('units'),
            'raw': v.get('raw')
        }
    # attempt compute NLR if neutrophils and lymphocytes present
    try:
        n = out.get('Neutrophils', {}).get('value')
        l = out.get('Lymphocytes', {}).get('value')
        if n is not None and l is not None and l != 0:
            out['NLR'] = {'value': round(n / l, 2), 'units': None}
    except Exception:
        pass
    return out

# ---------- Route Engine V4 ----------

# color mapping by severity score
COLOR_MAP = {
    5: {'label':'critical','color':'#b91c1c','tw':'bg-red-600','urgency':'high'},
    4: {'label':'severe','color':'#f97316','tw':'bg-orange-500','urgency':'high'},
    3: {'label':'moderate','color':'#f59e0b','tw':'bg-yellow-400','urgency':'medium'},
    2: {'label':'borderline','color':'#facc15','tw':'bg-yellow-300','urgency':'low'},
    1: {'label':'normal','color':'#10b981','tw':'bg-green-500','urgency':'low'},
}


def age_group(age: Optional[float]) -> str:
    if age is None:
        return 'adult'
    try:
        a = float(age)
        ... (truncated by tool)
