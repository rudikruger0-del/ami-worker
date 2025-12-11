# worker.py
"""
AMI Health Worker V5 - single-file worker.py (Max-accuracy mode: local OCR preferred)

Features:
- pypdf for digital text extraction
- pdf2image + pytesseract for scanned PDFs (preferred)
- OpenAI Vision fallback (only if OPENAI_API_KEY is set and pytesseract fails)
- Robust regex parsing for CBC & chemistry, canonical mapping
- Route Engine V5 with ferritin route, stronger microcytic logic, sepsis/high NLR detection
- Trend analysis when previous results exist for same patient_id
- Saves ai_results into Supabase table (env SUPABASE_TABLE)
- Color-coded severity map (Green/Yellow/Orange/Red)
"""

import os
import io
import re
import time
import json
import math
import base64
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

# HTTP client for fallback / downloads
import requests

# PDF and OCR libs
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

# Optional local OCR
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# Supabase client
try:
    from supabase import create_client
    HAS_SUPABASE = True
except Exception:
    HAS_SUPABASE = False

# OpenAI client (the OpenAI class, used as fallback for OCR or model completion if configured)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

from dotenv import load_dotenv

# ---------- config & logging ----------
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
OCR_PAGE_LIMIT = int(os.getenv("OCR_PAGE_LIMIT", "8"))  # limit pages to OCR for speed
MAX_AI_INPUT_CHARS = int(os.getenv("MAX_AI_INPUT_CHARS", "12000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v5")

# Initialize Supabase & OpenAI clients
supabase = None
if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Connected to Supabase")
    except Exception as e:
        logger.exception("Failed to create supabase client: %s", e)
        supabase = None
else:
    logger.warning("Supabase not configured or client missing")

openai_client = None
if HAS_OPENAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    except Exception as e:
        logger.exception("Failed to initialize OpenAI client: %s", e)
        openai_client = None
elif OPENAI_API_KEY:
    logger.warning("OpenAI package not available; OpenAI fallback disabled")

# ---------- Utilities: PDF download / digital extraction ----------

def download_pdf_bytes_from_record(record: Dict[str, Any]) -> bytes:
    """Download PDF bytes either from Supabase storage or an external URL in the record."""
    # Priority: file_path in Supabase storage, then pdf_url
    if record.get("file_path") and supabase:
        path = record["file_path"]
        # supabase.storage.from_(bucket).download returns a response-like object
        res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
        if hasattr(res, "data"):
            return res.data
        # some SDKs return bytes directly
        return res
    elif record.get("pdf_url"):
        url = record["pdf_url"]
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    else:
        raise ValueError("No file_path or pdf_url available on record")

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Try to extract selectable text using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages_text = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages_text.append(t)
        joined = "\n\n".join(pages_text).strip()
        return joined
    except Exception as e:
        logger.warning("pypdf failed: %s", e)
        return ""

def is_scanned_pdf_by_text(pdf_text: str) -> bool:
    """Heuristic: if extracted text length is below threshold, treat as scanned."""
    if not pdf_text:
        return True
    return len(pdf_text.strip()) < TEXT_LENGTH_THRESHOLD

# ---------- OCR: pytesseract preferred, OpenAI Vision fallback ----------

def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI, max_pages: int = OCR_PAGE_LIMIT) -> List[Image.Image]:
    imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
    if max_pages and len(imgs) > max_pages:
        imgs = imgs[:max_pages]
    return imgs

def ocr_image_with_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract not available")
    # Preprocess image (grayscale + thresholding can help)
    gray = img.convert("L")
    w, h = gray.size
    # upscale small images for better OCR
    if max(w, h) < 1200:
        factor = int(max(1, 1200 / max(w, h)))
        gray = gray.resize((w * factor, h * factor), Image.LANCZOS)
    try:
        txt = pytesseract.image_to_string(gray, lang="eng")
    except Exception as e:
        logger.exception("pytesseract error: %s", e)
        txt = ""
    return txt or ""

def openai_vision_ocr_image(img: Image.Image) -> str:
    """Fallback OCR using OpenAI image -> text. Uses OpenAI client if available."""
    if not openai_client:
        raise RuntimeError("OpenAI client not configured")
    # Resize to reasonable width
    max_w = 1200
    w, h = img.size
    if w > max_w:
        ratio = max_w / float(w)
        img = img.resize((max_w, int(h * ratio)), Image.LANCZOS)
    buff = io.BytesIO()
    img.convert("RGB").save(buff, format="JPEG", quality=85)
    b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    # Compose a clear prompt asking for plain extracted text
    system_prompt = "You are an OCR assistant. Extract all visible plain text from the supplied image. Return only the raw extracted text."
    # Use the OpenAI Chat Completions via the OpenAI client (chat capable)
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # choose an image-capable chat model if available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "text", "text": "Extract text from this image (return only text)."},
                                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}
            ],
            temperature=0,
            max_tokens=5000,
        )
        content = resp.choices[0].message.content
        if isinstance(content, list):
            # sometimes content streaming
            return "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return content or ""
    except Exception as e:
        logger.exception("OpenAI OCR failed: %s", e)
        raise

def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    """Run OCR on the PDF pages: prefer pytesseract, fallback to OpenAI vision if configured."""
    try:
        images = pdf_bytes_to_images(pdf_bytes)
    except Exception as e:
        logger.exception("Failed to render PDF to images: %s", e)
        raise

    page_texts: List[str] = []
    for i, img in enumerate(images):
        page_text = ""
        # try pytesseract first
        if HAS_PYTESSERACT:
            try:
                page_text = ocr_image_with_pytesseract(img)
                logger.info("OCR page %d via pytesseract length=%d", i, len(page_text))
            except Exception as e:
                logger.warning("pytesseract failed on page %d: %s", i, e)
                page_text = ""
        # fallback to OpenAI vision if pytesseract missing or returned nothing
        if not page_text and openai_client:
            try:
                page_text = openai_vision_ocr_image(img)
                logger.info("OCR page %d via OpenAI length=%d", i, len(page_text))
            except Exception as e:
                logger.warning("OpenAI OCR failed page %d: %s", i, e)
                page_text = ""
        page_texts.append(page_text)
    return "\n---PAGE_BREAK---\n".join(page_texts)

# ---------- Parsing: robust regex extraction ----------

VALUE_RE = r'(-?\d+(?:\.\d+)?)'
UNIT_RE = r'([a-zA-Z/%\^\-\u00B2\u00B30-9\s\.]+)?'  # allow degree symbols and superscripts
REF_RE = r'\(?(?:ref(?:erence)?[:\s]*)?([0-9\.\-\–to\s/]+)\)?'

COMMON_KEYS = {
    "hb": ["hb", "haemoglobin", "hemoglobin"],
    "rbc": ["rbc", "erythrocyte", "erythrocyte count"],
    "hct": ["hct", "haematocrit", "hematocrit"],
    "mcv": ["mcv", "mean corpuscular volume"],
    "mch": ["mch", "mean corpuscular haemoglobin", "mean corpuscular hemoglobin"],
    "mchc": ["mchc"],
    "rdw": ["rdw"],
    "wbc": ["wbc", "white cell", "white blood cell", "leukocyte", "leucocyte"],
    "neutrophils_pc": ["neutrophils %", "neutrophils%", "neutrophil%","neutrophils"],
    "neutrophils_abs": ["neutrophils absolute", "neutrophil count", "neutrophils x10"],
    "lymphocytes_pc": ["lymphocytes %", "lymphocytes"],
    "monocytes_pc": ["monocytes %", "monocytes"],
    "eosinophils_pc": ["eosinophils %", "eosinophils"],
    "basophils_pc": ["basophils %", "basophils"],
    "platelets": ["platelets", "thrombocytes", "platelet count", "plt"],
    "crp": ["crp", "c-reactive protein", "c reactive protein"],
    "creatinine": ["creatinine", "creat"],
    "sodium": ["sodium", "na"],
    "potassium": ["potassium", "k"],
    "chloride": ["chloride", "cl"],
    "urea": ["urea"],
    "ck": ["ck", "creatine kinase"],
    "alt": ["alt", "alanine aminotransferase", "alanine transaminase"],
    "ast": ["ast", "aspartate aminotransferase"],
    "ferritin": ["ferritin"],
}

LABEL_TO_KEY = {}
for k, labels in COMMON_KEYS.items():
    for lbl in labels:
        LABEL_TO_KEY[lbl.lower()] = k

def normalize_label(label: str) -> str:
    l = re.sub(r'[^a-z0-9% ]', ' ', label.lower())
    l = re.sub(r'\s+', ' ', l).strip()
    return l

def find_key_for_label(label: str) -> Optional[str]:
    l = normalize_label(label)
    if l in LABEL_TO_KEY:
        return LABEL_TO_KEY[l]
    for lbl, key in LABEL_TO_KEY.items():
        if lbl in l or l in lbl:
            return key
    return None

def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    """Scan text lines for analyte labels, values, units and optional reference ranges."""
    results: Dict[str, Dict[str, Any]] = {}
    # split into lines but preserve table-like formats
    lines = [ln.strip() for ln in re.split(r'\r\n|\n|\r', text) if ln.strip()]
    for line in lines:
        ln = line.strip()
        ln_lower = ln.lower()
        # label + value common pattern: "Hb 11.6 g/dL (ref: 12.4-16.7)"
        m_generic = re.findall(r'([A-Za-z\%\.\-/ ]{2,30})\s*[:\s]\s*' + VALUE_RE + r'\s*' + UNIT_RE + r'(?:.*' + REF_RE + r')?', ln)
        if m_generic:
            for grp in m_generic:
                label_raw = grp[0].strip()
                val = grp[1]
                unit = grp[2].strip() if grp[2] else None
                ref = grp[3].strip() if len(grp) > 3 and grp[3] else None
                key = find_key_for_label(label_raw)
                if key:
                    try:
                        v = float(val)
                    except:
                        try:
                            v = float(val.replace(",", "."))
                        except:
                            v = None
                    results.setdefault(key, {})['value'] = v
                    if unit:
                        results.setdefault(key, {})['unit'] = unit
                    if ref:
                        results.setdefault(key, {})['ref'] = ref
                    results.setdefault(key, {})['raw_line'] = ln
        # direct lookups for known labels
        for label, key in LABEL_TO_KEY.items():
            if label in ln_lower:
                # find number following label
                pat = re.compile(re.escape(label) + r'.{0,25}?(' + VALUE_RE + r')', re.IGNORECASE)
                mm = pat.search(ln_lower)
                if mm:
                    raw_val = mm.group(1)
                    try:
                        val = float(raw_val)
                    except:
                        try:
                            val = float(raw_val.replace(",", "."))
                        except:
                            val = None
                    unit_match = re.search(r'(' + UNIT_RE + r')', ln)
                    unit = unit_match.group(1).strip() if unit_match else None
                    results.setdefault(key, {})['value'] = val
                    if unit:
                        results.setdefault(key, {})['unit'] = unit
                    results.setdefault(key, {})['raw_line'] = ln
    # special parsing for neutrophils percentages and absolute x10^9/L forms
    for line in lines:
        if 'neutrophil' in line.lower():
            p = re.search(r'([0-9]{1,3}(?:\.\d+)?)\s*%', line)
            if p:
                try:
                    results.setdefault('neutrophils_pc', {})['value'] = float(p.group(1))
                except:
                    pass
            abs_match = re.search(r'([0-9]{1,3}\.\d{1,3})\s*(?:x10\^?\d?\/?L|x10\^?9\/L|x10\^?9\/L|10\^9\/L|/L)', line)
            if abs_match:
                try:
                    results.setdefault('neutrophils_abs', {})['value'] = float(abs_match.group(1))
                except:
                    pass
    # fallback: simple Hb patterns anywhere
    fallback = re.search(r'\b(hb|haemoglobin|hemoglobin)\b[^\d\n\r]{0,20}(' + VALUE_RE + r')', text, re.IGNORECASE)
    if fallback:
        try:
            results.setdefault('hb', {})['value'] = float(fallback.group(2))
        except:
            pass

    return results

# ---------- Canonical mapping ----------

CANONICAL_KEYS = [
    "Hb","RBC","HCT","MCV","MCH","MCHC","RDW","WBC","Neutrophils","Lymphocytes","Monocytes","Eosinophils","Basophils",
    "NLR","Platelets","Creatinine","CRP","Sodium","Potassium","Chloride","Urea","CK","ALT","AST","Ferritin"
]

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    map_rules = {
        "hb":"Hb","rbc":"RBC","hct":"HCT","mcv":"MCV","mch":"MCH","mchc":"MCHC","rdw":"RDW",
        "wbc":"WBC","neutrophils_pc":"Neutrophils","neutrophils_abs":"Neutrophils",
        "lymphocytes_pc":"Lymphocytes","monocytes_pc":"Monocytes","eosinophils_pc":"Eosinophils",
        "basophils_pc":"Basophils","platelets":"Platelets","creatinine":"Creatinine","crp":"CRP",
        "sodium":"Sodium","potassium":"Potassium","chloride":"Chloride","urea":"Urea","ck":"CK",
        "alt":"ALT","ast":"AST","ferritin":"Ferritin"
    }
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in parsed.items():
        if k in map_rules:
            canon = map_rules[k]
            out.setdefault(canon, {})
            # prefer numeric conversions
            try:
                out[canon]['value'] = float(v.get('value')) if v.get('value') is not None else None
            except:
                out[canon]['value'] = None
            if v.get('unit'):
                out[canon]['unit'] = v.get('unit')
            if v.get('ref'):
                out[canon]['ref'] = v.get('ref')
            out[canon]['raw'] = v.get('raw_line')
    # calculate NLR if percentages or absolute values available
    try:
        neut = out.get('Neutrophils', {}).get('value')
        lymph = out.get('Lymphocytes', {}).get('value')
        if neut is not None and lymph is not None and lymph != 0:
            out['NLR'] = {'value': round(float(neut) / float(lymph), 2), 'unit': None}
    except Exception:
        pass
    return out

# ---------- Route Engine V5: rules, routes, ddx, next steps, severity ----------

# severity map: 1 green / 2 yellow / 3 orange / 4 red / 5 deep red (critical)
SEVERITY_MAP = {
    1: {"label":"LOW","color":"#10B981","tw":"bg-green-500","urgency":"low"},
    2: {"label":"BORDERLINE","color":"#FACC15","tw":"bg-yellow-300","urgency":"low"},
    3: {"label":"MEDIUM","color":"#F59E0B","tw":"bg-orange-400","urgency":"medium"},
    4: {"label":"HIGH","color":"#F97316","tw":"bg-orange-600","urgency":"high"},
    5: {"label":"CRITICAL","color":"#B91C1C","tw":"bg-red-700","urgency":"high"},
}

def age_group_from_age(age: Optional[float]) -> str:
    try:
        if age is None:
            return "adult"
        a = float(age)
        if a < (1/12):
            return "neonate"
        if a < 1:
            return "infant"
        if a < 13:
            return "child"
        if a < 18:
            return "teen"
        if a < 65:
            return "adult"
        return "elderly"
    except:
        return "adult"

def score_severity_for_abnormality(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    """Heuristic severity scoring for individual analytes"""
    if value is None:
        return 1
    try:
        v = float(value)
    except:
        return 1
    # Hb
    if key == "Hb":
        low_cut = 13.0 if sex and sex.lower() == "male" else 12.0
        if age_group in ["child","infant","neonate"]:
            # use pediatric heuristics (simplified)
            if v < 8:
                return 5
            if v < 10:
                return 4
            if v < low_cut:
                return 3
            return 1
        if v < low_cut - 4:
            return 5
        if v < low_cut - 2:
            return 4
        if v < low_cut:
            return 3
        return 1
    # WBC
    if key == "WBC":
        if v > 30:
            return 5
        if v > 20:
            return 4
        if v > 11:
            return 3
        if v < 3:
            return 3
        return 1
    # CRP
    if key == "CRP":
        if v > 200:
            return 5
        if v > 100:
            return 4
        if v > 50:
            return 3
        if v > 10:
            return 2
        return 1
    # Platelets
    if key == "Platelets":
        if v < 10:
            return 5
        if v < 50:
            return 4
        if v < 100:
            return 3
        if v > 1000:
            return 4
        return 1
    # Creatinine (very rough: adult μmol/L)
    if key == "Creatinine":
        if v > 400:
            return 5
        if v > 250:
            return 4
        if v > 120:
            return 3
        return 1
    # CK (rhabdomyolysis)
    if key == "CK":
        if v > 10000:
            return 5
        if v > 5000:
            return 4
        if v > 1000:
            return 3
        return 1
    # Sodium
    if key == "Sodium":
        if v < 120 or v > 160:
            return 5
        if v < 125 or v > 155:
            return 4
        if v < 130 or v > 150:
            return 3
        return 1
    return 1

def route_engine_v5(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Produce patterns, routes, next_steps, ddx, severity and urgency.
    This is intentionally explicit and medical-minded (heuristics).
    """
    age = patient_meta.get("age")
    sex = (patient_meta.get("sex") or "unknown").lower()
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    scores = []
    urgency_levels = []

    def add_pattern(name: str, reason: str, score: int):
        patterns.append({"pattern": name, "reason": reason})
        scores.append(score)

    # Helper to append ddx unique
    def add_ddx(items: List[str]):
        for it in items:
            if it not in ddx:
                ddx.append(it)

    # Hb / anemia
    hb = canonical.get("Hb", {}).get("value")
    mcv = canonical.get("MCV", {}).get("value")
    mch = canonical.get("MCH", {}).get("value")
    rdw = canonical.get("RDW", {}).get("value")
    wbc = canonical.get("WBC", {}).get("value")
    crp = canonical.get("CRP", {}).get("value")
    plate = canonical.get("Platelets", {}).get("value")
    creat = canonical.get("Creatinine", {}).get("value")
    nlr = canonical.get("NLR", {}).get("value")
    sodium = canonical.get("Sodium", {}).get("value")
    ck = canonical.get("CK", {}).get("value")
    ferritin = canonical.get("Ferritin", {}).get("value")

    # anemia detection & microcytic algorithm
    if hb is not None:
        score = score_severity_for_abnormality("Hb", hb, ag, sex)
        if score > 1:
            add_pattern("anemia", f"Hb {hb} g/dL", score)
            # microcytic path
            if mcv is not None and mcv < 80:
                add_pattern("microcytic anemia", f"MCV {mcv} fL", max(score,3))
                routes.append("Iron Deficiency Route")
                add_ddx(["Iron deficiency anemia", "Thalassemia trait", "Chronic blood loss"])
                next_steps.append("Order ferritin and reticulocyte count; check for blood loss (stool/menstrual history)")
                # Ferritin route explicit
                if ferritin is not None:
                    if ferritin < 30:
                        add_pattern("low ferritin", f"Ferritin {ferritin} ng/mL", 4)
                        next_steps.append("Treat or investigate iron deficiency urgently")
                else:
                    next_steps.append("If ferritin not available: order ferritin (essential for microcytic workup)")
            elif mcv is not None and mcv > 100:
                add_pattern("macrocytic anemia", f"MCV {mcv} fL", max(score,3))
                routes.append("Macrocytic Route")
                add_ddx(["Vitamin B12 deficiency", "Folate deficiency", "Alcohol-related","Myelodysplasia"])
                next_steps.append("Order B12, folate, reticulocyte count; review meds and alcohol use")
            else:
                add_pattern("normocytic anemia", "MCV normal or missing", max(2, score))
                routes.append("Normocytic Route")
                add_ddx(["Acute blood loss", "Hemolysis", "Anaemia of chronic disease"])
                next_steps.append("Order reticulocyte, LDH, peripheral smear; consider inflammation-driven anaemia")

    # Infection / inflammation / sepsis suspicion
    if wbc is not None:
        wscore = score_severity_for_abnormality("WBC", wbc, ag, sex)
        if wscore > 1:
            add_pattern("leukocytosis", f"WBC {wbc} x10^9/L", wscore)
            routes.append("Leukocytosis Route")
            add_ddx(["Bacterial infection", "Acute inflammation", "Stress/trauma"])
            # if neutrophilic predominance (we may have absolute or percent)
            neut = canonical.get("Neutrophils", {}).get("value")
            if neut is not None:
                try:
                    n = float(neut)
                    # if percent
                    if n > 70:
                        add_pattern("neutrophilic predominance", f"Neutrophils {n}%", max(3, wscore))
                        routes.append("Bacterial infection route")
                        next_steps.append("Assess clinically for source; consider sepsis workup if unwell; blood cultures if febrile")
                except:
                    pass
    # CRP
    if crp is not None:
        cscore = score_severity_for_abnormality("CRP", crp, ag, sex)
        if cscore > 1:
            add_pattern("elevated inflammatory marker", f"CRP {crp} mg/L", cscore)
            routes.append("Inflammation Route")
            add_ddx(["Bacterial infection","Severe inflammatory state"])
            if crp > 50:
                next_steps.append("Consider urgent clinical review and sepsis pathway if symptomatic")

    # NLR high -> amplify suspicion
    if nlr is not None:
        try:
            nlr_v = float(nlr)
            if nlr_v > 10:
                add_pattern("very high NLR", f"NLR {nlr_v}", 5)
                routes.append("High NLR (sepsis) Route")
                add_ddx(["Severe bacterial infection","Critical inflammatory response"])
                next_steps.append("High suspicion for severe bacterial infection or sepsis; urgent clinical assessment")
            elif nlr_v > 5:
                add_pattern("high NLR", f"NLR {nlr_v}", 4)
                routes.append("High NLR Route")
                next_steps.append("Consider bacterial infection; clinical correlation required")
        except:
            pass

    # Platelets
    if plate is not None:
        pscore = score_severity_for_abnormality("Platelets", plate, ag, sex)
        if plate < 150:
            add_pattern("thrombocytopenia", f"Platelets {plate} x10^9/L", pscore)
            routes.append("Platelet: low route")
            add_ddx(["Immune thrombocytopenia","DIC","Bone marrow suppression"])
            next_steps.append("Check peripheral smear; repeat platelet count; evaluate bleeding risk")
        elif plate > 450:
            add_pattern("thrombocytosis", f"Platelets {plate} x10^9/L", 2)
            routes.append("Platelet: high route")
            add_ddx(["Reactive thrombocytosis","Myeloproliferative disease"])
            next_steps.append("Consider reactive causes; repeat and consider haematology referral if persistent")

    # Creatinine (renal)
    if creat is not None:
        cscore = score_severity_for_abnormality("Creatinine", creat, ag, sex)
        if cscore >= 3:
            add_pattern("elevated creatinine", f"Creatinine {creat} umol/L", cscore)
            routes.append("Renal / AKI route")
            add_ddx(["Acute kidney injury","Chronic kidney disease"])
            next_steps.append("Assess urine output; review nephrotoxins; repeat creatinine urgently and check electrolytes")

    # CK / rhabdo
    if ck is not None:
        cks = score_severity_for_abnormality("CK", ck, ag, sex)
        if cks >= 3:
            add_pattern("elevated CK (rhabdomyolysis physiology)", f"CK {ck} U/L", cks)
            routes.append("Rhabdomyolysis route")
            add_ddx(["Acute muscle injury","Rhabdomyolysis"])
            next_steps.append("Assess for muscle pain/urine changes; check creatinine, urine myoglobin; hydrate and monitor renal function")

    # Sodium disturbances (neurologic risk)
    if sodium is not None:
        sscore = score_severity_for_abnormality("Sodium", sodium, ag, sex)
        if sscore >= 3:
            add_pattern("sodium disturbance", f"Na {sodium}", sscore)
            routes.append("Electrolyte route")
            next_steps.append("Assess neurologic status; correct sodium carefully; consider admission if severe")

    # Combined patterns
    if hb is not None and crp is not None and wbc is not None:
        try:
            if float(hb) < 12 and float(crp) > 20 and float(wbc) > 11:
                add_pattern("anemia with inflammatory/infectious response", "Low Hb with high CRP and leukocytosis", 4)
                routes.append("Inflammation-related anemia route")
                next_steps.append("Treat source of infection; reassess Hb after infection controlled; do ferritin when CRP falls")
        except:
            pass

    # Age/sex notes
    age_note = ""
    if ag == "teen" and sex == "female":
        age_note = "Teenage female — consider menstrual blood loss and iron deficiency as high-likelihood causes."
        next_steps.append("Assess menstrual history; consider urgent ferritin + reticulocyte count")
    elif ag == "elderly":
        age_note = "Elderly patient — broaden differential to include chronic disease and possible malignancy."

    combined_score = max(scores) if scores else 1
    color_entry = SEVERITY_MAP.get(combined_score, SEVERITY_MAP[1])
    urgency_flag = color_entry["urgency"]

    # Build succinct summary lines
    summary_lines = []
    if patterns:
        summary_lines.append("Patterns: " + "; ".join([f"{p['pattern']}" for p in patterns]))
    if routes:
        summary_lines.append("Primary routes: " + "; ".join(routes))
    if ddx:
        summary_lines.append("Top differentials: " + ", ".join(ddx))
    if next_steps:
        summary_lines.append("Immediate actions: " + " | ".join(next_steps))

    final = {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "severity_score": combined_score,
        "urgency_flag": urgency_flag,
        "color": color_entry["color"],
        "tw_class": color_entry["tw"],
        "age_group": ag,
        "age_note": age_note,
        "summary": "\n".join(summary_lines[:8]) if summary_lines else "No significant abnormalities detected."
    }
    return final

# ---------- Trend analysis ----------

def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {"trend": "no_previous"}
    diffs = {}
    for k, v in current.items():
        prev_val = previous.get("canonical", {}).get(k, {}).get("value") if previous else None
        cur_val = v.get("value")
        if prev_val is None or cur_val is None:
            continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {"previous": prev_val, "current": cur_val, "delta": delta, "pct_change": pct}
        except Exception:
            pass
    return {"trend": diffs}

# ---------- Supabase save helper ----------

def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        logger.warning("Supabase not configured; skipping save")
        return
    try:
        payload = {"ai_status": "completed", "ai_results": ai_results, "ai_error": None}
        res = supabase.table(SUPABASE_TABLE).update(payload).eq("id", report_id).execute()
        logger.info("Saved ai_results for %s", report_id)
    except Exception:
        logger.exception("Failed to save ai_results for %s", report_id)

# ---------- Core processing ----------

def process_report(record: Dict[str, Any]) -> Dict[str, Any]:
    report_id = record.get("id") or record.get("report_id")
    logger.info("Processing report %s (path=%s)", report_id, record.get("file_path"))
    try:
        pdf_bytes = download_pdf_bytes_from_record(record)
    except Exception as e:
        logger.exception("Download error: %s", e)
        if supabase:
            supabase.table(SUPABASE_TABLE).update({"ai_status": "failed", "ai_error": str(e)}).eq("id", report_id).execute()
        return {"error": str(e)}

    # Extract digital text first
    pdf_text = extract_text_from_pdf_bytes(pdf_bytes)
    scanned = is_scanned_pdf_by_text(pdf_text)
    logger.info("Report %s scanned=%s (extracted_text_len=%d)", report_id, scanned, len(pdf_text or ""))

    if scanned:
        # Run OCR (local preferred)
        try:
            ocr_text = do_ocr_on_pdf(pdf_bytes)
            merged_text = ocr_text or ""
            logger.info("OCR produced %d chars", len(merged_text))
        except Exception as e:
            logger.exception("OCR failed: %s", e)
            merged_text = ""
    else:
        # Use digital text
        merged_text = pdf_text or ""

    # If still empty, mark failure
    if not merged_text.strip():
        err = "No text extracted after OCR and PDF extraction"
        logger.warning(err)
        if supabase:
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Parse values
    parsed = find_values_in_text(merged_text)
    canonical = canonical_map(parsed)

    # Fetch previous ai_results for trend analysis (by patient_id)
    previous = None
    try:
        if supabase and record.get("patient_id"):
            q = supabase.table(SUPABASE_TABLE).select("ai_results").eq("patient_id", record.get("patient_id")).order("created_at", desc=True).limit(1).execute()
            rows = q.data if hasattr(q, "data") else q
            if rows:
                previous = rows[0].get("ai_results")
    except Exception:
        logger.debug("Failed to fetch previous results", exc_info=True)

    trends = trend_analysis(canonical, previous)

    # Run route engine
    patient_meta = {"age": record.get("age"), "sex": record.get("sex")}
    routes = route_engine_v5(canonical, patient_meta, previous)

    # Decorated flags per analyte
    decorated = {}
    for key, meta in canonical.items():
        val = meta.get("value")
        score = score_severity_for_abnormality(key, val, age_group_from_age(record.get("age")), record.get("sex", "unknown"))
        cmap = SEVERITY_MAP.get(score, SEVERITY_MAP[1])
        decorated[key] = {"raw": meta, "decorated": {"severity": score, "urgency": cmap["urgency"], "color": cmap["color"], "tw_class": cmap["tw"]}}

    ai_results = {
        "canonical": canonical,
        "parsed": parsed,
        "routes": routes,
        "trends": trends,
        "decorated": decorated,
        "raw_text_excerpt": merged_text[:4000],
        "scanned": scanned,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    try:
        save_ai_results_to_supabase(report_id, ai_results)
    except Exception:
        logger.exception("Save failed for %s", report_id)

    logger.info("✅ Report %s processed successfully", report_id)
    return {"success": True, "data": ai_results}

# ---------- Poll loop ----------

def poll_and_process_once(limit: int = 5) -> None:
    if not supabase:
        logger.error("Supabase client not available - poll loop not started")
        return
    logger.info("AMI Worker V5 polling for pending reports...")
    try:
        res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status","pending").limit(limit).execute()
        rows = res.data if hasattr(res, "data") else res
        if not rows:
            logger.debug("No pending reports")
            return
        for r in rows:
            try:
                rid = r.get("id")
                logger.info("Found job: %s", rid)
                # mark processing
                supabase.table(SUPABASE_TABLE).update({"ai_status":"processing"}).eq("id", rid).execute()
                process_report(r)
            except Exception:
                logger.exception("Failed processing record %s", r.get("id"))
    except Exception:
        logger.exception("Polling error")

def main_loop():
    if not supabase:
        logger.error("Supabase not configured - exiting")
        return
    logger.info("Starting poll loop (interval %ds)", POLL_INTERVAL)
    while True:
        try:
            poll_and_process_once(limit=5)
        except Exception:
            logger.exception("Unexpected loop error")
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Poll once then exit")
    parser.add_argument("--test-pdf", help="Path to local PDF for testing")
    args = parser.parse_args()

    if args.test_pdf:
        # local test harness
        path = args.test_pdf
        with open(path, "rb") as f:
            pdfb = f.read()
        dummy = {"id":"local-test","file_path":None,"pdf_url":None,"patient_id":"local-test-patient","age":17,"sex":"female"}
        # override download fn
        def _dl_override(rec):
            return pdfb
        download_pdf_bytes_from_record = _dl_override  # type: ignore
        out = process_report(dummy)
        print(json.dumps(out, indent=2))
    else:
        if args.once:
            poll_and_process_once()
        else:
            main_loop()
