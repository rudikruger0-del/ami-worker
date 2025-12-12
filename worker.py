#!/usr/bin/env python3
# worker.py — AMI Health Worker V5 (Patched)
# FIXED OCR + FIXED DIGITAL PDF EXTRACTION (V4-STABLE)
# Patch: fixed nested/duplicated age_group_from_age and minor cleanup.

import os
import io
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps

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


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
OCR_LANG = os.getenv("OCR_LANG", "eng")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v5")

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


# ---------------------------------------------------------
# PDF DOWNLOAD
# ---------------------------------------------------------

def download_pdf_from_record(record: Dict[str, Any]) -> bytes:
    """Download PDF using pdf_url or Supabase storage path."""
    if "pdf_url" in record and record["pdf_url"]:
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if "file_path" in record and supabase:
        path = record["file_path"]
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
            return res.data if hasattr(res, "data") else res
        except Exception as e:
            logger.exception("Supabase download failed: %s", e)
            raise

    raise ValueError("No valid PDF source found.")


# ---------------------------------------------------------
# DIGITAL PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    """Extract digital text. DO NOT TOUCH — this is stable."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return ""

    parts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            parts.append(txt)
        except Exception:
            parts.append("")

    return "\n".join(parts)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    """If text extraction is extremely short, treat as scanned."""
    txt = extract_text_with_pypdf(pdf_bytes)
    return len(txt.strip()) < TEXT_LENGTH_THRESHOLD


# ---------------------------------------------------------
# OCR (FIXED / RESTORED)
# ---------------------------------------------------------

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    FIXED VERSION:
    - Light grayscale only
    - Mild autocontrast
    - Mild median filter
    - Do NOT upsample aggressively (V4 behaviour)
    """
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img
    except Exception:
        return img


def ocr_image_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract unavailable")

    img = preprocess_image_for_ocr(img)
    config = "--oem 3 --psm 6"

    try:
        raw = pytesseract.image_to_string(img, lang=OCR_LANG, config=config)
    except Exception:
        return ""

    # DO NOT aggressively sanitize numbers — keep original spacing
    cleaned = ""
    for ch in raw:
        if ch == "\x0c":  # form feed
            continue
        cleaned += ch

    return cleaned


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=PDF_RENDER_DPI)


def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    """OCR each page."""
    images = pdf_to_images(pdf_bytes)
    out = []

    for img in images:
        out.append(ocr_image_pytesseract(img))

    return "\n\n---PAGE_BREAK---\n\n".join(out)

# ---------------------------------------------------------
# PART 2 — FIXED V4 CBC PARSER + NEW ANALYTES (Albumin, Ca, CaAdj, CO2, CK-MB)
# ---------------------------------------------------------

# Numeric patterns
VALUE_RE = r'(-?\d+\.\d+|-?\d+)'
PERCENT_RE = r'([0-9]{1,3}\.?\d*)\s*%'

# ---------------------------------------------------------
# MASTER ANALYTE DICTIONARY (extended but V4-safe)
# ---------------------------------------------------------

COMMON_KEYS = {
    # CBC
    "hb": ["hb", "haemoglobin", "hemoglobin"],
    "rbc": ["rbc", "erythrocyte"],
    "hct": ["hct", "hematocrit", "haematocrit"],
    "mcv": ["mcv"],
    "mch": ["mch"],
    "mchc": ["mchc"],
    "rdw": ["rdw"],
    "wbc": ["wbc", "white cell count", "white blood cell"],
    "platelets": ["platelets", "plt", "thrombocytes"],

    # Differential %
    "neutrophils_pc": ["neutrophils %", "neutrophils", "neu %"],
    "lymphocytes_pc": ["lymphocytes %", "lymphocytes", "lym %"],
    "monocytes_pc": ["monocytes %", "monocytes", "mono %"],
    "eosinophils_pc": ["eosinophils %", "eosinophils", "eos %"],
    "basophils_pc": ["basophils %", "basophils", "baso %"],

    # Absolute neutrophils
    "neutrophils_abs": ["neutrophil count", "neutrophils abs", "anc"],

    # Metabolic
    "creatinine": ["creatinine", "creat"],
    "urea": ["urea", "bun"],
    "sodium": ["sodium", "na "],
    "potassium": ["potassium", "k "],
    "chloride": ["chloride", "cl "],

    # LFT
    "alt": ["alt"],
    "ast": ["ast"],

    # CK / CK-MB
    "ck": ["ck", "creatine kinase"],
    "ck_mb": ["ck-mb", "ck mb", "ckmb"],

    # CRP
    "crp": ["crp"],

    # NEW — Albumin
    "albumin": ["albumin", "alb"],

    # NEW — Calcium
    "calcium": ["calcium", "ca "],

    # NEW — Adjusted Calcium
    "calcium_adjusted": ["ca adjusted", "corrected calcium", "ca adj"],

    # NEW — CO₂ / Total CO₂ / HCO₃
    "co2": ["co2", "total co2", "bicarbonate", "hco3"],
}


# QUICK LOOKUP
LABEL_TO_KEY = {}
for key, labels in COMMON_KEYS.items():
    for lbl in labels:
        LABEL_TO_KEY[lbl.lower()] = key


# ---------------------------------------------------------
# Helper: label normalisation
# ---------------------------------------------------------

def normalize_label(lbl: str) -> str:
    """Lowercase, strip punctuation, remove clutter."""
    return re.sub(r"[^a-z0-9 ]", "", lbl.lower()).strip()


def find_key_for_label(raw_label: str) -> Optional[str]:
    """Exact or fuzzy match."""
    cleaned = normalize_label(raw_label)

    if cleaned in LABEL_TO_KEY:
        return LABEL_TO_KEY[cleaned]

    for lbl, key in LABEL_TO_KEY.items():
        if cleaned == lbl:
            return key
        if lbl in cleaned or cleaned in lbl:
            return key

    return None


# ---------------------------------------------------------
# SAFE FLOAT — EXACT V4 BEHAVIOUR
# ---------------------------------------------------------

def safe_float(s: str) -> Optional[float]:
    """
    SAFE VERSION (V4):

    ✔ preserve spacing except between digits
    ✔ allow comma decimals
    ✔ NEVER join letters + digits
    ✔ NEVER merge multiple decimals
    ✔ NEVER "fix" numbers silently
    """

    if not s:
        return None

    # Normalize commas
    s = s.replace(",", ".")

    # Remove letters but keep digits, minus, dot, space
    s = re.sub(r"[^0-9.\- ]", "", s)

    # Join digits split by spaces only: "1 2 . 5" → "12.5"
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)

    s = s.strip()
    if s == "":
        return None

    try:
        return float(s)
    except:
        return None


# ---------------------------------------------------------
# IMPOSSIBLE VALUE FIXER (SAFE)
# ---------------------------------------------------------

def normalize_impossible_values(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:

    def fix(key, value):
        if value is None:
            return None

        try:
            v = float(value)
        except:
            return None

        # Hb: OCR turning 145 → 14.5
        if key == "hb" and v > 40:
            return v / 10

        # Platelets extremely large (OCR: 4500000 → 450)
        if key == "platelets" and v > 1_000_000:
            return v / 1000

        # Neutrophil % must be 0–100
        if key == "neutrophils_pc" and (v < 0 or v > 100):
            return None

        # Calcium cannot be 10+ mmol/L → error → divide by 10
        if key == "calcium" and v > 15:
            return v / 10

        # CK-MB cannot be 90000
        if key == "ck_mb" and v > 10000:
            return None

        return v

    for key in list(results.keys()):
        if "value" in results[key]:
            results[key]["value"] = fix(key, results[key]["value"])

    return results


# ---------------------------------------------------------
# MAIN V4-STYLE PARSER (RESTORED)
# ---------------------------------------------------------

def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    """
    EXACT V4 behaviour:
    ✔ line-by-line parsing
    ✔ label detection
    ✔ extract % values
    ✔ extract numeric values
    ✔ minimal transformation
    ✔ supports new analytes
    """

    results: Dict[str, Dict[str, Any]] = {}

    lines = [ln.strip() for ln in re.split(r"\r|\n", text) if ln.strip()]

    # Pass 1 → labelled lines
    for line in lines:
        low = line.lower()

        for label, key in LABEL_TO_KEY.items():
            if label in low:

                # Percent?
                pm = re.search(PERCENT_RE, line)
                if pm:
                    val = safe_float(pm.group(1))
                    if val is not None:
                        results.setdefault(key, {})["value"] = val
                        results[key]["raw_line"] = line
                    continue

                # Normal numeric values
                m = re.search(VALUE_RE, line)
                if m:
                    val = safe_float(m.group(1))
                    if val is not None:
                        results.setdefault(key, {})["value"] = val
                        results[key]["raw_line"] = line

                # Unit
                if m:
                    um = re.search(rf"{m.group(1)}\s*([A-Za-z/^-]+)", line)
                    if um:
                        results.setdefault(key, {})["unit"] = um.group(1)

    # Pass 2 → absolute neutrophils (ANC)
    for line in lines:
        if "neut" in line.lower():
            abs_m = re.search(r"(\d+\.\d+)\s*x\s*10", line)
            if abs_m:
                val = safe_float(abs_m.group(1))
                if val is not None:
                    results.setdefault("neutrophils_abs", {})["value"] = val
                    results["neutrophils_abs"]["raw_line"] = line

    # Finally → correct impossible OCR values
    results = normalize_impossible_values(results)

    return results
# ---------------------------------------------------------
# PART 3 — CANONICAL MAPPING (RESTORED V4 BEHAVIOUR)
# ---------------------------------------------------------

# Canonical analytes expected by the UI, DB, and Route Engine
CANONICAL_KEYS = [
    "Hb", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW",
    "WBC",
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
    "Platelets",

    # Chemistry
    "Creatinine", "Urea", "Sodium", "Potassium", "Chloride",

    # LFT / muscle enzymes
    "ALT", "AST", "CK", "CK_MB",

    # Inflammation marker
    "CRP",

    # New analytes you added
    "Albumin",
    "Calcium",
    "Calcium_Adjusted",
    "CO2",

    # Derived
    "NLR"
]

# Mapping parsed key → canonical key
CANON_MAP = {
    "hb": "Hb",
    "rbc": "RBC",
    "hct": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",

    "wbc": "WBC",
    "neutrophils_pc": "Neutrophils",
    "neutrophils_abs": "Neutrophils",
    "lymphocytes_pc": "Lymphocytes",
    "monocytes_pc": "Monocytes",
    "eosinophils_pc": "Eosinophils",
    "basophils_pc": "Basophils",

    "platelets": "Platelets",

    "creatinine": "Creatinine",
    "urea": "Urea",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",

    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ck_mb": "CK_MB",

    "crp": "CRP",

    # NEW analytes from your request
    "albumin": "Albumin",
    "calcium": "Calcium",
    "calcium_adjusted": "Calcium_Adjusted",
    "co2": "CO2",
}


def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    RESTORED EXACT V4 BEHAVIOUR:
    - Only map keys that exist in parsed
    - Never invent values
    - Never overwrite good data
    - Preserve units and raw lines
    - Adds NLR if neutrophils + lymphocytes present
    """

    out: Dict[str, Dict[str, Any]] = {}

    for p_key, pdata in parsed.items():
        canon = CANON_MAP.get(p_key)
        if not canon:
            continue  # ignore unknown fields

        value = pdata.get("value")
        unit = pdata.get("unit")
        raw = pdata.get("raw_line")

        out[canon] = {"value": value}

        if isinstance(unit, str):
            out[canon]["unit"] = unit

        if raw:
            out[canon]["raw"] = raw

    # ---------------------------------------------------------
    # NLR — EXACT V4 LOGIC
    # ---------------------------------------------------------
    try:
        neut = out.get("Neutrophils", {}).get("value")
        lymph = out.get("Lymphocytes", {}).get("value")

        if neut is not None and lymph is not None and lymph > 0:
            out["NLR"] = {"value": round(float(neut) / float(lymph), 2)}
    except Exception:
        pass

    return out
# ---------------------------------------------------------
# PART 4 — DECORATED VALUES (FLAGS, COLORS, RISK BARS)
# ---------------------------------------------------------

FLAG_COLORS = {
    "normal": "#ffffff",   # white
    "low":    "#f59e0b",   # orange
    "high":   "#b91c1c"    # red
}

COLOR_MAP = {
    1: {"label": "normal",   "color": "#10b981", "tw": "bg-green-500",  "urgency": "low"},
    2: {"label": "mild",     "color": "#facc15", "tw": "bg-yellow-300", "urgency": "low"},
    3: {"label": "moderate", "color": "#f59e0b", "tw": "bg-yellow-400", "urgency": "medium"},
    4: {"label": "severe",   "color": "#ef4444", "tw": "bg-red-500",    "urgency": "high"},
    5: {"label": "critical", "color": "#b91c1c", "tw": "bg-red-700",    "urgency": "high"},
}


# ------------------------------
# FLAGGING LOGIC (same as V4)
# ------------------------------
def flag_for_key(key: str, value: Optional[float], sex: str = "unknown"):
    if value is None:
        return "normal", FLAG_COLORS["normal"]

    try:
        v = float(value)
    except:
        return "normal", FLAG_COLORS["normal"]

    k = key.lower()

    # Haemoglobin
    if k == "hb":
        low = 12 if sex.lower() == "female" else 13
        high = 15.5 if sex.lower() == "female" else 17.5
        if v < low:  return "low", FLAG_COLORS["low"]
        if v > high: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # WBC
    if k == "wbc":
        if v < 4:  return "low", FLAG_COLORS["low"]
        if v > 11: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Platelets
    if k == "platelets":
        if v < 150: return "low", FLAG_COLORS["low"]
        if v > 450: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Creatinine
    if k == "creatinine":
        if v < 45:  return "low", FLAG_COLORS["low"]
        if v > 120: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # CRP
    if k == "crp":
        if v <= 10: return "normal", FLAG_COLORS["normal"]
        if v <= 50: return "high", FLAG_COLORS["low"]
        return "high", FLAG_COLORS["high"]

    # Sodium
    if k == "sodium":
        if v < 135: return "low", FLAG_COLORS["low"]
        if v > 145: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Potassium
    if k == "potassium":
        if v < 3.5: return "low", FLAG_COLORS["low"]
        if v > 5.1: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Chloride
    if k == "chloride":
        if v < 98:  return "low", FLAG_COLORS["low"]
        if v > 107: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Albumin
    if k == "albumin":
        if v < 35: return "low", FLAG_COLORS["low"]
        if v > 50: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Calcium
    if k == "calcium":
        if v < 2.1: return "low", FLAG_COLORS["low"]
        if v > 2.6: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # Adjusted Calcium
    if k == "calcium_adjusted":
        if v < 2.1: return "low", FLAG_COLORS["low"]
        if v > 2.6: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # CO₂
    if k == "co2":
        if v < 22: return "low", FLAG_COLORS["low"]
        if v > 29: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # CK-MB
    if k == "ck_mb":
        if v > 7: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    # LFT / Muscle markers
    if k in ("alt", "ast", "ck"):
        if v > 200: return "high", FLAG_COLORS["high"]
        if v > 100: return "high", FLAG_COLORS["low"]
        return "normal", FLAG_COLORS["normal"]

    # MCV
    if k == "mcv":
        if v < 80: return "low", FLAG_COLORS["low"]
        if v > 100: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    return "normal", FLAG_COLORS["normal"]


# ------------------------------
# RISK BARS — UI only (0–100%)
# ------------------------------
def risk_percentage_for_key(key: str, value: Optional[float]) -> int:
    if value is None:
        return 0

    try:
        v = float(value)
    except:
        return 0

    k = key.lower()

    if k == "crp":
        return min(100, int((v / 200) * 100))

    if k == "wbc":
        if v <= 11: return int((v/11)*20)
        if v <= 20: return 30 + int(((v-11)/9)*30)
        return min(100, 60 + int(((v-20)/30)*40))

    if k in ("neutrophils", "nlr"):
        if v <= 3: return int((v/3)*10)
        if v <= 6: return 15 + int(((v-3)/3)*25)
        if v <= 10: return 40 + int(((v-6)/4)*30)
        return min(100, 70 + int(((v-10)/30)*30))

    if k == "creatinine":
        if v <= 120: return int((v/120)*20)
        if v <= 200: return 25 + int(((v-120)/80)*30)
        return min(100, 60 + int(((v-200)/300)*40))

    if k == "platelets":
        if 100 <= v <= 450: return 5
        if v < 100: return min(100, 30 + int(((100-v)/100)*70))
        if v > 450: return min(100, 20 + int(((v-450)/1000)*80))

    if k == "hb":
        if v >= 12: return 5
        if v >= 10: return 20
        if v >= 8: return 50
        return min(100, 70 + int(((12-v)/12)*30))

    return 0


def age_group_from_age(age: Optional[int]) -> str:
    """Return age group string used by route engine."""
    if age is None:
        return "adult"

    try:
        a = int(age)
    except:
        return "adult"

    if a < 1:
        return "infant"
    if a < 13:
        return "child"
    if a < 18:
        return "teen"
    if a < 65:
        return "adult"
    return "elderly"

# ---------------------------------------------------------
# PART 5 — ROUTE ENGINE (V4 LOGIC + NEW ANALYTES + SEVERITY TEXT)
# ---------------------------------------------------------

def score_severity_for_abnormality(key: str, value: Optional[float],
                                   age_group: str, sex: str) -> int:
    """
    INTERNAL severity scoring (1–5). NOT exposed to the user.
    Returns numeric score used to compute severity_text.
    """

    if value is None:
        return 1

    try:
        v = float(value)
    except:
        return 1

    k = key.lower()

    # Haemoglobin
    if k == "hb":
        low = 12 if sex.lower() == "female" else 13
        if v < low - 3: return 4
        if v < low: return 3
        return 1

    # WBC
    if k == "wbc":
        if v > 30: return 5
        if v > 20: return 4
        if v > 11: return 3
        return 1

    # CRP
    if k == "crp":
        if v > 200: return 4
        if v > 100: return 3
        if v > 10:  return 2
        return 1

    # Neutrophils/NLR
    if k in ("neutrophils", "nlr"):
        if v > 12: return 5
        if v > 7:  return 4
        if v > 3:  return 3
        return 1

    # Creatinine
    if k == "creatinine":
        if v > 400: return 5
        if v > 200: return 4
        if v > 120: return 3
        return 1

    # Potassium
    if k == "potassium":
        if v < 2.8 or v > 6.5: return 5
        if v < 3.2 or v > 6.0: return 4
        if v < 3.5 or v > 5.5: return 3
        return 1

    # Sodium
    if k == "sodium":
        if v < 120 or v > 160: return 5
        if v < 125 or v > 155: return 4
        if v < 130 or v > 150: return 3
        return 1

    # Platelets
    if k == "platelets":
        if v < 10: return 5
        if v < 30: return 4
        if v < 100: return 3
        if v > 1000: return 4
        return 1

    # CK
    if k == "ck":
        if v > 5000: return 5
        if v > 1000: return 4
        if v > 300:  return 3
        return 1

    # CK-MB
    if k == "ck_mb":
        if v > 30: return 4
        if v > 7:  return 3
        return 1

    # LFTs
    if k in ("alt", "ast"):
        if v > 500: return 4
        if v > 200: return 3
        return 1

    # Albumin
    if k == "albumin":
        if v < 25: return 4
        if v < 35: return 2
        return 1

    # Calcium
    if k == "calcium":
        if v < 1.8 or v > 3.0: return 4
        if v < 2.1 or v > 2.6: return 3
        return 1

    # Adjusted Calcium
    if k == "calcium_adjusted":
        if v < 1.8 or v > 3.0: return 4
        if v < 2.1 or v > 2.6: return 3
        return 1

    # CO₂
    if k == "co2":
        if v < 16 or v > 40: return 4
        if v < 22 or v > 29: return 3
        return 1

    return 1  # default normal


def severity_text_from_score(score: int) -> str:
    if score <= 1: return "normal"
    if score == 2: return "mild"
    if score == 3: return "moderate"
    if score == 4: return "severe"
    return "critical"


# ---------------------------------------------------------
# ROUTE ENGINE (RESTORED V4 + NEW ANALYTES)
# ---------------------------------------------------------

def route_engine(canonical: Dict[str, Dict[str, Any]],
                 patient_meta: Dict[str, Any],
                 previous: Optional[Dict[str, Any]] = None):

    age = patient_meta.get("age")
    sex = patient_meta.get("sex", "unknown")
    ag = age_group_from_age(age)

    # Pull values safely
    def gv(k): return canonical.get(k, {}).get("value")

    Hb = gv("Hb")
    MCV = gv("MCV")
    WBC = gv("WBC")
    Neut = gv("Neutrophils")
    Lym = gv("Lymphocytes")
    CRP = gv("CRP")
    Plt = gv("Platelets")
    Creat = gv("Creatinine")
    Na = gv("Sodium")
    K = gv("Potassium")
    Ca = gv("Calcium")
    CaAdj = gv("Calcium_Adjusted")
    Alb = gv("Albumin")
    CO2 = gv("CO2")
    ALT = gv("ALT")
    AST = gv("AST")
    CK = gv("CK")
    CKMB = gv("CK_MB")

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    severity_scores = []

    def add_pattern(name, reason, sev):
        patterns.append({"pattern": name, "reason": reason})
        severity_scores.append(sev)

    # -----------------------------
    # ANEMIA (V4 logic)
    # -----------------------------
    if Hb is not None:
        low = 12 if sex.lower() == "female" else 13
        if Hb < low:
            add_pattern("anemia", f"Hb {Hb}", 3)
            if MCV is not None and MCV < 80:
                add_pattern("microcytosis", f"MCV {MCV}", 3)
                routes.append("Iron-deficiency route")
                ddx += ["Iron deficiency anemia", "Thalassemia trait"]
                next_steps.append("Order ferritin + reticulocytes.")
            elif MCV is not None and MCV > 100:
                add_pattern("macrocytosis", f"MCV {MCV}", 3)
                routes.append("Macrocytosis route")
                ddx += ["B12 deficiency", "Folate deficiency"]
                next_steps.append("Order B12 + folate.")
            else:
                routes.append("Normocytic anemia route")
                ddx += ["Acute blood loss", "Anemia of inflammation"]
                next_steps.append("Order reticulocytes; evaluate bleeding.")

    # -----------------------------
    # INFECTION / INFLAMMATION
    # -----------------------------
    sepsis_flag = False

    if WBC and WBC > 11:
        add_pattern("leukocytosis", f"WBC {WBC}", 3)
        ddx += ["Infection", "Inflammation"]
        routes.append("Infection route")

    if Neut and Neut > 80:
        add_pattern("neutrophilia", f"Neutrophils {Neut}%", 3)
        routes.append("Bacterial infection route")

    if CRP and CRP > 10:
        add_pattern("elevated CRP", f"CRP {CRP}", 2)
        ddx += ["Inflammation", "Possible infection"]

    # V4-style simple sepsis flag
    if WBC and CRP and WBC > 20 and CRP > 100:
        sepsis_flag = True
        ddx.append("Sepsis")
        routes.append("Sepsis consideration")
        next_steps.append("Urgent clinical assessment recommended.")

    # -----------------------------
    # PLATELETS
    # -----------------------------
    if Plt is not None:
        if Plt < 150:
            add_pattern("thrombocytopenia", f"Platelets {Plt}", 3)
            ddx += ["ITP", "Bone marrow suppression"]
        elif Plt > 450:
            add_pattern("thrombocytosis", f"Platelets {Plt}", 2)
            ddx += ["Reactive thrombocytosis"]

    # -----------------------------
    # RENAL
    # -----------------------------
    if Creat is not None and Creat > 120:
        add_pattern("renal impairment", f"Creatinine {Creat}", 3)
        routes.append("Renal route")
        ddx.append("Possible AKI")
        next_steps.append("Repeat creatinine; check hydration status.")

    # -----------------------------
    # ELECTROLYTES
    # -----------------------------
    if K is not None:
        if K < 3.0:
            add_pattern("hypokalemia", f"K {K}", 3)
            next_steps.append("Correct potassium.")
        elif K > 5.5:
            add_pattern("hyperkalemia", f"K {K}", 4)
            next_steps.append("Urgent K correction + ECG.")

    if Na is not None:
        if Na < 130 or Na > 150:
            add_pattern("sodium derangement", f"Na {Na}", 3)

    # -----------------------------
    # NEW ANALYTES
    # -----------------------------
    if Alb is not None and Alb < 35:
        add_pattern("low albumin", f"Albumin {Alb}", 2)
        ddx += ["Chronic illness", "Nutritional deficiency"]

    if Ca is not None:
        if Ca < 2.1: add_pattern("hypocalcemia", f"Calcium {Ca}", 2)
        elif Ca > 2.6: add_pattern("hypercalcemia", f"Calcium {Ca}", 3)

    if CaAdj is not None:
        if CaAdj < 2.1: add_pattern("adjusted hypocalcemia", f"CaAdj {CaAdj}", 2)
        elif CaAdj > 2.6: add_pattern("adjusted hypercalcemia", f"CaAdj {CaAdj}", 3)

    if CO2 is not None:
        if CO2 < 22: add_pattern("low CO₂", f"CO₂ {CO2}", 2)
        elif CO2 > 29: add_pattern("high CO₂", f"CO₂ {CO2}", 2)

    if CK is not None and CK > 1000:
        add_pattern("possible rhabdomyolysis", f"CK {CK}", 4)
        ddx.append("Rhabdomyolysis")
        next_steps.append("Check urine, fluids, repeat CK.")

    if CKMB is not None and CKMB > 7:
        add_pattern("elevated CK-MB", f"CK-MB {CKMB}", 3)
        ddx.append("Possible myocardial involvement")

    # ---------------------------------------------------------
    # SIMPLE DIAGNOSTIC POSSIBILITIES (accurate + safe)
    # ---------------------------------------------------------
    diag_poss = []

    if WBC and WBC > 11:
        diag_poss.append(f"Infection likely — WBC {WBC}")

    if sepsis_flag:
        diag_poss.append("Sepsis possible (WBC + CRP very high)")

    if Hb and Hb < (12 if sex.lower()=="female" else 13):
        diag_poss.append(f"Anemia — Hb {Hb}")

    if Creat and Creat > 120:
        diag_poss.append(f"Renal impairment — Creatinine {Creat}")

    if not diag_poss:
        diag_poss.append("No major abnormalities detected")

    # ---------------------------------------------------------
    # FINAL SEVERITY
    # ---------------------------------------------------------
    if severity_scores:
        max_sev = max(severity_scores)
    else:
        max_sev = 1

    severity_text = severity_text_from_score(max_sev)
    color_entry = COLOR_MAP[max_sev]

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "severity_text": severity_text,
        "diagnostic_possibilities": diag_poss,
        "urgency_flag": color_entry["urgency"],
        "color": color_entry["color"],
        "tw_class": color_entry["tw"],
        "age_group": ag,
        "age_note": ("Teenage female — consider menstrual blood loss."
                     if ag == "teen" and sex.lower() == "female" else ""),
        "summary": "Diagnostic possibilities:\n• " + "\n• ".join(diag_poss)
    }
# ---------------------------------------------------------
# PART 6 — TRENDS + SAVE + PROCESS_REPORT + POLLER
# ---------------------------------------------------------

def trend_analysis(current: Dict[str, Dict[str, Any]],
                   previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Simple V4-style trend analysis: shows % change where possible."""
    if not previous:
        return {"trend": "no_previous"}

    diffs = {}
    for key, val in current.items():
        cur_val = val.get("value")
        prev_val = None

        if isinstance(previous, dict):
            prev_val = previous.get("canonical", {}).get(key, {}).get("value")

        if cur_val is None or prev_val is None:
            continue

        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[key] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": delta,
                "pct_change": pct
            }
        except:
            continue

    if diffs:
        return {"trend": diffs}
    else:
        return {"trend": "no_change"}


# ---------------------------------------------------------
# SAVE RESULTS INTO SUPABASE
# ---------------------------------------------------------

def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        logger.warning("Supabase client not available — skipping save.")
        return

    try:
        payload = {
            "ai_status": "completed",
            "ai_results": ai_results,
            "ai_error": None
        }
        supabase.table(SUPABASE_TABLE).update(payload).eq("id", report_id).execute()
        logger.info(f"Saved ai_results for report {report_id}")
    except Exception as e:
        logger.exception("Failed to save ai_results: %s", e)


# ---------------------------------------------------------
# PROCESS_REPORT — MAIN ORCHESTRATION
# ---------------------------------------------------------

def process_report(record: Dict[str, Any]) -> None:
    report_id = record.get("id") or record.get("report_id") or "<unknown>"
    logger.info(f"Processing report {report_id}")

    # 1) Download PDF
    try:
        pdf_bytes = download_pdf_from_record(record)
    except Exception as e:
        logger.exception("Failed to download PDF: %s", e)
        if supabase:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", report_id).execute()
        return

    # 2) Check if scanned
    scanned = is_scanned_pdf(pdf_bytes)

    # Extract text
    if scanned:
        if not HAS_PYTESSERACT:
            err = "Scanned PDF detected but pytesseract is unavailable."
            logger.error(err)
            if supabase:
                supabase.table(SUPABASE_TABLE).update({
                    "ai_status": "failed",
                    "ai_error": err
                }).eq("id", report_id).execute()
            return
        text = do_ocr_on_pdf(pdf_bytes)
    else:
        text = extract_text_with_pypdf(pdf_bytes)

    # 3) Parse CBC values
    parsed = find_values_in_text(text)

    # 4) Canonical mapping
    canonical = canonical_map(parsed)

    # 5) Get previous result for trends
    previous = None
    if supabase:
        try:
            pid = record.get("patient_id")
            if pid:
                q = (
                    supabase.table(SUPABASE_TABLE)
                    .select("ai_results")
                    .eq("patient_id", pid)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )

                rows = q.data if hasattr(q, "data") else q
                if rows:
                    previous = rows[0].get("ai_results")
        except Exception as e:
            logger.debug("Trend lookup failed: %s", e)

    trends = trend_analysis(canonical, previous)

    # 6) Route engine
    patient_meta = {
        "age": record.get("age"),
        "sex": record.get("sex", "unknown")
    }
    routes = route_engine(canonical, patient_meta, previous)

    # 7) DECORATED ANALYTES (Part 4 logic)
    decorated = {}
    sex = record.get("sex", "unknown")

    for key in CANONICAL_KEYS:
        item = canonical.get(key, {})
        value = item.get("value")

        flag, color = flag_for_key(key, value, sex)

        # internal severity score
        sev_num = score_severity_for_abnormality(
            key, value, age_group_from_age(record.get("age")), sex
        )
        sev_text = severity_text_from_score(sev_num)

        # UI risk bar
        pct = risk_percentage_for_key(key, value)
        risk_color = (
            "#b91c1c" if pct >= 80 else
            "#ef4444" if pct >= 60 else
            "#f59e0b" if pct >= 40 else
            "#facc15" if pct >= 20 else
            "#10b981"
        )

        decorated[key] = {
            "value": value,
            "unit": item.get("unit"),
            "flag": flag,
            "color": color,
            "severity_text": sev_text,
            "risk_bar": {"percentage": pct, "color": risk_color}
        }

    # 8) Build final ai_results
    ai_results = {
        "canonical": canonical,
        "parsed": parsed,
        "routes": routes,
        "decorated": decorated,
        "trends": trends,
        "raw_text_excerpt": text[:5000],
        "scanned": scanned,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # 9) Save results
    try:
        save_ai_results_to_supabase(report_id, ai_results)
    except Exception as e:
        logger.exception("Failed saving final results: %s", e)


# ---------------------------------------------------------
# POLLING LOOP — SAME AS V4 (stable)
# ---------------------------------------------------------

def poll_and_process():
    if not supabase:
        logger.error("Supabase client not configured. Worker cannot run.")
        return

    logger.info(f"Starting polling loop: interval={POLL_INTERVAL}s, table={SUPABASE_TABLE}")

    while True:
        try:
            res = (
                supabase.table(SUPABASE_TABLE)
                .select("*")
                .eq("ai_status", "pending")
                .limit(10)
                .execute()
            )

            rows = res.data if hasattr(res, "data") else res

            if rows:
                for r in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update(
                            {"ai_status": "processing"}
                        ).eq("id", r.get("id")).execute()

                        process_report(r)

                    except Exception as e:
                        logger.exception("Processing error for ID %s: %s", r.get("id"), e)
                        try:
                            supabase.table(SUPABASE_TABLE).update({
                                "ai_status": "failed",
                                "ai_error": str(e)
                            }).eq("id", r.get("id")).execute()
                        except:
                            pass
            else:
                logger.debug("No pending reports.")

        except Exception as e:
            logger.exception("Poll loop error: %s", e)

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------
# CLI ENTRYPOINT
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMI Health Worker V5 — final engine")
    parser.add_argument("--test-pdf", help="Path to local PDF for testing")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdfb = f.read()

        dummy_record = {
            "id": "local-test",
            "patient_id": "local-patient",
            "age": 30,
            "sex": "female",
            "file_path": args.test_pdf
        }

        # override downloader
        globals()["download_pdf_from_record"] = lambda _r: pdfb

        process_report(dummy_record)
        print("Test run complete.")

    else:
        poll_and_process()
