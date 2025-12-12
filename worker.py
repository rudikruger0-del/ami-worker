#!/usr/bin/env python3

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

# Optional imports
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

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
OCR_LANG = os.getenv("OCR_LANG", "eng")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v6")

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None

# ---------------------------
# PDF DOWNLOAD
# ---------------------------

def download_pdf_from_record(record: Dict[str, Any]) -> bytes:
    if "pdf_url" in record and record["pdf_url"]:
        import requests
        url = record["pdf_url"]
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content

    elif "file_path" in record and supabase:
        path = record["file_path"]
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
            if hasattr(res, "data"):
                return res.data
            return res
        except Exception as e:
            logger.exception("Supabase download failed: %s", e)
            raise

    else:
        raise ValueError("Missing 'pdf_url' or 'file_path'")

# ---------------------------
# PDF TEXT EXTRACTION
# ---------------------------

def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except:
        return ""

    parts = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except:
            t = ""
        parts.append(t)

    return "\n".join(parts)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    txt = extract_text_with_pypdf(pdf_bytes)
    return len(txt.strip()) < TEXT_LENGTH_THRESHOLD

# ---------------------------
# OCR ENGINE (pytesseract)
# ---------------------------

def preprocess_image_for_ocr(img: Image.Image, target_min_dim: int = 1600) -> Image.Image:
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        w, h = img.size
        if max(w, h) < target_min_dim:
            factor = max(1, int(target_min_dim / max(w, h)))
            img = img.resize((w * factor, h * factor), Image.LANCZOS)
        return img
    except:
        return img


def ocr_image_pytesseract(img: Image.Image, lang: str = OCR_LANG) -> str:
    if not HAS_PYTESSERACT:
        return ""

    img2 = preprocess_image_for_ocr(img)
    config = "--oem 3 --psm 6"

    try:
        t = pytesseract.image_to_string(img2, lang=lang, config=config)
    except:
        t = ""

    t = "".join(ch if (31 < ord(ch) < 127 or ch in "\n\r\t") else " " for ch in t)
    return t


def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=dpi)


def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    images = pdf_to_images(pdf_bytes)
    texts = []
    for img in images:
        text = ocr_image_pytesseract(img)
        texts.append(text)
    return "\n---PAGE_BREAK---\n".join(texts)

# ---------------------------
# SMART NUMERIC CLEANER
# ---------------------------

def safe_float(text: str) -> Optional[float]:
    if text is None:
        return None

    text = re.sub(r"/\s*7", "", text)
    text = text.replace(",", ".")
    text = re.sub(r"[^0-9.\-\s]", "", text)
    text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)

    if text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    text = text.strip()
    if text == "":
        return None

    try:
        return float(text)
    except:
        return None

# ---------------------------
# TABLE PARSER V6 (NEW)
# ---------------------------

TABLE_LABEL_MAP = {
    "hb": "Hb",
    "hgb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",

    "rbc": "RBC",
    "hct": "HCT",
    "haematocrit": "HCT",

    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",

    "wbc": "WBC",
    "neut": "Neutrophils",
    "neutrophils": "Neutrophils",
    "lymph": "Lymphocytes",
    "mono": "Monocytes",
    "mono abs": "Monocytes",
    "eos": "Eosinophils",
    "baso": "Basophils",

    "plt": "Platelets",
    "platelets": "Platelets",

    "crp": "CRP",
    "creat": "Creatinine",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "bun": "Urea",

    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "cl": "Chloride",
    "chloride": "Chloride",

    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ck-mb": "CK_MB",

    "ca": "Calcium",
    "ca adj": "Calcium_Adjusted",
    "albumin": "Albumin",
    "co2": "CO2",
    "tco2": "CO2",
    "total co2": "CO2",
}


def normalize_label_for_table(label: str) -> str:
    label = label.lower().strip()
    label = re.sub(r"[^a-z0-9 ]", "", label)
    return label


def map_table_label(label: str) -> Optional[str]:
    l = normalize_label_for_table(label)
    if l in TABLE_LABEL_MAP:
        return TABLE_LABEL_MAP[l]

    for key in TABLE_LABEL_MAP:
        if key in l:
            return TABLE_LABEL_MAP[key]

    return None


def parse_table_block(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for ln in lines:
        ln_clean = re.sub(r"\s+", " ", ln.strip())
        if ln_clean == "":
            continue

        tokens = ln_clean.split(" ")

        if len(tokens) < 2:
            continue

        label = tokens[0]
        mapped = map_table_label(label)
        if not mapped:
            continue

        numeric = None
        flag = None

        for t in tokens[1:]:
            v = safe_float(t)
            if v is not None:
                numeric = v
            elif t.upper() in ("L", "H", "LL", "HH"):
                flag = t.upper()

        if numeric is not None:
            out[mapped] = {"value": numeric, "raw_line": ln_clean}
            if flag:
                out[mapped]["flag"] = flag

    return out
# ---------------------------
# FALLBACK LINE PARSER (non-table)
# ---------------------------

LABEL_TO_KEY = {
    "hb": "Hb",
    "hgb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",

    "rbc": "RBC",
    "hct": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",

    "wbc": "WBC",
    "white cell": "WBC",
    "leukocyte": "WBC",
    "leucocyte": "WBC",

    "neut": "Neutrophils",
    "neutrophils": "Neutrophils",
    "lymph": "Lymphocytes",
    "lymphocytes": "Lymphocytes",
    "mono": "Monocytes",
    "monocytes": "Monocytes",
    "eos": "Eosinophils",
    "eosinophils": "Eosinophils",
    "baso": "Basophils",
    "basophils": "Basophils",

    "plt": "Platelets",
    "platelets": "Platelets",

    "crp": "CRP",
    "creat": "Creatinine",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "bun": "Urea",

    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "cl": "Chloride",
    "chloride": "Chloride",

    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ckmb": "CK_MB",
    "ck-mb": "CK_MB",
}


def normalize_label_general(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9 ]", "", t)
    t = t.strip()
    return t


def find_key_for_label_general(lbl: str) -> Optional[str]:
    lbl2 = normalize_label_general(lbl)
    if lbl2 in LABEL_TO_KEY:
        return LABEL_TO_KEY[lbl2]

    for k in LABEL_TO_KEY:
        if k in lbl2:
            return LABEL_TO_KEY[k]

    return None


def parse_lines_fallback(lines: List[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for ln in lines:
        ln_clean = re.sub(r"\s+", " ", ln.strip())
        if ln_clean == "":
            continue

        m = re.match(r"([A-Za-z0-9\-/ ]{2,30})[:\-\s]+\s*([0-9\.\-,]+)", ln_clean)
        if not m:
            continue

        label_raw = m.group(1).strip()
        val_raw = m.group(2).strip()

        key = find_key_for_label_general(label_raw)
        if not key:
            continue

        v = safe_float(val_raw)
        if v is None:
            continue

        out[key] = {"value": v, "raw_line": ln_clean}

    return out

# ---------------------------
# MASTER PARSER — COMBINES TABLE + FALLBACK
# ---------------------------

def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    text = "".join(ch if (31 < ord(ch) < 127 or ch in "\n\r\t") else " " for ch in text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    parsed_table = parse_table_block(lines)
    parsed_lines = parse_lines_fallback(lines)

    merged: Dict[str, Dict[str, Any]] = {}
    merged.update(parsed_lines)
    merged.update(parsed_table)

    return merged

# ---------------------------
# CANONICAL MAPPING
# ---------------------------

CANONICAL_KEYS = [
    "Hb", "MCV", "MCH", "MCHC", "RDW", "WBC", "Neutrophils", "Lymphocytes",
    "Monocytes", "Eosinophils", "Basophils", "NLR", "Platelets", "Creatinine",
    "CRP", "Sodium", "Potassium", "Chloride", "Urea", "RBC", "HCT", "ALT", "AST", "CK",
    "CK_MB", "Calcium", "Calcium_Adjusted", "Albumin", "CO2"
]


def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for k, v in parsed.items():
        if k not in CANONICAL_KEYS:
            continue

        val = v.get("value")
        out[k] = {"value": val}

        if "raw_line" in v:
            out[k]["raw"] = v["raw_line"]
        if "flag" in v:
            out[k]["flag_from_table"] = v["flag"]

    neut = out.get("Neutrophils", {}).get("value")
    lymph = out.get("Lymphocytes", {}).get("value")
    if neut is not None and lymph is not None and lymph != 0:
        out["NLR"] = {"value": round(neut / lymph, 2)}

    return out

# ---------------------------
# AGE GROUP
# ---------------------------

def age_group_from_age(age: Optional[float]) -> str:
    if age is None:
        return "adult"
    try:
        a = float(age)
    except:
        return "adult"

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

# ---------------------------
# SEVERITY TEXT
# ---------------------------

def severity_text_from_score(score: int) -> str:
    if score <= 1:
        return "normal"
    if score == 2:
        return "mild"
    if score == 3:
        return "moderate"
    if score == 4:
        return "severe"
    return "critical"

# ---------------------------
# PER-KEY SEVERITY SCORING (INTERNAL)
# ---------------------------

def score_severity_for_abnormality_v6(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    if value is None:
        return 1

    try:
        v = float(value)
    except:
        return 1

    key_l = key.lower()
    s = 1

    if key_l == "hb":
        low_cut = 12.0 if sex.lower() == "female" else 13.0
        if age_group in ["neonate", "infant"]:
            low_cut = 14.0
        if v < low_cut - 4:
            s = 5
        elif v < low_cut - 2:
            s = 4
        elif v < low_cut:
            s = 3

    elif key_l == "wbc":
        if v > 30:
            s = 5
        elif v > 20:
            s = 4
        elif v > 12:
            s = 3

    elif key_l in ("neutrophils", "nlr"):
        if v > 12:
            s = 5
        elif v > 7:
            s = 4
        elif v > 3:
            s = 3

    elif key_l == "crp":
        if v > 250:
            s = 5
        elif v > 100:
            s = 4
        elif v > 50:
            s = 3
        elif v > 10:
            s = 2

    elif key_l == "creatinine":
        if v > 400:
            s = 5
        elif v > 200:
            s = 4
        elif v > 120:
            s = 3

    elif key_l == "platelets":
        if v < 10:
            s = 5
        elif v < 30:
            s = 4
        elif v < 100:
            s = 3
        elif v > 1000:
            s = 4

    elif key_l == "sodium":
        if v < 120 or v > 160:
            s = 5
        elif v < 125 or v > 155:
            s = 4
        elif v < 130 or v > 150:
            s = 3

    elif key_l == "potassium":
        if v < 2.8 or v > 6.5:
            s = 5
        elif v < 3.2 or v > 6.0:
            s = 4
        elif v < 3.5 or v > 5.5:
            s = 3

    return s

# ---------------------------
# FLAG COLOR LOGIC
# ---------------------------

def flag_for_key(key: str, value: Optional[float], sex: str = "unknown") -> Tuple[str, str]:
    if value is None:
        return ("normal", "#ffffff")

    try:
        v = float(value)
    except:
        return ("normal", "#ffffff")

    k = key.lower()

    if k == "hb":
        low = 12.0 if sex.lower() == "female" else 13.0
        high = 15.5 if sex.lower() == "female" else 17.5
        if v < low: return ("low", "#f59e0b")
        if v > high: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "wbc":
        if v < 4.0: return ("low", "#f59e0b")
        if v > 11.0: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "platelets":
        if v < 150: return ("low", "#f59e0b")
        if v > 450: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "creatinine":
        if v < 45: return ("low", "#f59e0b")
        if v > 120: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "crp":
        if v <= 10: return ("normal", "#ffffff")
        if v <= 50: return ("high", "#f59e0b")
        return ("high", "#b91c1c")

    if k == "sodium":
        if v < 135: return ("low", "#f59e0b")
        if v > 145: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "potassium":
        if v < 3.5: return ("low", "#f59e0b")
        if v > 5.1: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "mcv":
        if v < 80: return ("low", "#f59e0b")
        if v > 100: return ("high", "#b91c1c")
        return ("normal", "#ffffff")

    if k == "nlr":
        if v > 10: return ("high", "#b91c1c")
        if v > 5: return ("high", "#f59e0b")
        return ("normal", "#ffffff")

    if k in ("alt", "ast", "ck"):
        if v > 200: return ("high", "#b91c1c")
        if v > 100: return ("high", "#f59e0b")
        return ("normal", "#ffffff")

    return ("normal", "#ffffff")

# ---------------------------
# RISK BAR CALCULATOR
# ---------------------------

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
        if v <= 11: return int((v / 11) * 20)
        if v <= 20: return 30 + int(((v - 11) / 9) * 30)
        return min(100, 60 + int(((v - 20) / 30) * 40))

    if k in ("neutrophils", "nlr"):
        if v <= 3: return int((v / 3) * 10)
        if v <= 6: return 15 + int(((v - 3) / 3) * 25)
        if v <= 10: return 40 + int(((v - 6) / 4) * 30)
        return min(100, 70 + int(((v - 10) / 30) * 30))

    if k == "creatinine":
        if v <= 120: return int((v / 120) * 20)
        if v <= 200: return 25 + int(((v - 120) / 80) * 30)
        return min(100, 60 + int(((v - 200) / 300) * 40))

    if k == "hb":
        if v >= 12: return 5
        if v >= 10: return 20
        if v >= 8: return 50
        return min(100, 70 + int(((12 - v) / 12) * 30))

    if k == "platelets":
        if v >= 100 and v <= 450: return 5
        if v < 100: return min(100, 30 + int(((100 - v) / 100) * 70))
        if v > 450: return min(100, 20 + int(((v - 450) / 1000) * 80))

    return min(100, int(min(abs(v), 100)))
# ---------------------------
# ROUTE ENGINE V6
# ---------------------------

def generate_diagnostic_possibilities(canonical: Dict[str, Dict[str, Any]]) -> List[str]:
    Hb = canonical.get("Hb", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")
    Neut = canonical.get("Neutrophils", {}).get("value")
    CRP = canonical.get("CRP", {}).get("value")
    NLR = canonical.get("NLR", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")
    K = canonical.get("Potassium", {}).get("value")

    poss = []

    if (WBC and WBC > 12) or (Neut and Neut > 70) or (NLR and NLR > 10) or (CRP and CRP > 20):
        reasons = []
        if WBC and WBC > 12: reasons.append(f"WBC {WBC}")
        if Neut and Neut > 70: reasons.append(f"neutrophilia {Neut}%")
        if NLR and NLR > 10: reasons.append(f"NLR {NLR}")
        if CRP and CRP > 20: reasons.append(f"CRP {CRP}")
        poss.append("Sepsis / bacterial infection — " + "; ".join(reasons))

    if K is not None and (K < 3.2 or K > 6.0):
        poss.append(f"Electrolyte derangement — potassium {K} mmol/L")

    if Creat is not None and Creat > 120:
        poss.append(f"Acute kidney injury suspected — creatinine {Creat}")
    else:
        poss.append("Renal function normal — no evidence of AKI")

    if Hb is not None and Hb < 11:
        poss.append(f"Anemia — Hb {Hb}")
    else:
        poss.append("No anemia — Hb within expected range")

    if CRP is not None and CRP > 10:
        poss.append(f"Inflammatory response — CRP {CRP} mg/L")

    return poss


def route_engine_v6(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    age = patient_meta.get("age")
    sex = patient_meta.get("sex", "unknown")
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    per_key_scores = {}
    sepsis_flag = False

    def add_pattern(name, reason, score):
        patterns.append({"pattern": name, "reason": reason})
        per_key_scores[name] = max(per_key_scores.get(name, 1), score)

    hb = canonical.get("Hb", {}).get("value")
    mcv = canonical.get("MCV", {}).get("value")
    wbc = canonical.get("WBC", {}).get("value")
    crp = canonical.get("CRP", {}).get("value")
    neut = canonical.get("Neutrophils", {}).get("value")
    nlr = canonical.get("NLR", {}).get("value")
    plate = canonical.get("Platelets", {}).get("value")
    creat = canonical.get("Creatinine", {}).get("value")
    sodium = canonical.get("Sodium", {}).get("value")
    potassium = canonical.get("Potassium", {}).get("value")
    alt = canonical.get("ALT", {}).get("value")
    ast = canonical.get("AST", {}).get("value")
    ck = canonical.get("CK", {}).get("value")

    if hb is not None:
        s_hb = score_severity_for_abnormality_v6("Hb", hb, ag, sex)
        if s_hb > 1:
            add_pattern("anemia", f"Hb {hb}", s_hb)
            if mcv is not None and mcv < 80:
                add_pattern("microcytic anemia", f"MCV {mcv}", max(3, s_hb))
                routes.append("Iron deficiency route")
                ddx.extend(["Iron deficiency", "Thalassaemia trait", "Chronic blood loss"])
                next_steps.append("Order ferritin + reticulocyte count.")
            elif mcv is not None and mcv > 100:
                add_pattern("macrocytic anemia", f"MCV {mcv}", max(3, s_hb))
                routes.append("Macrocytic anemia route")
                ddx.extend(["B12 deficiency", "Folate deficiency"])
                next_steps.append("Order B12/folate.")
            else:
                add_pattern("normocytic anemia", "MCV normal", max(2, s_hb))
                routes.append("Normocytic route")
                ddx.extend(["Acute blood loss", "Hemolysis", "Anaemia of inflammation"])
                next_steps.append("Order retic + smear.")

    if wbc is not None and wbc > 11:
        s = score_severity_for_abnormality_v6("WBC", wbc, ag, sex)
        add_pattern("leukocytosis", f"WBC {wbc}", s)
        if neut is not None and neut >= 70:
            add_pattern("neutrophilic predominance", f"Neutrophils {neut}%", max(3, s))
            routes.append("Bacterial infection / Sepsis route")
            ddx.extend(["Bacterial infection", "Sepsis"])
            next_steps.append("Assess for sepsis; consider blood cultures + fluids.")
            sepsis_flag = True

    if crp is not None:
        s = score_severity_for_abnormality_v6("CRP", crp, ag, sex)
        if s > 1:
            add_pattern("elevated CRP", f"CRP {crp}", s)
            if crp > 50:
                routes.append("Inflammatory response route")
                ddx.extend(["Severe infection", "Inflammation"])
                next_steps.append("Urgent review; consider cultures.")
                if crp > 150:
                    sepsis_flag = True

    if nlr is not None:
        if nlr > 10:
            add_pattern("very high NLR", f"NLR {nlr}", 5)
            routes.append("High NLR / Sepsis route")
            next_steps.append("Urgent sepsis review.")
            sepsis_flag = True
        elif nlr > 5:
            add_pattern("high NLR", f"NLR {nlr}", 4)
            routes.append("High NLR route")
            next_steps.append("Search for infection source.")

    if plate is not None:
        s = score_severity_for_abnormality_v6("Platelets", plate, ag, sex)
        if plate < 150:
            add_pattern("thrombocytopenia", f"Platelets {plate}", s)
            ddx.extend(["ITP", "DIC", "Bone marrow failure"])
            next_steps.append("Repeat platelets; check smear.")
        elif plate > 450:
            add_pattern("thrombocytosis", f"Platelets {plate}", s)
            next_steps.append("Consider reactive thrombocytosis.")

    if creat is not None:
        s = score_severity_for_abnormality_v6("Creatinine", creat, ag, sex)
        if s >= 3:
            add_pattern("elevated creatinine", f"Creatinine {creat}", s)
            routes.append("AKI route")
            ddx.append("Acute kidney injury")
            next_steps.append("Repeat creatinine + check urine output.")

    if sodium is not None:
        s = score_severity_for_abnormality_v6("Sodium", sodium, ag, sex)
        if s >= 3:
            add_pattern("sodium derangement", f"Sodium {sodium}", s)
            next_steps.append("Correct sodium abnormalities.")

    if potassium is not None:
        s = score_severity_for_abnormality_v6("Potassium", potassium, ag, sex)
        if s >= 3:
            add_pattern("potassium derangement", f"Potassium {potassium}", s)
            next_steps.append("Correct potassium urgently; ECG monitoring.")

    if ck is not None and ck > 1000:
        add_pattern("rhabdomyolysis signal", f"CK {ck}", 4)
        routes.append("Rhabdomyolysis route")
        next_steps.append("Assess muscle pain; give fluids.")

    if (alt and alt > 200) or (ast and ast > 200):
        add_pattern("transaminitis", f"ALT {alt} AST {ast}", 3)
        routes.append("Hepatic route")
        next_steps.append("Check hepatitis panel + toxins.")

    if hb is not None and crp is not None and wbc is not None:
        if hb < 12 and crp > 20 and wbc > 11:
            add_pattern("anemia with inflammation", "Low Hb + high CRP + WBC", 4)
            routes.append("Infection + anemia route")
            next_steps.append("Treat infection; recheck Hb when CRP falls.")

    if ag == "teen" and sex.lower() == "female" and hb is not None and hb < 11:
        next_steps.append("Assess menstrual history; check ferritin.")

    ddx_rank = {}
    for i, d in enumerate(ddx):
        ddx_rank[d] = ddx_rank.get(d, 0) + (10 - i)

    if sepsis_flag:
        for d in ("Sepsis", "Bacterial infection", "Severe infection"):
            ddx_rank[d] = ddx_rank.get(d, 0) + 50

    ddx_sorted = sorted(ddx_rank.items(), key=lambda x: -x[1])
    ddx_list = [d for d, _ in ddx_sorted]

    severity_scores = list(per_key_scores.values()) or [1]
    combined = max(severity_scores) if severity_scores else 1
    severity_text = severity_text_from_score(combined)
    urgency = "high" if combined >= 4 else ("medium" if combined == 3 else "low")

    diagnostic_poss = generate_diagnostic_possibilities(canonical)

    summary_lines = []
    if diagnostic_poss:
        summary_lines.append("Diagnostic possibilities:\n• " + "\n• ".join(diagnostic_poss))
    if patterns:
        summary_lines.append("Patterns: " + "; ".join([p["pattern"] for p in patterns]))
    if routes:
        summary_lines.append("Routes: " + "; ".join(routes))
    if ddx_list:
        summary_lines.append("Top differentials: " + ", ".join(ddx_list))
    if next_steps:
        summary_lines.append("Immediate actions: " + " | ".join(next_steps))

    age_note = ""
    if ag == "elderly":
        age_note = "Elderly – broaden differential."

    color = "#b91c1c" if combined == 5 else "#ef4444" if combined == 4 else "#f59e0b" if combined == 3 else "#10b981"
    tw = "bg-red-700" if combined == 5 else "bg-red-500" if combined == 4 else "bg-yellow-400" if combined == 3 else "bg-green-500"

    risk_bars = {}
    for k in CANONICAL_KEYS:
        v = canonical.get(k, {}).get("value")
        pct = risk_percentage_for_key(k, v)
        if pct >= 80: c = "#b91c1c"
        elif pct >= 60: c = "#ef4444"
        elif pct >= 40: c = "#f59e0b"
        elif pct >= 20: c = "#facc15"
        else: c = "#10b981"
        risk_bars[k] = {"percentage": pct, "color": c}

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx_list,
        "severity_text": severity_text,
        "urgency_flag": urgency,
        "color": color,
        "tw_class": tw,
        "age_group": ag,
        "age_note": age_note,
        "diagnostic_possibilities": diagnostic_poss,
        "risk_bars": risk_bars,
        "summary": "\n".join(summary_lines) if summary_lines else "No significant abnormalities detected."
    }

# ---------------------------
# TREND ANALYSIS
# ---------------------------

def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {"trend": "no_previous"}

    diffs = {}
    for k, v in current.items():
        prev_val = previous.get(k, {}).get("value") if isinstance(previous, dict) else None
        cur_val = v.get("value")
        if prev_val is None or cur_val is None:
            continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {"previous": prev_val, "current": cur_val, "delta": delta, "pct_change": pct}
        except:
            pass

    return {"trend": diffs or "no_change"}

# ---------------------------
# SAVE TO SUPABASE
# ---------------------------

def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]):
    if not supabase:
        return
    try:
        supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": ai_results,
            "ai_error": None
        }).eq("id", report_id).execute()
    except Exception as e:
        logger.exception("Supabase save failed: %s", e)

# ---------------------------
# PROCESS REPORT
# ---------------------------

def process_report(record: Dict[str, Any]):
    report_id = record.get("id", "<unknown>")

    try:
        pdf_bytes = download_pdf_from_record(record)
    except Exception as e:
        if supabase:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", report_id).execute()
        return

    scanned = is_scanned_pdf(pdf_bytes)
    if scanned:
        if not HAS_PYTESSERACT:
            err = "Scanned PDF but pytesseract unavailable."
            if supabase:
                supabase.table(SUPABASE_TABLE).update({
                    "ai_status": "failed",
                    "ai_error": err
                }).eq("id", report_id).execute()
            return
        text = do_ocr_on_pdf(pdf_bytes)
    else:
        text = extract_text_with_pypdf(pdf_bytes)

    parsed = find_values_in_text(text)
    canonical = canonical_map(parsed)

    if not canonical:
        ai_results = {
            "error": "No analytes extracted",
            "parsed": parsed,
            "raw_text_excerpt": text[:5000],
            "scanned": scanned
        }
        save_ai_results_to_supabase(report_id, ai_results)
        return

    previous = None
    if supabase:
        try:
            pid = record.get("patient_id")
            if pid:
                q = supabase.table(SUPABASE_TABLE).select("ai_results").eq("patient_id", pid)\
                    .order("created_at", desc=True).limit(1).execute()
                rows = q.data if hasattr(q, "data") else q
                if rows:
                    previous = rows[0].get("ai_results")
        except:
            pass

    trends = trend_analysis(canonical, previous)
    routes = route_engine_v6(canonical, record, previous)

    decorated = {}
    sex = record.get("sex", "unknown")

    for k in CANONICAL_KEYS:
        val = canonical.get(k, {})
        value = val.get("value")
        unit = val.get("unit") if isinstance(val.get("unit"), str) else None
        s_num = score_severity_for_abnormality_v6(k, value, age_group_from_age(record.get("age")), sex)
        s_text = severity_text_from_score(s_num)
        flag, flag_color = flag_for_key(k, value, sex)
        pct = risk_percentage_for_key(k, value)

        if pct >= 80: rcol = "#b91c1c"
        elif pct >= 60: rcol = "#ef4444"
        elif pct >= 40: rcol = "#f59e0b"
        elif pct >= 20: rcol = "#facc15"
        else: rcol = "#10b981"

        decorated[k] = {
            "value": value,
            "unit": unit,
            "flag": flag,
            "color": flag_color,
            "severity_text": s_text,
            "risk_bar": {"percentage": pct, "color": rcol}
        }

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

    save_ai_results_to_supabase(report_id, ai_results)

# ---------------------------
# POLLING LOOP
# ---------------------------

def poll_and_process():
    if not supabase:
        logger.error("Supabase missing")
        return

    while True:
        try:
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending")\
                .limit(10).execute()
            rows = res.data if hasattr(res, "data") else res

            if rows:
                for r in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update({"ai_status": "processing"})\
                            .eq("id", r.get("id")).execute()
                        process_report(r)
                    except Exception as e:
                        supabase.table(SUPABASE_TABLE).update({
                            "ai_status": "failed",
                            "ai_error": str(e)
                        }).eq("id", r.get("id")).execute()
            else:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            logger.exception("Polling error: %s", e)
            time.sleep(POLL_INTERVAL)

# ---------------------------
# MAIN
# ---------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AMI Health Worker V6")
    parser.add_argument("--test-pdf", help="Path to a PDF for local testing")
    parser.add_argument("--once", action="store_true", help="Poll once then exit")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdfb = f.read()
        dummy = {
            "id": "local-test",
            "patient_id": "local-patient",
            "age": 30,
            "sex": "male",
            "file_path": args.test_pdf,
        }
        def dl_override(rec):
            return pdfb
        globals()["download_pdf_from_record"] = dl_override
        process_report(dummy)
        print("Test run complete.")
    else:
        if args.once:
            poll_and_process()
        else:
            poll_and_process()
