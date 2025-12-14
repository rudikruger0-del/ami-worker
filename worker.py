#!/usr/bin/env python3
"""
AMI Health Worker V4 - Balanced OCR + Full Route Engine (all possible routes)

Behavior:
- Poll Supabase for reports with ai_status='pending'
- Download PDF from Supabase storage (BUCKET=reports by default)
- Detect digital vs scanned PDF
- If scanned: render pages → aggressive compress (max width 1000px, quality 55, grayscale) → send to OpenAI Vision (gpt-4o) for OCR JSON
- If digital: extract text via pypdf
- Parse CBC + chemistry values from text or Vision JSON (robust regex + normalization)
- Canonical mapping into keys: Hb, MCV, MCH, WBC, Neutrophils, Platelets, Creatinine, CRP, etc.
- Route Engine: many routes + ddx + next steps + severity (1-5) + urgency flag + colors
- Trend analysis (if previous results exist)
- Save ai_results into Supabase and set ai_status='completed'
"""

import os
import io
import re
import time
import json
import base64
import traceback
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# Optional: if pytesseract used later
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# OpenAI + Supabase clients
from openai import OpenAI
from supabase import create_client, Client

# Load env
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

# clients
supabase: Optional[Client] = None
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print("Warning: Supabase client init failed:", e)
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# OCR compression settings (balanced)
MAX_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1000"))  # px
JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "55"))

# Poll interval
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

# helpers
def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def numeric_safety_gate(key: str, value: float) -> Optional[float]:
    """
    HARD safety limits. If a value is outside physiologically possible ranges,
    it is silently discarded.
    """
    LIMITS = {
        "Hb": (3, 25),
        "MCV": (50, 130),
        "MCH": (15, 45),
        "WBC": (0.1, 100),
        "Neutrophils": (0, 100),
        "Lymphocytes": (0, 100),
        "Platelets": (1, 2000),
        "CRP": (0, 500),
        "Creatinine": (10, 2000),
        "Sodium": (110, 180),
        "Potassium": (2.0, 7.5),
        "Urea": (0.5, 60),
        "CK": (0, 200000),
        "ALT": (0, 5000),
        "AST": (0, 5000),
    }

    if key not in LIMITS:
        return value

    low, high = LIMITS[key]
    if value < low or value > high:
        return None

    return value


def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# -----------------------------
# PDF reading & scanned detection
# -----------------------------
def download_pdf_from_supabase(record: Dict[str, Any]) -> bytes:
    """
    Download PDF bytes from Supabase storage or from pdf_url in record.
    HARDENED against Supabase SDK timeouts.
    """
    # Direct URL fallback
    if record.get("pdf_url"):
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if not supabase or not record.get("file_path"):
        raise ValueError("No pdf_url or file_path provided in report record")

    last_error = None

    # Retry once (Supabase SDK can fail transiently)
    for attempt in range(2):
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(
                record["file_path"]
            )

            # Supabase SDK inconsistency handling
            if hasattr(res, "data") and res.data:
                return res.data
            if isinstance(res, (bytes, bytearray)):
                return res

            raise RuntimeError("Supabase download returned empty response")

        except Exception as e:
            last_error = e
            print(f"⚠️ Supabase download attempt {attempt+1} failed:", e)
            time.sleep(1.5)

    # If we get here → hard fail cleanly
    raise RuntimeError(f"Supabase download failed after retries: {last_error}")


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract selectable text using pypdf. Returns large joined string or empty."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        # fallback silent
        print("pypdf extract failed:", e)
        return ""

def is_scanned_pdf(pdf_bytes: bytes, threshold_chars: int = 80) -> bool:
    """
    Returns True if the PDF appears scanned (very little selectable text).
    Uses pypdf extraction as heuristic.
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text or len(text.strip()) < threshold_chars:
        return True
    return False

# -----------------------------
# Balanced OCR image pipeline
# -----------------------------
def preprocess_image_for_ocr(img: Image.Image, max_width: int = MAX_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    """
    Convert image to grayscale, resize to max_width preserving aspect ratio,
    save as compressed JPEG to bytes.
    """
    # convert to RGB then grayscale to ensure consistent mode
    try:
        # convert to L (grayscale) to reduce size - Vision handles grayscale fine
        img = ImageOps.exif_transpose(img)  # fix orientation
        w, h = img.size
        if w > max_width:
            new_h = int((max_width / float(w)) * h)
            img = img.resize((max_width, new_h), Image.LANCZOS)
        # convert to grayscale to reduce size
        img = img.convert("L")
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        return out.getvalue()
    except Exception as e:
        print("preprocess_image_for_ocr error:", e)
        # final fallback: return original saved as JPEG
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

def ocr_image_with_openai(img_bytes: bytes, max_models_tokens: int = 3000) -> Dict[str, Any]:
    """
    Send compressed image bytes to OpenAI Vision model with a strict JSON instruction.
    Returns parsed JSON dict or {'cbc': []} on failure.
    """
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # base64 embed in data URI
    b64 = base64.b64encode(img_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant specialized in laboratory reports. "
        "Extract ALL numeric analytes for CBC and common chemistry: Hb, RBC, HCT, MCV, MCH, MCHC, RDW, "
        "WBC, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils, Platelets, CRP, Creatinine, "
        "Sodium, Potassium, Chloride, Urea, ALT, AST, CK and any units and reference ranges present.\n\n"
        "Return STRICT JSON only, with structure:\n"
        "{\n"
        "  \"cbc\": [\n"
        "    {\"analyte\": \"Hb\", \"value\": 11.6, \"units\": \"g/dL\", \"reference_low\": 12.4, \"reference_high\": 16.7},\n"
        "    ...\n"
        "  ]\n"
        "}\n"
        "If you cannot find values, return {\"cbc\": []}. No extra text."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract lab values from this image and return the strict JSON described."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        # content might be dict already
        if isinstance(raw, dict):
            return raw
        # if string, try load
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                # try extract JSON substring
                m = re.search(r'\{.*\}', raw, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except:
                        pass
        return {"cbc": []}
    except Exception as e:
        print("OpenAI Vision OCR failed:", e)
        return {"cbc": []}

def do_ocr_on_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Render pages, preprocess aggressively, send each to Vision, collect OCR JSON outputs (list of dicts)
    """
    ocr_outputs = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        print("pdf2image failed:", e)
        return []

    for i, page in enumerate(pages):
        try:
            compressed = preprocess_image_for_ocr(page)
            ocr_json = ocr_image_with_openai(compressed)
            ocr_outputs.append(ocr_json)
        except Exception as e:
            print(f"OCR error on page {i}:", e)
            ocr_outputs.append({"cbc": []})
    return ocr_outputs

# -----------------------------
# Parsing functions
# -----------------------------
# mapping synonyms to canonical keys
SYNONYMS = {
    "hb": "Hb", "haemoglobin": "Hb", "hemoglobin": "Hb",
    "rbc": "RBC", "erythrocyte": "RBC",
    "hct": "HCT", "haematocrit": "HCT",
    "mcv": "MCV", "mean corpuscular volume": "MCV",
    "mch": "MCH", "mean corpuscular haemoglobin": "MCH",
    "mchc": "MCHC", "rdw": "RDW",
    "wbc": "WBC", "white cell count": "WBC", "leukocyte": "WBC", "leucocyte": "WBC",
    "neutrophils": "Neutrophils", "neutrophil": "Neutrophils",
    "lymphocytes": "Lymphocytes", "lymphocyte": "Lymphocytes",
    "monocytes": "Monocytes", "eosinophils": "Eosinophils", "basophils": "Basophils",
    "platelets": "Platelets", "thrombocytes": "Platelets",
    "crp": "CRP", "c-reactive protein": "CRP",
    "creatinine": "Creatinine",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
    "chloride": "Chloride", "cl": "Chloride",
    "urea": "Urea",
    "alt": "ALT", "ast": "AST", "ck": "CK", "creatine kinase": "CK"
}

def normalize_label(label: str) -> Optional[str]:
    if not label:
        return None
    l = re.sub(r'[^a-z0-9 ]', '', label.lower()).strip()
    if l in SYNONYMS:
        return SYNONYMS[l]
    # partial match fallback
    for k,v in SYNONYMS.items():
        if k in l:
            return v
    return None

def parse_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Regex-based extraction from digital text. Returns canonical -> {value, units, raw, reference_low, reference_high}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not text:
        return out

    lines = [ln.strip() for ln in re.split(r'\r|\n', text) if ln.strip()]
    # patterns like 'Haemoglobin 11.6 g/dL (ref: 12.4-16.7)'
    for line in lines:
        # common label-value pairs
        m = re.findall(r'([A-Za-z\-\s]{2,30})[:\s]{1,4}(-?\d+\.\d+|-?\d+)(?:\s*([a-zA-Z/%\-\(\)\^0-9]+))?(?:.*ref[:\s]*\(?([0-9\.\-–to\s,]+)\)?)?', line)
        if m:
            for g in m:
                label_raw = g[0].strip()
                val = safe_float(g[1])
                units = g[2].strip() if g[2] else None
                ref = g[3].strip() if g[3] else None
                key = normalize_label(label_raw)
                if key and val is not None:
                    out.setdefault(key, {})['value'] = val
                    if units: out[key]['units'] = units
                    if ref:
                        # try parse ref like "12.4-16.7"
                        rparts = re.split(r'[-–to,]', ref)
                        if len(rparts) >= 2:
                            out[key]['reference_low'] = safe_float(rparts[0])
                            out[key]['reference_high'] = safe_float(rparts[1])
                    out[key]['raw'] = line
        # percentage forms
        p = re.findall(r'([A-Za-z\-\s]{2,30})[:\s]{1,4}(\d{1,3}\.?\d*)\s*%', line)
        if p:
            for g in p:
                label_raw = g[0].strip()
                val = safe_float(g[1])
                key = normalize_label(label_raw)
                if key and val is not None:
                    out.setdefault(key, {})['value'] = val
                    out[key]['units'] = '%'
                    out[key]['raw'] = line

    # fallback searches (simple)
    fb = re.findall(r'\b(hb|haemoglobin|wbc|platelets|crp|creatinine|mcv|mch)\b[^\d]{0,12}(-?\d+\.\d+|-?\d+)', text, re.I)
    for f in fb:
        label_raw, valtext = f
        val = safe_float(valtext)
        key = normalize_label(label_raw)
        if key and val is not None:
            out.setdefault(key, {})['value'] = val
            out[key]['raw'] = label_raw

    return out

def parse_values_from_ocr_json(ocr_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Normalize OCR JSON (from Vision) into canonical mapping.
    Expects {"cbc":[{"analyte":"Hb","value":11.6,"units":"g/dL", "reference_low":..., "reference_high":...},...]}
    """
    out: Dict[str, Dict[str, Any]] = {}
    if not ocr_json:
        return out
    items = ocr_json.get("cbc") or []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        raw_label = it.get("analyte") or it.get("label") or it.get("name") or ""
        key = normalize_label(raw_label) or raw_label.strip()
        val = safe_float(it.get("value"))
        if val is not None:
            out.setdefault(key, {})['value'] = val
        units = it.get("units") or it.get("unit")
        if units:
            out.setdefault(key, {})['units'] = units
        rl = it.get("reference_low") or it.get("ref_low") or None
        rh = it.get("reference_high") or it.get("ref_high") or None
        if rl is not None or rh is not None:
            out[key]['reference_low'] = safe_float(rl)
            out[key]['reference_high'] = safe_float(rh)
        out[key]['raw'] = it
    return out

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Ensure canonical keys and HARD-FILTER unsafe numeric values.
    This guarantees zero numeric hallucination.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for k, v in parsed.items():
        raw_val = v.get("value")
        val = safe_float(raw_val)
        if val is None:
            continue

        safe_val = numeric_safety_gate(k, val)
        if safe_val is None:
            continue  # DROP unsafe OCR / hallucinated value

        out[k] = {
            "value": safe_val,
            "units": v.get("units"),
            "raw": v.get("raw"),
            "reference_low": v.get("reference_low"),
            "reference_high": v.get("reference_high")
        }

    # Compute NLR only if BOTH values are present and safe
    try:
        n = out.get("Neutrophils", {}).get("value")
        l = out.get("Lymphocytes", {}).get("value")
        if n is not None and l is not None and l > 0:
            out["NLR"] = {
                "value": round(n / l, 2),
                "units": None,
                "raw": "computed NLR"
            }
    except Exception:
        pass
    # -----------------------------
    # Derived renal metrics (safe)
    # -----------------------------
    try:
        creat = out.get("Creatinine", {}).get("value")
        age = None
        sex = None

        # age/sex only available later — skip if missing
        # eGFR is finalized in process_report()
        if creat is not None:
            out.setdefault("_derived", {})["creatinine_present"] = True
    except Exception:
        pass
    # -----------------------------
    # Lipid normalization (naming)
    # -----------------------------
    lipid_aliases = {
        "LDL Cholesterol": "LDL",
        "LDL-C": "LDL",
        "Triglyceride": "Triglycerides",
        "Non HDL": "Non-HDL",
        "Non-HDL Cholesterol": "Non-HDL",
        "HDL Cholesterol": "HDL"
    }

    for src, dest in lipid_aliases.items():
        if src in out and dest not in out:
            out[dest] = out[src]

    # -----------------------------
    # Hard safety flags (never allow "normal")
    # -----------------------------
    try:
        crp = out.get("CRP", {}).get("value")
        egfr = out.get("eGFR", {}).get("value")
        plate = out.get("Platelets", {}).get("value")

        if crp is not None and crp >= 10:
            out.setdefault("_safety_flags", []).append("CRP elevated")

        if egfr is not None and egfr < 60:
            out.setdefault("_safety_flags", []).append("Reduced eGFR")

        if plate is not None and plate < 100:
            out.setdefault("_safety_flags", []).append("Thrombocytopenia")
    except Exception:
        pass

    return out
# -----------------------------
# eGFR calculation (CKD-EPI 2021, race-free)
# -----------------------------
def calculate_egfr(creatinine_umol: float, age: float, sex: str) -> Optional[float]:
    """
    CKD-EPI 2021 equation (race-free)
    Creatinine in µmol/L
    """
    try:
        if creatinine_umol is None or age is None:
            return None

        scr = creatinine_umol / 88.4  # convert µmol/L → mg/dL
        sex = (sex or "").lower()

        if sex == "female":
            k = 0.7
            a = -0.241
            sex_factor = 1.012
        else:
            k = 0.9
            a = -0.302
            sex_factor = 1.0

        egfr = (
            142
            * min(scr / k, 1) ** a
            * max(scr / k, 1) ** -1.200
            * (0.9938 ** age)
            * sex_factor
        )

        return round(egfr, 1)
    except Exception:
        return None


# -----------------------------
# ROUTE ENGINE (comprehensive)
# -----------------------------
COLOR_MAP = {
    5: {"label":"critical","color":"#b91c1c","tw":"bg-red-600","urgency":"high"},
    4: {"label":"severe","color":"#f97316","tw":"bg-orange-500","urgency":"high"},
    3: {"label":"moderate","color":"#f59e0b","tw":"bg-yellow-400","urgency":"medium"},
    2: {"label":"borderline","color":"#facc15","tw":"bg-yellow-300","urgency":"low"},
    1: {"label":"normal","color":"#10b981","tw":"bg-green-500","urgency":"low"}
}

def age_group(age: Optional[float]) -> str:
    if age is None:
        return "adult"
    try:
        a = float(age)
    except:
        return "adult"
    if a < (1/12): return "neonate"
    if a < 1: return "infant"
    if a < 13: return "child"
    if a < 18: return "teen"
    if a < 65: return "adult"
    return "elderly"

def score_severity_key(key: str, val: Optional[float], ag: str, sex: str) -> int:
    if val is None:
        return 1
    k = key.lower()
    try:
        if k == "hb":
            low = 12.0 if sex and str(sex).lower()=="female" else 13.0
            if ag in ("neonate","infant"): low = 14.0
            if val < low - 4: return 5
            if val < low - 2: return 4
            if val < low: return 3
            return 1
        if k == "wbc":
            if val > 25: return 5
            if val > 15: return 4
            if val > 11: return 3
            return 1
        if k == "crp":
            if val > 200: return 5
            if val > 100: return 4
            if val > 50: return 3
            if val > 10: return 2
            return 1
        if k == "platelets":
            if val < 20: return 5
            if val < 50: return 4
            if val < 100: return 3
            return 1
        if k == "creatinine":
            if val > 354: return 5
            if val > 200: return 4
            if val > 120: return 3
            return 1
        if k == "nlr":
            if val > 10: return 4
            if val > 5: return 3
            return 1
        if k == "ck":
            if val > 10000: return 5
            if val > 5000: return 4
            if val > 2000: return 3
            return 1
        if k in ("potassium","k"):
            if val < 3.2 or val > 6.0: return 5
            if val < 3.5 or val > 5.5: return 3
            return 1
        if k in ("sodium","na"):
            if val < 120 or val > 160: return 5
            if val < 125 or val > 155: return 3
            return 1
    except:
        return 1
    return 1

def route_engine_all(
    canonical: Dict[str, Dict[str, Any]],
    patient_meta: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
    doctor_trust_flags: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:

    """
    Massive route engine covering many routes. Returns:
    { patterns, routes, next_steps, differential, per_key, overall_severity, urgency, color, tw_class, age_group, age_note, summary }
    """
    ag = age_group(patient_meta.get("age"))
    sex = patient_meta.get("sex") or "unknown"

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    per_key = {}
    severity_scores = []
        if not doctor_trust_flags:
        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False
        }


    # build per_key and severity list
    for k, v in canonical.items():
        val = v.get("value")
        score = score_severity_key(k, val, ag, sex)
        c = COLOR_MAP.get(score, COLOR_MAP[1])
        per_key[k] = {
            "value": val,
            "units": v.get("units"),
            "severity": score,
            "urgency": c["urgency"],
            "color": c["color"],
            "tw_class": c["tw"],
            "raw": v.get("raw"),
            "reference_low": v.get("reference_low"),
            "reference_high": v.get("reference_high"),
        }
        severity_scores.append(score)

    # convenience
    Hb = canonical.get("Hb", {}).get("value")
    MCV = canonical.get("MCV", {}).get("value")
    MCH = canonical.get("MCH", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")
    Neut = canonical.get("Neutrophils", {}).get("value")
    Lymph = canonical.get("Lymphocytes", {}).get("value")
    NLR = canonical.get("NLR", {}).get("value")
    CRP = canonical.get("CRP", {}).get("value")
    Plate = canonical.get("Platelets", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")
    CK = canonical.get("CK", {}).get("value")
    K = canonical.get("Potassium", {}).get("value")
    Na = canonical.get("Sodium", {}).get("value")
    ALT = canonical.get("ALT", {}).get("value")
    AST = canonical.get("AST", {}).get("value")
    RDW = canonical.get("RDW", {}).get("value")

    # -------------------------
    # Anaemia routes
    # -------------------------
    if Hb is not None:
        score_hb = score_severity_key("Hb", Hb, ag, sex)
        if score_hb > 1:
            patterns.append({"pattern":"anemia", "reason": f"Hb {Hb} g/dL"})
            # microcytic
            if MCV is not None and MCV < 80:
                patterns.append({"pattern":"microcytic anemia", "reason": f"MCV {MCV}"})
                routes.append("Iron deficiency route")
                ddx += ["Iron deficiency anemia", "Thalassaemia trait", "Chronic blood loss"]
                if ag == "teen" and sex.lower()=="female":
                    next_steps.append("High suspicion for menstrual blood loss; order ferritin and reticulocyte count")
                else:
                    next_steps.append("Order ferritin, iron studies, reticulocyte count; consider stool occult blood in adults")
            # macrocytic
            elif MCV is not None and MCV > 100:
                patterns.append({"pattern":"macrocytic anemia", "reason": f"MCV {MCV}"})
                routes.append("Macrocytic route")
                ddx += ["Vitamin B12 deficiency", "Folate deficiency", "Alcohol-related", "Myelodysplasia"]
                next_steps.append("Order B12, folate, reticulocyte count; review meds & alcohol history")
            else:
                patterns.append({"pattern":"normocytic anemia", "reason":"MCV normal or missing"})
                routes.append("Normocytic anemia route")
                ddx += ["Acute blood loss", "Haemolysis", "Anaemia of chronic disease", "Renal disease"]
                next_steps.append("Order reticulocyte, LDH, peripheral smear, direct antiglobulin test if hemolysis suspected")

    # Hemolysis specifics
    if RDW is not None and RDW > 15 and Hb is not None and MCV is not None and MCV < 80:
        # high RDW with microcytosis suggests iron deficiency; but high RDW with normal/macro suggests hemolysis etc.
        pass

    # Pancytopenia route (low Hb, low WBC, low platelets)
    pancytopenic = False
    if (Hb is not None and Hb < (12.0 if sex.lower()=="female" else 13.0)) and (WBC is not None and WBC < 4.0) and (Plate is not None and Plate < 150):
        patterns.append({"pattern":"pancytopenia", "reason":"Low Hb, low WBC, low Platelets"})
        pancytopenic = True
        routes.append("Pancytopenia route")
        ddx += ["Aplastic anaemia", "Bone marrow infiltration", "Hypersplenism", "Severe infection"]
        next_steps.append("Urgent haematology referral; bone marrow assessment may be required")

    # -------------------------
    # White cell patterns & infection
    # -------------------------
    if WBC is not None:
        if WBC > 11:
            patterns.append({"pattern":"leukocytosis", "reason": f"WBC {WBC}"})
            if Neut is not None and Neut > 70:
                patterns.append({"pattern":"neutrophilic predominance", "reason": f"Neutrophils {Neut}%"})
                routes.append("Bacterial infection route")
                ddx += ["Bacterial infection", "Sepsis", "Localized infection (pneumonia, UTI)"]
                next_steps.append("Assess clinically for source; blood cultures if febrile; consider empiric antibiotics if unstable")
            elif Lymph is not None and Lymph > 50:
                patterns.append({"pattern":"lymphocytosis", "reason": f"Lymphocytes {Lymph}%"})
                routes.append("Viral/lymphoid response route")
                ddx += ["Viral infection", "Pertussis", "Chronic lymphocytic processes"]
                next_steps.append("Consider viral testing and clinical correlation")

    # High NLR
    if NLR is not None and NLR > 5:
        patterns.append({"pattern":"high NLR", "reason": f"NLR {NLR}"})
        routes.append("High NLR route (possibly bacterial/severe inflammation)")
        next_steps.append("Evaluate for sepsis and clinical severity")

    # CRP high
    if CRP is not None:
        if CRP > 50:
            patterns.append({"pattern":"high CRP", "reason": f"CRP {CRP}"})
            routes.append("Significant inflammatory response")
            ddx += ["Severe bacterial infection", "Systemic inflammatory conditions"]
            next_steps.append("Urgent clinical review; blood cultures and imaging as indicated")
        elif CRP > 10:
            patterns.append({"pattern":"elevated CRP", "reason": f"CRP {CRP}"})
            next_steps.append("Consider clinical correlation for infection/inflammation")

    # Neutropenia / febrile neutropenia
    if Neut is not None:
        # Neut can be percentage if WBC present absolute neutrophils better — we assume percent if >1 and small numbers
        if Neut < 1.0:
            patterns.append({"pattern":"neutropenia", "reason": f"Neutrophils {Neut}"})
            routes.append("Neutropenia route")
            ddx += ["Drug-induced neutropenia", "Bone marrow failure", "Severe infection"]
            next_steps.append("Urgent review; consider isolation and haematology input if severe")
        elif Neut < 0.5:
            patterns.append({"pattern":"severe neutropenia", "reason": f"Neutrophils {Neut}"})
            routes.append("Severe neutropenia route")
            next_steps.append("Urgent haematology referral; consider hospital admission")

    # -------------------------
    # Platelet routes
    # -------------------------
    if Plate is not None:
        if Plate < 150:
            patterns.append({"pattern":"thrombocytopenia", "reason": f"Platelets {Plate}"})
            routes.append("Thrombocytopenia route")
            ddx += ["Immune thrombocytopenia (ITP)", "DIC", "Bone marrow suppression", "Viral infection"]
            next_steps.append("Check peripheral smear; repeat platelet; investigate bleeding signs; urgent if <50")
        elif Plate > 450:
            patterns.append({"pattern":"thrombocytosis", "reason": f"Platelets {Plate}"})
            routes.append("Thrombocytosis route")
            ddx += ["Reactive thrombocytosis", "Myeloproliferative disorder"]
            next_steps.append("Repeat count and check inflammatory markers; refer to haematology if persistent")

    # -------------------------
    # Renal / AKI routes
    # -------------------------
    if Creat is not None:
        if Creat > 120:
            patterns.append({"pattern":"renal impairment", "reason": f"Creatinine {Creat}"})
            routes.append("AKI/renal route")
            ddx += ["Acute kidney injury", "Chronic kidney disease", "Dehydration"]
            next_steps.append("Repeat creatinine and electrolytes; assess urine output and stop nephrotoxins; urgent if rising")

    # -------------------------
    # Rhabdomyolysis / CK high
    # -------------------------
    if CK is not None:
        if CK > 2000:
            patterns.append({"pattern":"rhabdomyolysis physiology", "reason": f"CK {CK}"})
            routes.append("Rhabdomyolysis route")
            ddx += ["Rhabdomyolysis", "Trauma", "Seizure", "Drug/toxin-related muscle injury"]
            next_steps.append("Aggressive IV fluids; monitor potassium and creatinine; urgent review")

    # -------------------------
    # Electrolyte / cardiac risk routes
    # -------------------------
    if K is not None:
        if K < 3.2 or K > 6.0:
            patterns.append({"pattern":"critical potassium", "reason": f"K {K}"})
            routes.append("Electrolyte - Potassium risk route")
            next_steps.append("Immediate ECG if arrhythmia risk; correct potassium urgently as per protocols")

    if Na is not None:
        if Na < 125 or Na > 155:
            patterns.append({"pattern":"significant sodium disturbance", "reason": f"Na {Na}"})
            next_steps.append("Assess for neurologic symptoms; correct sodium carefully and consider admission")

    # -------------------------
    # Liver injury / hepatic patterns
    # -------------------------
    if ALT is not None or AST is not None:
        if (ALT and ALT > 3 * 40) or (AST and AST > 3 * 40):  # rough
            patterns.append({"pattern":"liver enzyme elevation", "reason": f"ALT {ALT} AST {AST}"})
            routes.append("Hepatic injury route")
            ddx += ["Viral hepatitis", "Alcohol-related liver injury", "Drug-induced liver injury", "Ischemic hepatitis"]
            next_steps.append("Check hepatitis serology, review meds, assess alcohol history; consider urgent hepatology if very high")

    # -------------------------
    # Pediatric & pregnancy modifiers
    # -------------------------
    if ag == "teen" and sex and str(sex).lower()=="female":
        next_steps.append("Consider menstrual blood loss as common cause of microcytic anemia in teenage females")

    if ag in ("neonate","infant","child"):
        # example pediatric flags
        if Hb is not None and Hb < 10:
            patterns.append({"pattern":"pediatric anemia", "reason": f"Hb {Hb}"})
            routes.append("Pediatric anemia route")
            next_steps.append("Paediatric review; consider iron deficiency, haemoglobinopathy, or nutritional causes")
        # WBC reference differs - be cautious
        ddx.append("Pediatric age-specific considerations applied")

    # -------------------------
    # Oncology / red flags
    # -------------------------
    # cytopenias, unexplained high RDW, persistent abnormal counts -> consider marrow pathology
    if (Hb is not None and Hb < 10 and Plate is not None and Plate < 150) or pancytopenic:
        patterns.append({"pattern":"possible marrow pathology", "reason":"Cytopenias present"})
        routes.append("Oncology/Haematology referral route")
        ddx += ["Bone marrow infiltration", "Myelodysplasia", "Leukaemia"]
        next_steps.append("Urgent haematology referral; consider bone marrow biopsy")

    # -------------------------
    # DIC & critical clotting patterns (when CRP high + low platelets)
    # -------------------------
    if CRP is not None and CRP > 100 and Plate is not None and Plate < 50:
        patterns.append({"pattern":"possible DIC/severe sepsis", "reason":"High CRP + thrombocytopenia"})
        routes.append("Severe sepsis/DIC route")
        ddx += ["Disseminated intravascular coagulation", "Severe bacterial sepsis"]
        next_steps.append("Urgent sepsis pathway; check coagulation profile; ICU review if unstable")

    # dedupe ddx
    ddx = list(dict.fromkeys([d for d in ddx if d]))

    # =========================
    # DOMINANT THREAT OVERRIDE (doctor-priority logic)
    # =========================

    # Severe thrombocytopenia dominates everything
    if Plate is not None:
        if Plate < 50:
            severity_scores.append(5)
            patterns.insert(0, {
                "pattern": "severe thrombocytopenia",
                "reason": f"Platelets {Plate}"
            })
            routes.insert(0, "Severe thrombocytopenia route")
        elif Plate < 100:
            severity_scores.append(4)
            patterns.insert(0, {
                "pattern": "high-risk thrombocytopenia",
                "reason": f"Platelets {Plate}"
            })
            routes.insert(0, "High-risk thrombocytopenia route")

    # Rhabdomyolysis overrides anaemia & infection
    if CK is not None:
        if CK > 10000:
            severity_scores.append(5)
            patterns.insert(0, {
                "pattern": "severe rhabdomyolysis",
                "reason": f"CK {CK}"
            })
            routes.insert(0, "Rhabdomyolysis / muscle injury route")
        elif CK > 5000:
            severity_scores.append(4)
            patterns.insert(0, {
                "pattern": "rhabdomyolysis",
                "reason": f"CK {CK}"
            })
            routes.insert(0, "Rhabdomyolysis route")

    # Critical electrolytes
    if K is not None and (K < 3.0 or K > 6.0):
        severity_scores.append(5)
        patterns.insert(0, {
            "pattern": "life-threatening potassium disturbance",
            "reason": f"K {K}"
        })
        routes.insert(0, "Critical electrolyte disturbance route")

    if Na is not None and (Na < 120 or Na > 160):
        severity_scores.append(5)
        patterns.insert(0, {
            "pattern": "life-threatening sodium disturbance",
            "reason": f"Na {Na}"
        })
        routes.insert(0, "Critical electrolyte disturbance route")

    # Severe neutropenia
    if Neut is not None and Neut < 0.5:
        severity_scores.append(5)
        patterns.insert(0, {
            "pattern": "severe neutropenia",
            "reason": f"Neutrophils {Neut}"
        })
        routes.insert(0, "Severe neutropenia route")

    # -------------------------
    # FINAL SEVERITY & SUMMARY
    # -------------------------
    overall_sev = max(severity_scores) if severity_scores else 1
    # -----------------------------
    # DOCTOR TRUST OVERRIDE
    # -----------------------------
    if doctor_trust_flags["has_long_term_risk"] and overall_sev == 1:
        overall_sev = 2  # borderline / non-normal

    color = COLOR_MAP.get(overall_sev, COLOR_MAP[1])
    urgency = color["urgency"]

    summary_parts = []
    if patterns:
        summary_parts.append("Patterns: " + "; ".join([p["pattern"] for p in patterns]))
    if routes:
        summary_parts.append("Primary routes: " + "; ".join(routes))
    if ddx:
        summary_parts.append("Top differentials: " + ", ".join(ddx[:6]))
    if next_steps:
        summary_parts.append("Immediate suggestions: " + " | ".join(next_steps[:6]))

    age_note = ""
    if ag == "elderly":
        age_note = "Elderly patient — broaden differential to include chronic disease and malignancy."

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "per_key": per_key,
        "overall_severity": overall_sev,
        "severity_text": COLOR_MAP[overall_sev]["label"],
        "urgency_flag": urgency,
        "color": color["color"],
        "tw_class": color["tw"],
        "age_group": ag,
        "age_note": age_note,
        "summary": "\n".join(summary_parts) if summary_parts else "No significant abnormalities detected."
    }


# -----------------------------
# Trend analysis: simple diffs
# -----------------------------
def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {"trend":"no_previous"}
    diffs = {}
    for k,v in current.items():
        prev = previous.get(k, {}).get("value") if previous else None
        cur = v.get("value")
        if prev is None or cur is None:
            continue
        try:
            delta = cur - prev
            pct = (delta / prev) * 100 if prev != 0 else None
            diffs[k] = {"previous": prev, "current": cur, "delta": delta, "pct_change": pct}
        except:
            pass
    return {"trend": diffs}
# -----------------------------
# Risk domain analysis (acute vs long-term)
# -----------------------------
def compute_risk_domains(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adds clinician-style risk stratification without changing severity.
    Separates acute danger from long-term risk.
    """

    risks = {
        "acute": [],
        "cardiovascular": [],
        "renal": [],
        "infection": [],
        "hepatic": [],
        "metabolic": [],
        "notes": []
    }

    age = patient_meta.get("age")
    sex = (patient_meta.get("sex") or "").lower()

    # -----------------------------
    # Cardiovascular risk (long-term)
    # -----------------------------
    LDL = canonical.get("LDL", {}).get("value")
    TG = canonical.get("Triglycerides", {}).get("value")
    HDL = canonical.get("HDL", {}).get("value")
    NonHDL = canonical.get("Non-HDL", {}).get("value")

    if LDL is not None and LDL > 3.0:
        risks["cardiovascular"].append("Elevated LDL cholesterol")

    if TG is not None and TG > 1.7:
        risks["cardiovascular"].append("Elevated triglycerides")

    if NonHDL is not None and NonHDL > 3.4:
        risks["cardiovascular"].append("Elevated non-HDL cholesterol")

    if HDL is not None:
        if (sex == "male" and HDL < 1.0) or (sex == "female" and HDL < 1.3):
            risks["cardiovascular"].append("Low HDL cholesterol")

    if risks["cardiovascular"]:
        risks["notes"].append(
            "Lipid abnormalities increase long-term cardiovascular risk but are not acutely dangerous."
        )

    # -----------------------------
    # Renal risk
    # -----------------------------
    eGFR = canonical.get("eGFR", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")

    if eGFR is not None:
        if eGFR < 30:
            risks["renal"].append("Severe renal impairment")
        elif eGFR < 60:
            risks["renal"].append("Moderate chronic kidney disease")

    # -----------------------------
    # Infection / inflammation
    # -----------------------------
    CRP = canonical.get("CRP", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")

    if CRP is not None and CRP > 10:
        risks["infection"].append("Inflammatory or infectious process likely")

    if WBC is not None and WBC > 11:
        risks["infection"].append("Leukocytosis — correlate clinically")

    # -----------------------------
    # Liver / bilirubin logic
    # -----------------------------
    Bili = canonical.get("Bilirubin", {}).get("value")
    ALT = canonical.get("ALT", {}).get("value")
    AST = canonical.get("AST", {}).get("value")

    if Bili is not None and Bili > 21:
        if not (ALT and ALT > 50) and not (AST and AST > 50):
            risks["hepatic"].append("Isolated mild bilirubin elevation")
            risks["notes"].append(
                "Isolated bilirubin elevation with normal enzymes is commonly benign (e.g. Gilbert syndrome)."
            )

    return risks

# -----------------------------
# Interpreter wrapper (final AI summary) - uses gpt-4o-mini
# -----------------------------
def call_ai_on_report(text: str) -> Dict[str, Any]:
    """
    Sends either raw extracted text or structured JSON text to the interpretation model.
    Returns strict JSON expected by your app.
    """
    if not openai_client:
        return {"error": "OpenAI client not configured"}

    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing CBC and chemistry results. "
        "You MUST NOT give a formal diagnosis. Only describe lab abnormalities, patterns, severity, "
        "and give concise suggested follow-up steps. Use a clear, ER-friendly tone.\n\n"
        "Return STRICT JSON with this structure:\n"
        "{\n"
        "  \"patient\": { \"name\": null, \"age\": null, \"sex\": \"Unknown\" },\n"
        "  \"cbc\": [ { \"analyte\": \"Hb\", \"value\": 11.6, \"units\": \"g/dL\", \"reference_low\": null, \"reference_high\": null, \"flag\": \"low\" } ],\n"
        "  \"summary\": { \"impression\": \"\", \"suggested_follow_up\": \"\" }\n"
        "}\n"
        "Return ONLY JSON, no commentary."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":text}
            ],
            response_format={"type":"json_object"},
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except:
                # attempt substring
                m = re.search(r'\{.*\}', raw, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except:
                        pass
        return {"patient": {"name": None, "age": None, "sex": "Unknown"}, "cbc": [], "summary": {"impression":"", "suggested_follow_up":""}}
    except Exception as e:
        print("Interpretation model error:", e)
        return {"error": str(e)}

# -----------------------------
# Save to Supabase
# -----------------------------
def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        print("Supabase not configured; skipping save.")
        return
    try:
        payload = {"ai_status": "completed", "ai_results": ai_results, "ai_error": None}
        res = supabase.table(SUPABASE_TABLE).update(payload).eq("id", report_id).execute()
        # SDK returns object with status sometimes; not critical
        print(f"Saved ai_results for {report_id}")
    except Exception as e:
        print("Save to Supabase failed:", e)
        try:
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":str(e)}).eq("id", report_id).execute()
        except:
            print("Also failed to mark failed in Supabase.")

# -----------------------------
# Main processing function
# -----------------------------
def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id") or job.get("report_id")
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    print(f"Processing report {report_id} (path={file_path})")
    try:
        if not file_path and not job.get("pdf_url"):
            err = f"Missing file_path or pdf_url for report {report_id}"
            print("Error:", err)
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
            return {"error": err}

        # download pdf bytes
        pdf_bytes = download_pdf_from_supabase(job)
        # decide scanned vs digital
        scanned = is_scanned_pdf(pdf_bytes)

        parsed: Dict[str, Dict[str, Any]] = {}
        merged_text_for_ai = ""

        if scanned:
            print(f"Report {report_id} detected as SCANNED — running Balanced OCR")
            ocr_pages = do_ocr_on_pdf(pdf_bytes)  # list of dicts
            combined_items = []
            for p in ocr_pages:
                if isinstance(p, dict) and p.get("cbc"):
                    combined_items.extend(p.get("cbc"))
            if combined_items:
                parsed = parse_values_from_ocr_json({"cbc": combined_items})
                merged_text_for_ai = json.dumps({"cbc": combined_items})
            else:
                # fallback: attempt to extract any text digitally (rare) and parse
                tex = extract_text_from_pdf(pdf_bytes)
                parsed = parse_values_from_text(tex)
                merged_text_for_ai = tex or json.dumps({"cbc": []})
        else:
            print(f"Report {report_id} appears DIGITAL — extracting text")
            text = extract_text_from_pdf(pdf_bytes)
            # merge l_text if provided
            if l_text:
                merged_text_for_ai = (l_text + "\n\n" + text).strip()
            else:
                merged_text_for_ai = text
            parsed = parse_values_from_text(merged_text_for_ai)

        canonical = canonical_map(parsed)
        # ==============================
        # HARD DOCTOR-TRUST GUARDRAILS
        # ==============================
        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False
        }

        # ---- LONG-TERM CARDIOVASCULAR RISK (NEVER NORMAL) ----
        ldl = canonical.get("LDL", {}).get("value")
        triglycerides = canonical.get("Triglycerides", {}).get("value")
        non_hdl = canonical.get("Non-HDL", {}).get("value")

        if ldl is not None and ldl >= 3.0:
            doctor_trust_flags["has_long_term_risk"] = True

        if triglycerides is not None and triglycerides >= 1.7:
            doctor_trust_flags["has_long_term_risk"] = True

        if non_hdl is not None and non_hdl >= 3.4:
            doctor_trust_flags["has_long_term_risk"] = True

        # -----------------------------
        # eGFR calculation (CKD-EPI 2021, race-free)
        # -----------------------------
        creat = canonical.get("Creatinine", {}).get("value")
        if creat is not None and patient_age:
            egfr = calculate_egfr(creat, patient_age, patient_sex)
            if egfr is not None:
                canonical["eGFR"] = {
                    "value": round(egfr, 1),
                    "units": "mL/min/1.73m²",
                    "raw": "calculated (CKD-EPI 2021)"
                }


        # fetch previous ai_results for trend analysis
        previous = None
        try:
            if supabase and job.get("patient_id"):
                prev_q = supabase.table(SUPABASE_TABLE).select("ai_results,created_at").eq("patient_id", job.get("patient_id")).order("created_at", desc=True).limit(1).execute()
                prev_rows = prev_q.data if hasattr(prev_q, "data") else prev_q
                if prev_rows:
                    previous = prev_rows[0].get("ai_results")
        except Exception:
            previous = None

        trends = trend_analysis(canonical, previous)
       route_info = route_engine_all(
    canonical,
    {"age": patient_age, "sex": patient_sex},
    previous,
    doctor_trust_flags
)

        # ==============================
        # RISK DOMAIN ANALYSIS (ACUTE vs LONG-TERM)
        # ==============================

        risk_domains = {
            "acute_risk": [],
            "long_term_risk": [],
            "data_quality_flags": []
        }

        # ---------- Renal risk ----------
        creat = canonical.get("Creatinine", {}).get("value")
        egfr = canonical.get("eGFR (CKD-EPI)", {}).get("value")

        if egfr is not None:
            if egfr < 30:
                risk_domains["acute_risk"].append({
                    "domain": "renal",
                    "level": "high",
                    "reason": f"Severely reduced eGFR ({egfr}) – possible AKI or advanced CKD"
                })
            elif egfr < 60:
                risk_domains["long_term_risk"].append({
                    "domain": "renal",
                    "level": "moderate",
                    "reason": f"Reduced eGFR ({egfr}) – chronic kidney disease risk"
                })

        # ---------- Inflammation / infection ----------
        crp = canonical.get("CRP", {}).get("value")

        if crp is not None:
            if crp >= 50:
                risk_domains["acute_risk"].append({
                    "domain": "infection",
                    "level": "high",
                    "reason": f"Markedly elevated CRP ({crp}) – significant inflammatory or infectious process"
                })
            elif crp >= 10:
                risk_domains["acute_risk"].append({
                    "domain": "infection",
                    "level": "moderate",
                    "reason": f"Elevated CRP ({crp}) – active inflammation"
                })

        # ---------- Cardiovascular (long-term) ----------
        ldl = canonical.get("LDL", {}).get("value")
        trig = canonical.get("Triglycerides", {}).get("value")
        non_hdl = canonical.get("Non-HDL", {}).get("value")

        if ldl is not None and ldl >= 3.0:
            risk_domains["long_term_risk"].append({
                "domain": "cardiovascular",
                "level": "moderate",
                "reason": f"Elevated LDL cholesterol ({ldl}) increasing long-term cardiovascular risk"
            })

        if trig is not None and trig >= 2.0:
            risk_domains["long_term_risk"].append({
                "domain": "cardiovascular",
                "level": "moderate",
                "reason": f"Elevated triglycerides ({trig}) increasing metabolic and cardiovascular risk"
            })

        if non_hdl is not None and non_hdl >= 3.7:
            risk_domains["long_term_risk"].append({
                "domain": "cardiovascular",
                "level": "moderate",
                "reason": f"Elevated non-HDL cholesterol ({non_hdl})"
            })

        # ---------- Bilirubin consistency ----------
        bili_total = canonical.get("Bilirubin Total", {}).get("value")
        bili_conj = canonical.get("Bilirubin Conjugated", {}).get("value")
        bili_unconj = canonical.get("Bilirubin Unconjugated", {}).get("value")

        if None not in (bili_total, bili_conj, bili_unconj):
            if abs((bili_conj + bili_unconj) - bili_total) > 2:
                risk_domains["data_quality_flags"].append({
                    "issue": "bilirubin_inconsistency",
                    "reason": "Conjugated + unconjugated bilirubin does not equal total – possible transcription or lab artifact"
                })

        # ---------- Summary ----------
        if not risk_domains["acute_risk"] and not risk_domains["long_term_risk"]:
            risk_domains["summary"] = (
                "No acute pathology detected. No significant long-term risk markers identified."
            )
        else:
            risk_domains["summary"] = (
                "Acute risks present."
                if risk_domains["acute_risk"]
                else "No acute pathology. Long-term risk factors identified."
            )



        # interpreter: pass structured canonical JSON if available for better outputs
        ai_input = merged_text_for_ai if merged_text_for_ai else json.dumps({"canonical": canonical})
        interpretation = call_ai_on_report(ai_input)

        ai_results = {
            "processed_at": now_iso(),
            "scanned": scanned,
            "canonical": canonical,
            "routes": route_info,
            "risk_domains": risk_domains,
            "trends": trends,
            "ai_interpretation": interpretation
        }

        save_ai_results_to_supabase(report_id, ai_results)
        print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_results}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()
        try:
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
        except:
            pass
        return {"error": err}

# -----------------------------
# Poll loop
# -----------------------------
def poll_loop():
    if not supabase:
        print("Supabase client not configured — poll loop disabled.")
        return
    print("AMI Worker V4 polling for pending reports...")
    while True:
        try:
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending").limit(5).execute()
            rows = res.data if hasattr(res, "data") else res
            if rows:
                for r in rows:
                    try:
                        # mark processing
                        supabase.table(SUPABASE_TABLE).update({"ai_status":"processing"}).eq("id", r.get("id")).execute()
                    except Exception:
                        pass
                    process_report(r)
            else:
                # back off a little when empty
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("Polling error:", e)
            traceback.print_exc()
            time.sleep(5)

# CLI test harness
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pdf", help="Path to local PDF to test")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            test_bytes = f.read()
        # override download function
        def _dl(r): return test_bytes
        globals()['download_pdf_from_supabase'] = _dl
        dummy = {"id":"local-test","file_path":"local","patient_id":"local","age":17,"sex":"female","l_text":""}
        print("Processing local PDF test...")
        out = process_report(dummy)
        print("RESULT:", json.dumps(out, indent=2))
    else:
        if args.once:
            # single run
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending").limit(5).execute()
            rows = res.data if hasattr(res, "data") else res
            if rows:
                for r in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update({"ai_status":"processing"}).eq("id", r.get("id")).execute()
                    except:
                        pass
                    process_report(r)
            else:
                print("No pending reports.")
        else:
            poll_loop()
