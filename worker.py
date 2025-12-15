#!/usr/bin/env python3
"""
AMI Health Worker — Production Medical AI Pipeline

SAFETY-CRITICAL SYSTEM
• Zero hallucinations
• No inferred labs
• Conservative, doctor-trust first
• CBC + Chemistry
• Severity TEXT only
• No admission recommendations
"""

# =====================================================
# STANDARD LIBRARY IMPORTS
# =====================================================

import os
import io
import re
import time
import json
import base64
import traceback
from typing import Dict, Any, Optional, List

# =====================================================
# THIRD-PARTY IMPORTS
# =====================================================

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# Optional OCR (local)
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# Supabase
from supabase import create_client, Client

# OpenAI (Vision + interpreter later)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# =====================================================
# ENVIRONMENT VARIABLES
# =====================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_KEY")
    or os.getenv("SUPABASE_KEY")
)

SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

MAX_OCR_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1000"))
JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "55"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL or SUPABASE_KEY missing — worker cannot start"
    )

# =====================================================
# CLIENT INITIALISATION
# =====================================================

supabase: Optional[Client] = None
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print("⚠️ Supabase client initialisation failed:", e)

openai_client = None
if OpenAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("⚠️ OpenAI client initialisation failed:", e)

# =====================================================
# TIME / NUMERIC UTILITIES
# =====================================================

def now_iso() -> str:
    """
    UTC timestamp in ISO-8601 format.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_float(value: Any) -> Optional[float]:
    """
    Convert to float safely.
    Returns None if conversion fails.
    """
    try:
        return float(value)
    except Exception:
        return None


def numeric_safety_gate(key: str, value: float) -> Optional[float]:
    """
    HARD physiologic safety limits.
    Values outside these ranges are discarded.
    This prevents hallucinated or corrupted numbers.
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

# =====================================================
# END PART 1
# =====================================================
# =====================================================
# PDF INGESTION
# =====================================================

def download_pdf_from_supabase(job: Dict[str, Any]) -> bytes:
    """
    Download PDF bytes from Supabase storage or direct pdf_url.
    HARDENED: retries and defensive handling.
    """

    # Direct URL override (used in some pipelines)
    if job.get("pdf_url"):
        import requests
        resp = requests.get(job["pdf_url"], timeout=30)
        resp.raise_for_status()
        return resp.content

    if not supabase or not job.get("file_path"):
        raise RuntimeError("No pdf_url or file_path provided")

    last_error = None

    for attempt in range(2):
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(
                job["file_path"]
            )

            # Supabase SDK inconsistencies
            if hasattr(res, "data") and res.data:
                return res.data

            if isinstance(res, (bytes, bytearray)):
                return res

            raise RuntimeError("Empty response from Supabase")

        except Exception as e:
            last_error = e
            print(f"⚠️ Supabase download attempt {attempt + 1} failed:", e)
            time.sleep(1.5)

    raise RuntimeError(f"Supabase download failed: {last_error}")


# =====================================================
# PDF TEXT EXTRACTION (DIGITAL)
# =====================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract selectable text using pypdf.
    Returns empty string if extraction fails.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                continue
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("pypdf extraction failed:", e)
        return ""


def is_scanned_pdf(pdf_bytes: bytes, threshold_chars: int = 80) -> bool:
    """
    Heuristic:
    - Very little or no selectable text → scanned PDF
    """
    text = extract_text_from_pdf(pdf_bytes)
    if not text or len(text.strip()) < threshold_chars:
        return True
    return False


# =====================================================
# OCR IMAGE PREPROCESSING
# =====================================================

def preprocess_image_for_ocr(
    img: Image.Image,
    max_width: int = MAX_OCR_WIDTH,
    quality: int = JPEG_QUALITY
) -> bytes:
    """
    Resize + grayscale + compress image for OCR.
    """

    try:
        img = ImageOps.exif_transpose(img)
        w, h = img.size

        if w > max_width:
            new_h = int((max_width / float(w)) * h)
            img = img.resize((max_width, new_h), Image.LANCZOS)

        img = img.convert("L")

        buf = io.BytesIO()
        img.save(
            buf,
            format="JPEG",
            quality=quality,
            optimize=True
        )
        return buf.getvalue()

    except Exception as e:
        print("Image preprocess failed:", e)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


# =====================================================
# OCR (VISION)
# =====================================================

def ocr_image_with_openai(img_bytes: bytes) -> Dict[str, Any]:
    """
    Send image to OpenAI Vision.
    STRICT JSON ONLY.
    """

    if not openai_client:
        return {"cbc": []}

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant for laboratory reports. "
        "Extract numeric lab values with analyte name, value, and units. "
        "Return STRICT JSON only:\n"
        "{ \"cbc\": [ { \"analyte\": \"Hb\", \"value\": 11.6, \"units\": \"g/dL\" } ] }"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract lab values from this image."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw = resp.choices[0].message.content

        if isinstance(raw, dict):
            return raw

        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r"\{.*\}", raw, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass

        return {"cbc": []}

    except Exception as e:
        print("OpenAI OCR failed:", e)
        return {"cbc": []}


def do_ocr_on_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Render PDF pages and OCR each page.
    Returns list of OCR JSON dicts.
    """

    outputs: List[Dict[str, Any]] = []

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        print("pdf2image failed:", e)
        return outputs

    for idx, page in enumerate(pages):
        try:
            img_bytes = preprocess_image_for_ocr(page)
            ocr_json = ocr_image_with_openai(img_bytes)
            outputs.append(ocr_json)
        except Exception as e:
            print(f"OCR failed on page {idx}:", e)
            outputs.append({"cbc": []})

    return outputs

# =====================================================
# END PART 2
# =====================================================
# =====================================================
# LABEL NORMALISATION
# =====================================================

SYNONYMS = {
    # CBC
    "hb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "wbc": "WBC",
    "white cell count": "WBC",
    "leukocytes": "WBC",
    "leucocytes": "WBC",
    "neutrophils": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "monocytes": "Monocytes",
    "eosinophils": "Eosinophils",
    "basophils": "Basophils",
    "platelets": "Platelets",

    # Inflammation / renal
    "crp": "CRP",
    "c reactive protein": "CRP",
    "creatinine": "Creatinine",
    "urea": "Urea",

    # Electrolytes
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "chloride": "Chloride",
    "cl": "Chloride",

    # Liver
    "bilirubin": "Bilirubin",
    "bilirubin total": "Bilirubin Total",
    "bilirubin conjugated": "Bilirubin Conjugated",
    "bilirubin unconjugated": "Bilirubin Unconjugated",
    "alt": "ALT",
    "ast": "AST",
    "alp": "ALP",
    "alkaline phosphatase": "ALP",
    "ggt": "GGT",

    # Lipids
    "cholesterol": "Cholesterol",
    "cholesterol total": "Cholesterol",
    "triglycerides": "Triglycerides",
    "hdl": "HDL",
    "ldl": "LDL",
    "non hdl": "Non-HDL",
    "non-hdl": "Non-HDL",

    # Muscle
    "ck": "CK",
    "creatine kinase": "CK",
}


def normalize_label(label: str) -> Optional[str]:
    """
    Map raw label text to canonical analyte name.
    """
    if not label:
        return None

    clean = re.sub(r"[^a-z0-9 ]", "", label.lower()).strip()

    for k, v in SYNONYMS.items():
        if k in clean:
            return v

    return None


# =====================================================
# FLAG EXTRACTION (H / L)
# =====================================================

def extract_hl_flag(token: Optional[str]) -> Optional[str]:
    """
    Convert H / L markers to semantic flags.
    """
    if not token:
        return None

    token = token.strip().upper()

    if token == "H":
        return "high"
    if token == "L":
        return "low"

    return None


# =====================================================
# CHEMISTRY TABLE PARSER (COLUMN-BASED)
# =====================================================

def parse_chemistry_table(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse chemistry-style tables like:
    CRP        < 5 mg/L     3
    Creatinine 64-104 umol  81
    """

    out: Dict[str, Dict[str, Any]] = {}

    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 6:
            continue

        # Try: LABEL ..... VALUE [H/L]
        m = re.match(
            r'^([A-Za-z][A-Za-z\s\-\(\)]+?)\s{2,}'
            r'(-?\d+\.?\d*)\s*([A-Za-z/%µ]+)?\s*([HL])?$',
            line
        )

        if not m:
            continue

        label_raw, value_raw, units, flag = m.groups()

        key = normalize_label(label_raw)
        val = safe_float(value_raw)

        if not key or val is None:
            continue

        out[key] = {
            "value": val,
            "units": units,
            "flag": extract_hl_flag(flag),
            "raw": line,
        }

    return out


# =====================================================
# INLINE VALUE PARSER (LABEL: VALUE UNIT H/L)
# =====================================================

def parse_inline_values(text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    for line in text.splitlines():
        matches = re.findall(
            r'([A-Za-z][A-Za-z\s\-/]{2,40})[:\s]{1,4}'
            r'(-?\d+\.?\d*)\s*([A-Za-z/%µ]+)?\s*([HL])?',
            line
        )

        for label_raw, value_raw, units, flag in matches:
            key = normalize_label(label_raw)
            val = safe_float(value_raw)

            if not key or val is None:
                continue

            out[key] = {
                "value": val,
                "units": units,
                "flag": extract_hl_flag(flag),
                "raw": line,
            }

    return out


# =====================================================
# OCR JSON PARSER
# =====================================================

def parse_values_from_ocr_json(
    ocr_json: Dict[str, Any]
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}

    items = ocr_json.get("cbc") or []
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue

        label_raw = it.get("analyte") or it.get("label") or ""
        key = normalize_label(label_raw)
        val = safe_float(it.get("value"))

        if not key or val is None:
            continue

        out[key] = {
            "value": val,
            "units": it.get("units"),
            "flag": extract_hl_flag(it.get("flag")),
            "raw": it,
        }

    return out


# =====================================================
# MASTER PARSER ENTRYPOINT
# =====================================================

def parse_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Three-pass parser:
    1) Inline values
    2) Chemistry tables
    3) Deduplication (inline wins)
    """

    parsed: Dict[str, Dict[str, Any]] = {}

    inline = parse_inline_values(text)
    table = parse_chemistry_table(text)

    # Inline has priority
    parsed.update(table)
    parsed.update(inline)

    return parsed

# =====================================================
# END PART 3
# =====================================================
# =====================================================
# CANONICAL MAP (PURE, NO CLINICAL LOGIC)
# =====================================================

def canonical_map(
    parsed: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    HARD safety gate + canonical normalization.
    NO interpretation here.
    """

    canonical: Dict[str, Dict[str, Any]] = {}

    for key, item in parsed.items():
        val = safe_float(item.get("value"))
        if val is None:
            continue

        safe_val = numeric_safety_gate(key, val)
        if safe_val is None:
            continue

        canonical[key] = {
            "value": safe_val,
            "units": item.get("units"),
            "flag": item.get("flag"),
            "raw": item.get("raw"),
        }

    # -----------------------------
    # SAFE DERIVED VALUES
    # -----------------------------

    try:
        neut = canonical.get("Neutrophils", {}).get("value")
        lymph = canonical.get("Lymphocytes", {}).get("value")

        if neut is not None and lymph is not None and lymph > 0:
            canonical["NLR"] = {
                "value": round(neut / lymph, 2),
                "units": None,
                "flag": None,
                "raw": "derived NLR",
            }
    except Exception:
        pass

    return canonical


# =====================================================
# eGFR (CKD-EPI 2021, RACE-FREE)
# =====================================================

def calculate_egfr(
    creatinine_umol: float,
    age: float,
    sex: str
) -> Optional[float]:
    """
    CKD-EPI 2021 equation.
    Creatinine in µmol/L.
    """

    try:
        if creatinine_umol is None or age is None:
            return None

        scr = creatinine_umol / 88.4
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

# =====================================================
# END PART 4
# =====================================================
# =====================================================
# ROUTE ENGINE — PATTERNS → ROUTES → NEXT STEPS
# =====================================================

SEVERITY_MAP = {
    1: "normal",
    2: "mild",
    3: "moderate",
    4: "severe",
    5: "critical",
}


def age_group(age: Optional[float]) -> str:
    if age is None:
        return "adult"
    try:
        a = float(age)
    except Exception:
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


def score_severity(key: str, value: Optional[float], sex: str) -> int:
    """
    Conservative per-analyte severity.
    """
    if value is None:
        return 1

    k = key.lower()
    sex = (sex or "").lower()

    try:
        if k == "hb":
            low = 12.0 if sex == "female" else 13.0
            if value < low - 4:
                return 5
            if value < low - 2:
                return 4
            if value < low:
                return 3
            return 1

        if k == "wbc":
            if value > 25:
                return 5
            if value > 15:
                return 4
            if value > 11:
                return 3
            return 1

        if k == "platelets":
            if value < 20:
                return 5
            if value < 50:
                return 4
            if value < 100:
                return 3
            return 1

        if k == "crp":
            if value > 200:
                return 5
            if value > 100:
                return 4
            if value > 50:
                return 3
            if value > 10:
                return 2
            return 1

        if k == "creatinine":
            if value > 350:
                return 5
            if value > 200:
                return 4
            if value > 120:
                return 3
            return 1

        if k == "potassium":
            if value < 3.0 or value > 6.0:
                return 5
            if value < 3.5 or value > 5.5:
                return 3
            return 1

        if k == "sodium":
            if value < 120 or value > 160:
                return 5
            if value < 125 or value > 155:
                return 3
            return 1

        if k == "ck":
            if value > 10000:
                return 5
            if value > 5000:
                return 4
            if value > 2000:
                return 3
            return 1

        if k == "nlr":
            if value > 10:
                return 4
            if value > 5:
                return 3
            return 1

    except Exception:
        return 1

    return 1


def route_engine_all(
    canonical: Dict[str, Dict[str, Any]],
    patient_meta: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    doctor_trust_flags: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Core clinical reasoning engine.
    """

    age = patient_meta.get("age")
    sex = patient_meta.get("sex") or "unknown"
    ag = age_group(age)

    patterns: List[Dict[str, str]] = []
    routes: List[str] = []
    next_steps: List[str] = []
    differentials: List[str] = []
    per_analyte: Dict[str, Any] = {}

    severity_scores: List[int] = []

    # ----------------------------------
    # Per-analyte scoring
    # ----------------------------------

    for key, item in canonical.items():
        val = item.get("value")
        sev = score_severity(key, val, sex)
        per_analyte[key] = {
            "value": val,
            "units": item.get("units"),
            "flag": item.get("flag"),
            "severity": SEVERITY_MAP[sev],
        }
        severity_scores.append(sev)

    # ----------------------------------
    # Convenience values
    # ----------------------------------

    Hb = canonical.get("Hb", {}).get("value")
    MCV = canonical.get("MCV", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")
    Neut = canonical.get("Neutrophils", {}).get("value")
    Lymph = canonical.get("Lymphocytes", {}).get("value")
    Plate = canonical.get("Platelets", {}).get("value")
    CRP = canonical.get("CRP", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")
    CK = canonical.get("CK", {}).get("value")
    K = canonical.get("Potassium", {}).get("value")
    Na = canonical.get("Sodium", {}).get("value")
    LDL = canonical.get("LDL", {}).get("value")
    TG = canonical.get("Triglycerides", {}).get("value")
    NonHDL = canonical.get("Non-HDL", {}).get("value")

    # ----------------------------------
    # Anaemia
    # ----------------------------------

    if Hb is not None:
        patterns.append({"pattern": "anaemia", "reason": f"Hb {Hb}"})
        if MCV is not None:
            if MCV < 80:
                routes.append("Microcytic anaemia pathway")
                differentials += [
                    "Iron deficiency",
                    "Thalassaemia trait",
                    "Chronic blood loss",
                ]
                next_steps.append(
                    "Check ferritin, iron studies, and reticulocyte count"
                )
            elif MCV > 100:
                routes.append("Macrocytic anaemia pathway")
                differentials += [
                    "Vitamin B12 deficiency",
                    "Folate deficiency",
                    "Alcohol-related",
                ]
                next_steps.append(
                    "Check vitamin B12, folate, and review medications"
                )
            else:
                routes.append("Normocytic anaemia pathway")
                differentials += [
                    "Anaemia of chronic disease",
                    "Renal disease",
                    "Acute blood loss",
                ]
                next_steps.append(
                    "Check reticulocyte count and inflammatory markers"
                )

    # ----------------------------------
    # Infection / inflammation
    # ----------------------------------

    if WBC is not None and WBC > 11:
        patterns.append({"pattern": "leukocytosis", "reason": f"WBC {WBC}"})
        differentials.append("Infection")

    if CRP is not None and CRP > 10:
        patterns.append({"pattern": "elevated CRP", "reason": f"CRP {CRP}"})
        next_steps.append(
            "Correlate with clinical signs of infection or inflammation"
        )

    # ----------------------------------
    # Platelets
    # ----------------------------------

    if Plate is not None and Plate < 150:
        patterns.append(
            {"pattern": "thrombocytopenia", "reason": f"Platelets {Plate}"}
        )
        differentials.append("Bone marrow suppression")
        next_steps.append("Repeat platelet count and review smear")

    # ----------------------------------
    # Renal
    # ----------------------------------

    if Creat is not None and Creat > 120:
        patterns.append(
            {"pattern": "renal impairment", "reason": f"Creatinine {Creat}"}
        )
        differentials.append("Acute kidney injury")
        next_steps.append(
            "Review renal function trend and hydration status"
        )

    # ----------------------------------
    # Electrolytes
    # ----------------------------------

    if K is not None and (K < 3.5 or K > 5.5):
        patterns.append(
            {"pattern": "potassium abnormality", "reason": f"K {K}"}
        )
        next_steps.append(
            "Assess ECG risk and contributing medications"
        )

    if Na is not None and (Na < 130 or Na > 150):
        patterns.append(
            {"pattern": "sodium abnormality", "reason": f"Na {Na}"}
        )
        next_steps.append(
            "Assess volume status and neurologic symptoms"
        )

    # ----------------------------------
    # Rhabdomyolysis
    # ----------------------------------

    if CK is not None and CK > 2000:
        patterns.append({"pattern": "elevated CK", "reason": f"CK {CK}"})
        next_steps.append(
            "Monitor renal function and electrolytes"
        )

    # ----------------------------------
    # Long-term cardiovascular risk
    # ----------------------------------

    if (
        (LDL is not None and LDL >= 3.0)
        or (TG is not None and TG >= 1.7)
        or (NonHDL is not None and NonHDL >= 3.4)
    ):
        doctor_trust_flags["has_long_term_risk"] = True
        patterns.append(
            {
                "pattern": "long-term cardiovascular risk",
                "reason": "Lipid abnormalities detected",
            }
        )
        next_steps.append(
            "Address cardiovascular risk factors in routine follow-up"
        )

    # ----------------------------------
    # Final severity
    # ----------------------------------

    overall_sev = max(severity_scores) if severity_scores else 1

    if doctor_trust_flags.get("has_long_term_risk") and overall_sev == 1:
        overall_sev = 2  # force non-normal

    severity_text = SEVERITY_MAP[overall_sev]

    # ----------------------------------
    # Summary
    # ----------------------------------

    if not patterns and doctor_trust_flags.get("has_long_term_risk"):
        summary = (
            "No acute abnormalities detected. "
            "Long-term cardiovascular risk assessment recommended."
        )
    elif not patterns:
        summary = "No acute abnormalities detected."
    else:
        summary = " | ".join(
            [p["pattern"] for p in patterns]
        )

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differentials": list(dict.fromkeys(differentials)),
        "per_analyte": per_analyte,
        "overall_severity": overall_sev,
        "severity_text": severity_text,
        "summary": summary,
    }

# =====================================================
# END PART 5
# =====================================================
# =====================================================
# MAIN REPORT PROCESSOR
# =====================================================

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id") or job.get("report_id")
    patient_id = job.get("patient_id")
    file_path = job.get("file_path")
    pdf_url = job.get("pdf_url")
    l_text = job.get("l_text") or ""
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    print(f"Processing report {report_id}")

    try:
        if not file_path and not pdf_url:
            raise RuntimeError("Missing file_path or pdf_url")

        # ----------------------------------
        # Download PDF
        # ----------------------------------

        pdf_bytes = download_pdf_from_supabase(job)

        # ----------------------------------
        # Detect scanned vs digital
        # ----------------------------------

        scanned = is_scanned_pdf(pdf_bytes)

        parsed: Dict[str, Dict[str, Any]] = {}
        merged_text_for_ai = ""

        # ----------------------------------
        # Extract text / OCR
        # ----------------------------------

        if scanned:
            print("Detected SCANNED PDF — running OCR")
            ocr_pages = do_ocr_on_pdf(pdf_bytes)

            combined_items: List[Dict[str, Any]] = []
            for p in ocr_pages:
                if isinstance(p, dict) and p.get("cbc"):
                    combined_items.extend(p.get("cbc"))

            if combined_items:
                parsed = parse_values_from_ocr_json({"cbc": combined_items})
                merged_text_for_ai = json.dumps({"cbc": combined_items})
            else:
                text = extract_text_from_pdf(pdf_bytes)
                parsed = parse_values_from_text(text)
                merged_text_for_ai = text

        else:
            print("Detected DIGITAL PDF — extracting text")
            text = extract_text_from_pdf(pdf_bytes)
            merged_text_for_ai = (
                (l_text + "\n\n" + text).strip()
                if l_text else text
            )
            parsed = parse_values_from_text(merged_text_for_ai)

        # ----------------------------------
        # Canonical map (HARD GATE)
        # ----------------------------------

        canonical = canonical_map(parsed)

        # ----------------------------------
        # Doctor-trust flags (INITIALISE FIRST)
        # ----------------------------------

        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False,
        }

        # ----------------------------------
        # Long-term risk guardrails
        # ----------------------------------

        ldl = canonical.get("LDL", {}).get("value")
        tg = canonical.get("Triglycerides", {}).get("value")
        non_hdl = canonical.get("Non-HDL", {}).get("value")

        if ldl is not None and ldl >= 3.0:
            doctor_trust_flags["has_long_term_risk"] = True
        if tg is not None and tg >= 1.7:
            doctor_trust_flags["has_long_term_risk"] = True
        if non_hdl is not None and non_hdl >= 3.4:
            doctor_trust_flags["has_long_term_risk"] = True

        # ----------------------------------
        # eGFR calculation
        # ----------------------------------

        creat = canonical.get("Creatinine", {}).get("value")
        if creat is not None and patient_age:
            egfr = calculate_egfr(creat, patient_age, patient_sex)
            if egfr is not None:
                canonical["eGFR"] = {
                    "value": egfr,
                    "units": "mL/min/1.73m²",
                    "flag": None,
                    "raw": "calculated (CKD-EPI 2021)",
                }

        # ----------------------------------
        # Fetch previous ai_results (FIXED)
        # ----------------------------------

        previous = None
        try:
            if supabase and patient_id:
                prev_q = (
                    supabase
                    .table(SUPABASE_TABLE)
                    .select("ai_results,created_at")
                    .eq("patient_id", patient_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                prev_rows = (
                    prev_q.data if hasattr(prev_q, "data") else prev_q
                )
                if prev_rows:
                    previous = prev_rows[0].get("ai_results")
        except Exception:
            previous = None

        # ----------------------------------
        # Trend analysis
        # ----------------------------------

        trends = trend_analysis(canonical, previous)

        # ----------------------------------
        # Route engine
        # ----------------------------------

        route_info = route_engine_all(
            canonical,
            {"age": patient_age, "sex": patient_sex},
            previous,
            doctor_trust_flags,
        )

        # ----------------------------------
        # Final AI interpretation (optional)
        # ----------------------------------

        ai_interpretation = {}
        if openai_client and merged_text_for_ai:
            try:
                ai_interpretation = call_ai_on_report(merged_text_for_ai)
            except Exception:
                ai_interpretation = {}

        # ----------------------------------
        # Final payload
        # ----------------------------------

        ai_results = {
            "processed_at": now_iso(),
            "scanned": scanned,
            "canonical": canonical,
            "routes": route_info,
            "trends": trends,
            "doctor_trust_flags": doctor_trust_flags,
            "ai_interpretation": ai_interpretation,
        }

        # ----------------------------------
        # Save to Supabase
        # ----------------------------------

        try:
            supabase.table(SUPABASE_TABLE).update(
                {
                    "ai_status": "completed",
                    "ai_results": ai_results,
                    "ai_error": None,
                }
            ).eq("id", report_id).execute()
        except Exception as e:
            print("Supabase save failed:", e)

        print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_results}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        try:
            supabase.table(SUPABASE_TABLE).update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
        except Exception:
            pass

        return {"error": err}

# =====================================================
# END PART 6
# =====================================================
# =====================================================
# TREND ANALYSIS
# =====================================================

def trend_analysis(
    current: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Simple numeric trend analysis.
    No interpretation — numbers only.
    """
    if not previous:
        return {"trend": "no_previous"}

    diffs: Dict[str, Any] = {}

    for key, item in current.items():
        cur = item.get("value")
        prev = None

        try:
            prev = previous.get("canonical", {}).get(key, {}).get("value")
        except Exception:
            prev = None

        if cur is None or prev is None:
            continue

        try:
            delta = cur - prev
            pct = (delta / prev) * 100 if prev != 0 else None
            diffs[key] = {
                "previous": prev,
                "current": cur,
                "delta": round(delta, 3),
                "pct_change": round(pct, 1) if pct is not None else None,
            }
        except Exception:
            continue

    return {"trend": diffs if diffs else "no_change"}


# =====================================================
# OPTIONAL AI INTERPRETER (ADVISORY ONLY)
# =====================================================

def call_ai_on_report(text: str) -> Dict[str, Any]:
    """
    Advisory-only AI interpretation.
    NEVER authoritative.
    """
    if not openai_client:
        return {}

    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are a conservative clinical assistant. "
        "Describe lab abnormalities only. "
        "No diagnoses. No admission advice. "
        "Return STRICT JSON."
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        raw = resp.choices[0].message.content
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                return {}
        return {}

    except Exception as e:
        print("AI interpreter failed:", e)
        return {}


# =====================================================
# POLLING LOOP
# =====================================================

def poll_loop() -> None:
    """
    Continuously poll Supabase for pending reports.
    """
    if not supabase:
        print("Supabase not configured — poll loop disabled.")
        return

    print("🚀 AMI Health Worker started — polling for pending reports")

    while True:
        try:
            res = (
                supabase
                .table(SUPABASE_TABLE)
                .select("*")
                .eq("ai_status", "pending")
                .limit(5)
                .execute()
            )

            rows = res.data if hasattr(res, "data") else res

            if rows:
                for job in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update(
                            {"ai_status": "processing"}
                        ).eq("id", job.get("id")).execute()
                    except Exception:
                        pass

                    process_report(job)
            else:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            print("Polling error:", e)
            traceback.print_exc()
            time.sleep(5)


# =====================================================
# CLI HARNESS
# =====================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AMI Health Worker — Production"
    )
    parser.add_argument(
        "--test-pdf",
        help="Path to local PDF for testing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Poll once and exit",
    )

    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdf_bytes = f.read()

        def _mock_download(_: Dict[str, Any]) -> bytes:
            return pdf_bytes

        globals()["download_pdf_from_supabase"] = _mock_download

        dummy_job = {
            "id": "local-test",
            "file_path": "local",
            "patient_id": "local",
            "age": 45,
            "sex": "male",
            "l_text": "",
        }

        print("Running local PDF test...")
        result = process_report(dummy_job)
        print(json.dumps(result, indent=2))

    else:
        if args.once:
            res = (
                supabase
                .table(SUPABASE_TABLE)
                .select("*")
                .eq("ai_status", "pending")
                .limit(5)
                .execute()
            )
            rows = res.data if hasattr(res, "data") else res

            if rows:
                for job in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update(
                            {"ai_status": "processing"}
                        ).eq("id", job.get("id")).execute()
                    except Exception:
                        pass
                    process_report(job)
            else:
                print("No pending reports.")
        else:
            poll_loop()

# =====================================================
# END PART 7 — WORKER COMPLETE
# =====================================================
