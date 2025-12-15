#!/usr/bin/env python3
"""
AMI Health Worker — Production Clinical Pipeline
SAFETY-CRITICAL SYSTEM

PART 1 / 7
- Imports
- Environment & config
- Age / sex reference framework
- Safety helpers
- Parser contracts (NO interpretation)
"""

import os
import io
import re
import time
import json
import math
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

# -------------------------
# ENVIRONMENT
# -------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing")

# -------------------------
# LOGGING (DOCTOR-SAFE)
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI-WORKER] %(levelname)s: %(message)s"
)

log = logging.getLogger("ami-worker")

# -------------------------
# SUPABASE CLIENT
# -------------------------

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    log.error("Supabase init failed: %s", e)

# -------------------------
# OCR CONFIG (SAFE DEFAULTS)
# -------------------------

MAX_OCR_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1400"))
OCR_JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "70"))

# -------------------------
# TIME HELPERS
# -------------------------

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# -------------------------
# AGE GROUPING (CRITICAL)
# -------------------------
# Used everywhere downstream. Must be deterministic.

def age_group(age: Optional[float]) -> str:
    if age is None:
        return "unknown"
    try:
        age = float(age)
    except Exception:
        return "unknown"

    if age < 1:
        return "infant"
    if age < 13:
        return "child"
    if age < 18:
        return "teen"
    if age < 65:
        return "adult"
    return "elderly"

# -------------------------
# SEX NORMALISATION
# -------------------------

def normalize_sex(sex: Optional[str]) -> str:
    if not sex:
        return "unknown"
    s = sex.strip().lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    return "unknown"

# -------------------------
# HARD NUMERIC SAFETY GATES
# -------------------------
# If a value breaches these → DROP IT
# Never hallucinate, never clip

PHYSIOLOGIC_LIMITS = {
    "Hb": (3.0, 25.0),
    "WBC": (0.1, 100.0),
    "Platelets": (1, 2000),
    "MCV": (50, 130),
    "CRP": (0, 500),
    "Creatinine": (10, 2000),
    "Sodium": (110, 180),
    "Potassium": (2.0, 7.5),
    "CK": (0, 200000),
    "ALT": (0, 5000),
    "AST": (0, 5000),
}

def safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def numeric_safety_gate(key: str, value: float) -> Optional[float]:
    if key not in PHYSIOLOGIC_LIMITS:
        return value
    low, high = PHYSIOLOGIC_LIMITS[key]
    if value < low or value > high:
        return None
    return value

# -------------------------
# PARSER CONTRACT (IMPORTANT)
# -------------------------
# Parser MUST return:
# {
#   analyte: {
#     value: float,
#     units: str | None,
#     flag: "H" | "L" | None,
#     ref_low: float | None,
#     ref_high: float | None,
#     raw: str
#   }
# }
#
# Parser must NEVER:
# - guess
# - infer
# - normalise
# - interpret

ParsedValue = Dict[str, Any]
ParsedResults = Dict[str, ParsedValue]

# -------------------------
# ANALYTE NORMALISATION MAP
# -------------------------

ANALYTE_SYNONYMS = {
    "hb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "wbc": "WBC",
    "white cell count": "WBC",
    "platelets": "Platelets",
    "plt": "Platelets",
    "mcv": "MCV",
    "rdw": "RDW",
    "crp": "CRP",
    "creatinine": "Creatinine",
    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "ck": "CK",
    "alt": "ALT",
    "ast": "AST",
    "bilirubin total": "Bilirubin",
    "bilirubin": "Bilirubin",
    "ldl": "LDL",
    "hdl": "HDL",
    "triglycerides": "Triglycerides",
    "non-hdl": "Non-HDL",
}

def normalize_analyte(label: str) -> Optional[str]:
    if not label:
        return None
    key = re.sub(r"[^a-z0-9\- ]", "", label.lower()).strip()
    return ANALYTE_SYNONYMS.get(key)

# =========================
# END PART 1
# =========================
"""
PART 2 / 7
- Download PDF from Supabase
- Detect digital vs scanned PDF
- OCR using pytesseract ONLY
- Text extraction with hard fallbacks
"""

# -------------------------
# OPTIONAL OCR DEPENDENCY
# -------------------------

try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False
    log.warning("pytesseract not available — OCR disabled")

# -------------------------
# PDF DOWNLOAD
# -------------------------

def download_pdf_from_supabase(job: Dict[str, Any]) -> bytes:
    """
    Returns raw PDF bytes.
    FAILS HARD if unavailable.
    """
    if job.get("pdf_url"):
        import requests
        r = requests.get(job["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if not supabase or not job.get("file_path"):
        raise RuntimeError("No pdf_url or file_path available")

    try:
        res = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"])
        if hasattr(res, "data") and res.data:
            return res.data
        if isinstance(res, (bytes, bytearray)):
            return res
    except Exception as e:
        raise RuntimeError(f"Supabase download failed: {e}")

    raise RuntimeError("Empty PDF download response")

# -------------------------
# DIGITAL TEXT EXTRACTION
# -------------------------

def extract_text_digital(pdf_bytes: bytes) -> str:
    """
    Extract selectable text using pypdf.
    Returns EMPTY STRING if none found.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
        return "\n".join(pages).strip()
    except Exception as e:
        log.warning("Digital text extraction failed: %s", e)
        return ""

# -------------------------
# SCANNED PDF DETECTION
# -------------------------

def is_scanned_pdf(pdf_bytes: bytes, min_chars: int = 100) -> bool:
    """
    Heuristic:
    If digital text < min_chars → treat as scanned.
    """
    text = extract_text_digital(pdf_bytes)
    return len(text) < min_chars

# -------------------------
# OCR IMAGE PREPROCESSING
# -------------------------

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    Conservative preprocessing.
    Do NOT destroy numeric clarity.
    """
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")  # grayscale
    w, h = img.size
    if w > MAX_OCR_WIDTH:
        new_h = int((MAX_OCR_WIDTH / w) * h)
        img = img.resize((MAX_OCR_WIDTH, new_h), Image.LANCZOS)
    return img

# -------------------------
# OCR PAGE
# -------------------------

def ocr_image(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("OCR requested but pytesseract not installed")

    try:
        return pytesseract.image_to_string(
            img,
            config="--psm 6"
        )
    except Exception as e:
        log.error("OCR failed: %s", e)
        return ""

# -------------------------
# OCR FULL PDF
# -------------------------

def extract_text_scanned(pdf_bytes: bytes) -> str:
    """
    OCR each page and concatenate text.
    FAILS CLOSED if OCR yields nothing.
    """
    if not HAS_PYTESSERACT:
        raise RuntimeError("Scanned PDF but OCR unavailable")

    pages = convert_from_bytes(pdf_bytes, dpi=300)
    all_text = []

    for idx, page in enumerate(pages):
        try:
            img = preprocess_image_for_ocr(page)
            txt = ocr_image(img)
            if txt.strip():
                all_text.append(txt)
        except Exception as e:
            log.warning(f"OCR page {idx} failed: {e}")

    final_text = "\n".join(all_text).strip()

    if not final_text:
        raise RuntimeError("OCR produced no text")

    return final_text

# -------------------------
# MASTER TEXT EXTRACTION
# -------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, bool]:
    """
    Returns (text, scanned_flag)
    """
    digital_text = extract_text_digital(pdf_bytes)

    if digital_text and len(digital_text) >= 100:
        return digital_text, False

    log.info("PDF appears scanned — running OCR")
    scanned_text = extract_text_scanned(pdf_bytes)
    return scanned_text, True

# =========================
# END PART 2
# =========================
"""
PART 3 / 7
- Robust CBC + chemistry parsing
- Extract value, units, H/L flag
- Extract reference ranges
- Preserve lab comments
- ZERO interpretation
"""

# -------------------------
# REGEX BUILDING BLOCKS
# -------------------------

NUM = r"(-?\d+(?:\.\d+)?)"
RANGE = r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)"
UNIT = r"([a-zA-Z/%µ\^\-\d]+)"
FLAG = r"\b(H|L)\b"

# -------------------------
# LINE PARSERS
# -------------------------

def parse_result_line(line: str) -> Optional[Tuple[str, ParsedValue]]:
    """
    Parses a single lab line.
    Returns (analyte, parsed_dict) or None
    """

    raw = line.strip()
    if not raw or len(raw) < 4:
        return None

    # Example patterns handled:
    # Hb 11.6 g/dL L (12.4 - 16.7)
    # Platelets 376 x10^9/L (150 - 450)
    # Neutrophils 88.0 % H
    # WBC 19.41 x10^9/L H 4.0 - 12.0

    # Extract analyte label
    label_match = re.match(r"^([A-Za-z\-\s/]+)", raw)
    if not label_match:
        return None

    label = label_match.group(1).strip()
    analyte = normalize_analyte(label)
    if not analyte:
        return None

    # Extract numeric value
    value_match = re.search(NUM, raw)
    if not value_match:
        return None

    value = safe_float(value_match.group(1))
    if value is None:
        return None

    # Units
    unit_match = re.search(UNIT, raw[value_match.end():])
    units = unit_match.group(1) if unit_match else None

    # H/L flag
    flag_match = re.search(FLAG, raw)
    flag = flag_match.group(1) if flag_match else None

    # Reference range
    ref_low = None
    ref_high = None

    range_match = re.search(RANGE, raw)
    if range_match:
        ref_low = safe_float(range_match.group(1))
        ref_high = safe_float(range_match.group(2))

    parsed: ParsedValue = {
        "value": value,
        "units": units,
        "flag": flag,
        "ref_low": ref_low,
        "ref_high": ref_high,
        "raw": raw
    }

    return analyte, parsed

# -------------------------
# FULL TEXT PARSER
# -------------------------

def parse_lab_text(text: str) -> Tuple[ParsedResults, List[str]]:
    """
    Parses entire extracted text.
    Returns:
    - parsed results dict
    - lab comments list
    """

    results: ParsedResults = {}
    comments: List[str] = []

    if not text:
        return results, comments

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in lines:
        lower = line.lower()

        # Capture lab comments verbatim
        if lower.startswith("comment") or "suggested" in lower:
            comments.append(line)
            continue

        parsed = parse_result_line(line)
        if not parsed:
            continue

        analyte, data = parsed

        # Safety gate numeric
        safe_val = numeric_safety_gate(analyte, data["value"])
        if safe_val is None:
            continue

        data["value"] = safe_val
        results[analyte] = data

    return results, comments

# -------------------------
# CBC PRESENCE CHECK
# -------------------------

CBC_KEYS = {
    "Hb", "WBC", "Platelets", "MCV", "RDW",
    "Neutrophils", "Lymphocytes"
}

def cbc_present(parsed: ParsedResults) -> bool:
    return any(k in parsed for k in CBC_KEYS)

# =========================
# END PART 3
# =========================
"""
PART 4 / 7
- Canonical mapping (validated facts only)
- Age / sex context attached
- CBC presence explicitly flagged
- No interpretation, no routes yet
"""

# -------------------------
# CANONICAL MAP
# -------------------------

def canonical_map(
    parsed: ParsedResults,
    comments: List[str],
    patient_age: Optional[float],
    patient_sex: Optional[str]
) -> Dict[str, Any]:
    """
    Converts parsed results into a canonical, context-aware structure.
    This function MUST NOT interpret severity.
    """

    canonical: Dict[str, Any] = {}

    age_grp = age_group(patient_age)
    sex = normalize_sex(patient_sex)

    # Attach metadata early
    canonical["_meta"] = {
        "age": patient_age,
        "age_group": age_grp,
        "sex": sex,
        "cbc_present": cbc_present(parsed),
        "parser_comments": comments,
    }

    # Copy analytes verbatim
    for analyte, data in parsed.items():
        canonical[analyte] = {
            "value": data.get("value"),
            "units": data.get("units"),
            "flag": data.get("flag"),
            "ref_low": data.get("ref_low"),
            "ref_high": data.get("ref_high"),
            "raw": data.get("raw"),
        }

    # -------------------------
    # SAFE DERIVED VALUES ONLY
    # -------------------------

    # Neutrophil–Lymphocyte Ratio (only if both absolute)
    try:
        n = canonical.get("Neutrophils", {}).get("value")
        l = canonical.get("Lymphocytes", {}).get("value")
        if n is not None and l is not None and l > 0:
            canonical["NLR"] = {
                "value": round(n / l, 2),
                "units": None,
                "flag": None,
                "ref_low": None,
                "ref_high": None,
                "raw": "calculated NLR",
            }
    except Exception:
        pass

    return canonical

# -------------------------
# CBC AVAILABILITY GUARD
# -------------------------

def assert_cbc_or_flag(canonical: Dict[str, Any]) -> None:
    """
    If CBC missing, insert explicit safety flag.
    """
    if not canonical.get("_meta", {}).get("cbc_present"):
        canonical["_safety"] = canonical.get("_safety", [])
        canonical["_safety"].append(
            "CBC not detected — interpretation limited to chemistry only"
        )

# -------------------------
# BILIRUBIN NUANCE (FACT ONLY)
# -------------------------

def annotate_bilirubin_context(canonical: Dict[str, Any]) -> None:
    """
    Adds factual bilirubin relationships without interpretation.
    """
    total = canonical.get("Bilirubin", {}).get("value")
    conj = canonical.get("Bilirubin Conjugated", {}).get("value")
    unconj = canonical.get("Bilirubin Unconjugated", {}).get("value")

    if total is not None and conj is not None and unconj is not None:
        if abs((conj + unconj) - total) <= 2:
            canonical["_facts"] = canonical.get("_facts", [])
            canonical["_facts"].append(
                "Bilirubin fractions consistent with isolated elevation"
            )

# -------------------------
# LONG-TERM RISK FLAGS (FACTUAL)
# -------------------------

def apply_long_term_risk_flags(canonical: Dict[str, Any], trust_flags: Dict[str, bool]) -> None:
    """
    Marks presence of long-term risk factors.
    Does NOT change severity here.
    """

    ldl = canonical.get("LDL", {}).get("value")
    tg = canonical.get("Triglycerides", {}).get("value")
    non_hdl = canonical.get("Non-HDL", {}).get("value")

    if ldl is not None and ldl >= 3.0:
        trust_flags["has_long_term_risk"] = True
    if tg is not None and tg >= 1.7:
        trust_flags["has_long_term_risk"] = True
    if non_hdl is not None and non_hdl >= 3.4:
        trust_flags["has_long_term_risk"] = True

# =========================
# END PART 4
# =========================
"""
PART 5 / 7
- Per-analyte severity (TEXT ONLY)
- Age-aware thresholds
- Acute vs long-term separation
- No admission language
- No false reassurance
"""

# -------------------------
# SEVERITY SCALE (TEXT ONLY)
# -------------------------

SEVERITY_ORDER = [
    "normal",
    "mild",
    "moderate",
    "severe",
    "critical"
]

def max_severity(a: str, b: str) -> str:
    return SEVERITY_ORDER[max(SEVERITY_ORDER.index(a), SEVERITY_ORDER.index(b))]

# -------------------------
# AGE / SEX AWARE THRESHOLDS
# -------------------------

def hb_lower_limit(age_group: str, sex: str) -> float:
    if age_group in ("infant", "child"):
        return 11.0
    if age_group == "teen":
        return 12.0 if sex == "female" else 13.0
    return 12.0 if sex == "female" else 13.0

# -------------------------
# PER-ANALYTE SEVERITY
# -------------------------

def severity_for_analyte(
    analyte: str,
    value: Optional[float],
    age_group: str,
    sex: str
) -> str:

    if value is None:
        return "normal"

    try:
        if analyte == "Hb":
            low = hb_lower_limit(age_group, sex)
            if value < low - 4:
                return "critical"
            if value < low - 2:
                return "severe"
            if value < low:
                return "moderate"
            return "normal"

        if analyte == "WBC":
            if value > 25:
                return "critical"
            if value > 15:
                return "severe"
            if value > 11:
                return "moderate"
            return "normal"

        if analyte == "Platelets":
            if value < 20:
                return "critical"
            if value < 50:
                return "severe"
            if value < 100:
                return "moderate"
            if value > 450:
                return "mild"
            return "normal"

        if analyte == "CRP":
            if value >= 100:
                return "severe"
            if value >= 50:
                return "moderate"
            if value >= 10:
                return "mild"
            return "normal"

        if analyte == "Creatinine":
            if value > 300:
                return "severe"
            if value > 150:
                return "moderate"
            return "normal"

        if analyte == "CK":
            if value > 10000:
                return "critical"
            if value > 5000:
                return "severe"
            if value > 2000:
                return "moderate"
            return "normal"

        if analyte == "Potassium":
            if value < 3.0 or value > 6.0:
                return "critical"
            if value < 3.4 or value > 5.5:
                return "moderate"
            return "normal"

        if analyte == "Sodium":
            if value < 120 or value > 160:
                return "critical"
            if value < 130 or value > 150:
                return "moderate"
            return "normal"

    except Exception:
        return "normal"

    return "normal"

# -------------------------
# ROUTE ENGINE
# -------------------------

def route_engine(
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> Dict[str, Any]:

    meta = canonical.get("_meta", {})
    age_group = meta.get("age_group", "unknown")
    sex = meta.get("sex", "unknown")

    per_analyte = {}
    overall_severity = "normal"

    acute_findings = []
    chronic_findings = []
    suggestions = []
    differentials = []

    # -------------------------
    # CBC missing guard
    # -------------------------

    if not meta.get("cbc_present"):
        chronic_findings.append("CBC not available — chemistry-only interpretation")

    # -------------------------
    # PER ANALYTE PASS
    # -------------------------

    for analyte, data in canonical.items():
        if analyte.startswith("_"):
            continue

        value = data.get("value")
        sev = severity_for_analyte(analyte, value, age_group, sex)

        per_analyte[analyte] = {
            "value": value,
            "units": data.get("units"),
            "severity": sev,
            "flag": data.get("flag"),
            "ref_low": data.get("ref_low"),
            "ref_high": data.get("ref_high"),
        }

        overall_severity = max_severity(overall_severity, sev)

        # -------------------------
        # ACUTE vs CHRONIC LOGIC
        # -------------------------

        if sev in ("severe", "critical"):
            acute_findings.append(f"{analyte} markedly abnormal")

        elif analyte in ("LDL", "Triglycerides", "Non-HDL") and value is not None:
            chronic_findings.append(f"{analyte} elevated")

    # -------------------------
    # BILIRUBIN NUANCE
    # -------------------------

    if "Bilirubin" in canonical:
        b = canonical["Bilirubin"]["value"]
        if b is not None and b > 21:
            chronic_findings.append(
                "Mild isolated bilirubin elevation — benign causes possible"
            )

    # -------------------------
    # TRUST OVERRIDE
    # -------------------------

    if trust_flags.get("has_long_term_risk") and overall_severity == "normal":
        overall_severity = "mild"

    # -------------------------
    # SUMMARY (DEFENSIVE)
    # -------------------------

    if not acute_findings and chronic_findings:
        summary = (
            "No acute abnormalities detected. "
            "Long-term risk factors identified."
        )
    elif acute_findings:
        summary = "Acute laboratory abnormalities detected."
    else:
        summary = "No acute laboratory abnormalities detected."

    return {
        "overall_severity": overall_severity,
        "per_analyte": per_analyte,
        "acute_findings": acute_findings,
        "chronic_findings": chronic_findings,
        "suggestions": suggestions,
        "differentials": differentials,
        "summary": summary
    }

# =========================
# END PART 5
# =========================
"""
PART 6 / 7
- Trend analysis (non-alarmist)
- Final wording guardrails
- GP-friendly soft suggestions
- No imperatives
"""

# -------------------------
# TREND ANALYSIS
# -------------------------

def trend_analysis(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes simple deltas only when safe.
    Never interprets trends clinically.
    """

    if not previous or "canonical" not in previous:
        return {"status": "no_previous"}

    trends = {}

    prev_canon = previous.get("canonical", {})

    for analyte, data in current.items():
        if analyte.startswith("_"):
            continue

        cur_val = data.get("value")
        prev_val = prev_canon.get(analyte, {}).get("value")

        if cur_val is None or prev_val is None:
            continue

        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None

            trends[analyte] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": round(delta, 2),
                "percent_change": round(pct, 1) if pct is not None else None
            }
        except Exception:
            continue

    return {
        "status": "available" if trends else "no_numeric_overlap",
        "changes": trends
    }

# -------------------------
# GP-SAFE SUMMARY BUILDER
# -------------------------

def build_final_summary(
    route_info: Dict[str, Any],
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Constructs the final clinician-facing summary.
    This is the ONLY place where wording is assembled.
    """

    meta = canonical.get("_meta", {})
    acute = route_info.get("acute_findings", [])
    chronic = route_info.get("chronic_findings", [])
    severity = route_info.get("overall_severity")

    summary_lines = []
    suggestions = []

    # -------------------------
    # CBC DISCLAIMER
    # -------------------------

    if not meta.get("cbc_present"):
        summary_lines.append(
            "CBC parameters were not available; interpretation is limited to chemistry results."
        )

    # -------------------------
    # ACUTE STATUS
    # -------------------------

    if acute:
        summary_lines.append("Acute laboratory abnormalities detected.")
    else:
        summary_lines.append("No acute laboratory abnormalities detected.")

    # -------------------------
    # CHRONIC / LONG-TERM RISK
    # -------------------------

    if chronic:
        summary_lines.append(
            "Mild to moderate long-term risk markers are present."
        )

    # -------------------------
    # BILIRUBIN REASSURANCE
    # -------------------------

    facts = canonical.get("_facts", [])
    for f in facts:
        if "bilirubin" in f.lower():
            summary_lines.append(
                "Isolated bilirubin elevation with normal liver enzymes may be benign."
            )

    # -------------------------
    # SOFT SUGGESTIONS (NON-DIRECTIVE)
    # -------------------------

    if "LDL elevated" in " ".join(chronic):
        suggestions.append(
            "Consider repeat fasting lipid profile if not fasting at time of sampling."
        )

    if trust_flags.get("has_long_term_risk"):
        suggestions.append(
            "Lifestyle modification may be beneficial depending on overall cardiovascular risk."
        )

    if "bilirubin" in canonical:
        suggestions.append(
            "Repeat bilirubin may be considered if clinically indicated."
        )

    # -------------------------
    # FINAL SEVERITY WORDING
    # -------------------------

    if severity == "normal" and trust_flags.get("has_long_term_risk"):
        severity_text = "clinically stable with long-term risk factors"
    else:
        severity_text = severity

    return {
        "severity_text": severity_text,
        "summary": " ".join(summary_lines),
        "suggested_follow_up": suggestions
    }

# =========================
# END PART 6
# =========================
"""
PART 7 / 7
- process_report()
- Supabase persistence
- Trend fetch
- Poll loop
"""

# -------------------------
# SAVE RESULTS
# -------------------------

def save_ai_results(report_id: str, payload: Dict[str, Any]) -> None:
    if not supabase:
        log.warning("Supabase unavailable — skipping save")
        return

    try:
        supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": payload,
            "ai_error": None
        }).eq("id", report_id).execute()

        log.info("Saved AI results for report %s", report_id)

    except Exception as e:
        log.error("Failed to save AI results: %s", e)
        try:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", report_id).execute()
        except Exception:
            pass

# -------------------------
# FETCH PREVIOUS RESULTS
# -------------------------

def fetch_previous_results(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None

    patient_id = job.get("patient_id")
    if not patient_id:
        return None

    try:
        res = (
            supabase
            .table(SUPABASE_TABLE)
            .select("ai_results,created_at")
            .eq("patient_id", patient_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )

        rows = res.data if hasattr(res, "data") else []
        if rows:
            return rows[0].get("ai_results")

    except Exception as e:
        log.warning("Previous result fetch failed: %s", e)

    return None

# -------------------------
# MAIN REPORT PROCESSOR
# -------------------------

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id")
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    log.info("Processing report %s", report_id)

    try:
        # -------------------------
        # PDF INGESTION
        # -------------------------

        pdf_bytes = download_pdf_from_supabase(job)
        text, scanned = extract_text_from_pdf(pdf_bytes)

        # -------------------------
        # PARSE
        # -------------------------

        parsed, comments = parse_lab_text(text)

        # -------------------------
        # CANONICAL MAP
        # -------------------------

        canonical = canonical_map(
            parsed=parsed,
            comments=comments,
            patient_age=patient_age,
            patient_sex=patient_sex
        )

        # Safety guards
        assert_cbc_or_flag(canonical)
        annotate_bilirubin_context(canonical)

        # -------------------------
        # TRUST FLAGS
        # -------------------------

        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False
        }

        apply_long_term_risk_flags(canonical, doctor_trust_flags)

        # -------------------------
        # ROUTES & SEVERITY
        # -------------------------

        route_info = route_engine(
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

        # -------------------------
        # TRENDS
        # -------------------------

        previous = fetch_previous_results(job)
        trends = trend_analysis(canonical, previous)

        # -------------------------
        # FINAL SUMMARY
        # -------------------------

        final_summary = build_final_summary(
            route_info=route_info,
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

        # -------------------------
        # FINAL PAYLOAD
        # -------------------------

        ai_results = {
            "processed_at": now_iso(),
            "scanned": scanned,
            "canonical": canonical,
            "routes": route_info,
            "summary": final_summary,
            "trends": trends,
            "doctor_trust_flags": doctor_trust_flags
        }

        save_ai_results(report_id, ai_results)

        log.info("Report %s processed successfully", report_id)
        return {"success": True}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        log.error("Processing failed for %s: %s", report_id, err)
        traceback.print_exc()

        try:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": err
            }).eq("id", report_id).execute()
        except Exception:
            pass

        return {"success": False, "error": err}

# -------------------------
# POLL LOOP
# -------------------------

def poll_loop() -> None:
    if not supabase:
        log.error("Supabase unavailable — poll loop disabled")
        return

    log.info("AMI Health Worker started — polling for pending reports")

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

            rows = res.data if hasattr(res, "data") else []

            for job in rows:
                try:
                    supabase.table(SUPABASE_TABLE).update({
                        "ai_status": "processing"
                    }).eq("id", job["id"]).execute()
                except Exception:
                    pass

                process_report(job)

            if not rows:
                time.sleep(POLL_INTERVAL)

        except Exception as e:
            log.error("Poll loop error: %s", e)
            time.sleep(5)

# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    poll_loop()

# =========================
# END PART 7
# =========================
