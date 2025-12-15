#!/usr/bin/env python3
"""
AMI Health Worker — Clinical-Safe Deterministic Engine
SAFETY-CRITICAL SYSTEM

Design principles:
- Deterministic (no AI interpretation)
- Conservative wording
- Explicit limitations
- Age & sex aware
- CBC ≠ Chemistry (never blurred)
"""

# =========================
# Imports
# =========================

import os
import io
import re
import time
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# =========================
# Environment
# =========================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

# =========================
# Logging (doctor-safe)
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI] %(levelname)s: %(message)s"
)

log = logging.getLogger("ami-worker")

# =========================
# Supabase client
# =========================

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    log.error("Supabase init failed: %s", e)

# =========================
# OCR availability
# =========================

try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False
    log.warning("pytesseract not available — OCR disabled")

# =========================
# Time helpers
# =========================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# =========================
# Age & sex handling
# =========================

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

def normalize_sex(sex: Optional[str]) -> str:
    if not sex:
        return "unknown"
    s = sex.strip().lower()
    if s in ("m", "male"):
        return "male"
    if s in ("f", "female"):
        return "female"
    return "unknown"

# =========================
# Numeric safety
# =========================

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

def safe_float(val: Any) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None

def numeric_safety_gate(analyte: str, value: float) -> Optional[float]:
    if analyte not in PHYSIOLOGIC_LIMITS:
        return value
    low, high = PHYSIOLOGIC_LIMITS[analyte]
    if value < low or value > high:
        return None
    return value
# =========================
# PART 2 — PDF INGESTION & OCR
# =========================

MAX_OCR_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1400"))
OCR_JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "70"))

# -------------------------
# Download PDF
# -------------------------

def download_pdf_from_supabase(job: Dict[str, Any]) -> bytes:
    if job.get("pdf_url"):
        import requests
        r = requests.get(job["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if not supabase or not job.get("file_path"):
        raise RuntimeError("No PDF source available")

    res = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"])
    if hasattr(res, "data") and res.data:
        return res.data
    if isinstance(res, (bytes, bytearray)):
        return res

    raise RuntimeError("Empty PDF download")

# -------------------------
# Digital text extraction
# -------------------------

def extract_text_digital(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception:
        return ""

# -------------------------
# Scanned detection
# -------------------------

def is_scanned_pdf(pdf_bytes: bytes, min_chars: int = 100) -> bool:
    text = extract_text_digital(pdf_bytes)
    return len(text.strip()) < min_chars

# -------------------------
# OCR preprocessing
# -------------------------

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    w, h = img.size
    if w > MAX_OCR_WIDTH:
        new_h = int((MAX_OCR_WIDTH / w) * h)
        img = img.resize((MAX_OCR_WIDTH, new_h), Image.LANCZOS)
    return img

# -------------------------
# OCR page
# -------------------------

def ocr_image(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("OCR requested but pytesseract unavailable")
    return pytesseract.image_to_string(img, config="--psm 6")

# -------------------------
# OCR full PDF
# -------------------------

def extract_text_scanned(pdf_bytes: bytes) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("Scanned PDF but OCR disabled")

    pages = convert_from_bytes(pdf_bytes, dpi=300)
    all_text: List[str] = []

    for page in pages:
        try:
            img = preprocess_image_for_ocr(page)
            txt = ocr_image(img)
            if txt.strip():
                all_text.append(txt)
        except Exception:
            continue

    final = "\n".join(all_text).strip()
    if not final:
        raise RuntimeError("OCR produced no text")

    return final

# -------------------------
# Unified extraction
# -------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> Tuple[str, bool]:
    digital = extract_text_digital(pdf_bytes)
    if digital and len(digital) >= 100:
        return digital, False

    log.info("Scanned PDF detected — running OCR")
    return extract_text_scanned(pdf_bytes), True
# =========================
# PART 3 — PARSER (FACTS ONLY)
# =========================

ParsedValue = Dict[str, Any]
ParsedResults = Dict[str, ParsedValue]

# -------------------------
# Analyte normalization
# -------------------------

ANALYTE_SYNONYMS = {
    "hb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "wbc": "WBC",
    "white cell count": "WBC",
    "leukocytes": "WBC",
    "platelets": "Platelets",
    "plt": "Platelets",
    "mcv": "MCV",
    "rdw": "RDW",
    "neutrophils": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "crp": "CRP",
    "creatinine": "Creatinine",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "ck": "CK",
    "alt": "ALT",
    "ast": "AST",
    "bilirubin total": "Bilirubin",
    "bilirubin": "Bilirubin",
    "bilirubin conjugated": "Bilirubin Conjugated",
    "bilirubin unconjugated": "Bilirubin Unconjugated",
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

# -------------------------
# Regex blocks
# -------------------------

NUM = r"(-?\d+(?:\.\d+)?)"
RANGE = r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)"
UNIT = r"([a-zA-Z/%µ\^\-\d]+)"
FLAG = r"\b(H|L)\b"

# -------------------------
# Parse a single line
# -------------------------

def parse_result_line(line: str) -> Optional[Tuple[str, ParsedValue]]:
    raw = line.strip()
    if len(raw) < 4:
        return None

    label_match = re.match(r"^([A-Za-z\-\s/]+)", raw)
    if not label_match:
        return None

    label = label_match.group(1).strip()
    analyte = normalize_analyte(label)
    if not analyte:
        return None

    value_match = re.search(NUM, raw)
    if not value_match:
        return None

    value = safe_float(value_match.group(1))
    if value is None:
        return None

    unit_match = re.search(UNIT, raw[value_match.end():])
    units = unit_match.group(1) if unit_match else None

    flag_match = re.search(FLAG, raw)
    flag = flag_match.group(1) if flag_match else None

    ref_low = None
    ref_high = None
    range_match = re.search(RANGE, raw)
    if range_match:
        ref_low = safe_float(range_match.group(1))
        ref_high = safe_float(range_match.group(2))

    return analyte, {
        "value": value,
        "units": units,
        "flag": flag,
        "ref_low": ref_low,
        "ref_high": ref_high,
        "raw": raw
    }

# -------------------------
# Parse full text
# -------------------------

def parse_lab_text(text: str) -> Tuple[ParsedResults, List[str]]:
    results: ParsedResults = {}
    comments: List[str] = []

    if not text:
        return results, comments

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for line in lines:
        lower = line.lower()

        if lower.startswith("comment") or "suggested" in lower:
            comments.append(line)
            continue

        parsed = parse_result_line(line)
        if not parsed:
            continue

        analyte, data = parsed

        safe_val = numeric_safety_gate(analyte, data["value"])
        if safe_val is None:
            continue

        data["value"] = safe_val
        results[analyte] = data

    return results, comments

# -------------------------
# CBC presence detection
# -------------------------

CBC_KEYS = {
    "Hb", "WBC", "Platelets", "MCV", "RDW",
    "Neutrophils", "Lymphocytes"
}

def cbc_present(parsed: ParsedResults) -> bool:
    return any(k in parsed for k in CBC_KEYS)
# =========================
# PART 4 — CANONICAL MAP (FACTS ONLY)
# =========================

def canonical_map(
    parsed: ParsedResults,
    comments: List[str],
    patient_age: Optional[float],
    patient_sex: Optional[str]
) -> Dict[str, Any]:
    """
    Converts parsed lab values into a canonical structure.
    No severity, no routing, no interpretation.
    """

    canonical: Dict[str, Any] = {}

    age_grp = age_group(patient_age)
    sex = normalize_sex(patient_sex)

    # -------------------------
    # Metadata
    # -------------------------

    canonical["_meta"] = {
        "age": patient_age,
        "age_group": age_grp,
        "sex": sex,
        "cbc_present": cbc_present(parsed),
        "parser_comments": comments,
    }

    # -------------------------
    # Copy analytes verbatim
    # -------------------------

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

    # Neutrophil–Lymphocyte Ratio
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
                "raw": "Calculated NLR"
            }
    except Exception:
        pass

    return canonical

# -------------------------
# CBC availability guard
# -------------------------

def assert_cbc_or_flag(canonical: Dict[str, Any]) -> None:
    if not canonical.get("_meta", {}).get("cbc_present"):
        canonical["_safety"] = canonical.get("_safety", [])
        canonical["_safety"].append(
            "CBC not available — interpretation limited to chemistry only"
        )

# -------------------------
# Bilirubin factual context
# -------------------------

def annotate_bilirubin_context(canonical: Dict[str, Any]) -> None:
    total = canonical.get("Bilirubin", {}).get("value")
    conj = canonical.get("Bilirubin Conjugated", {}).get("value")
    unconj = canonical.get("Bilirubin Unconjugated", {}).get("value")

    if total is not None and conj is not None and unconj is not None:
        if abs((conj + unconj) - total) <= 2:
            canonical["_facts"] = canonical.get("_facts", [])
            canonical["_facts"].append(
                "Isolated bilirubin elevation with normal fractions"
            )

# -------------------------
# Long-term risk flags (FACT ONLY)
# -------------------------

def apply_long_term_risk_flags(
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> None:
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
# PART 5 — SEVERITY & ROUTING LOGIC
# =========================

SEVERITY_ORDER = ["normal", "mild", "moderate", "severe", "critical"]

def max_severity(a: str, b: str) -> str:
    return SEVERITY_ORDER[
        max(SEVERITY_ORDER.index(a), SEVERITY_ORDER.index(b))
    ]

# -------------------------
# Age / sex aware Hb limits
# -------------------------

def hb_lower_limit(age_group: str, sex: str) -> float:
    if age_group in ("infant", "child"):
        return 11.0
    if age_group == "teen":
        return 12.0 if sex == "female" else 13.0
    return 12.0 if sex == "female" else 13.0

# -------------------------
# Per-analyte severity
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
# Route engine
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

    # CBC missing guard
    if not meta.get("cbc_present"):
        chronic_findings.append(
            "CBC unavailable — chemistry-only interpretation"
        )

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

        if sev in ("severe", "critical"):
            acute_findings.append(f"{analyte} markedly abnormal")

        if analyte in ("LDL", "Triglycerides", "Non-HDL") and value is not None:
            chronic_findings.append(f"{analyte} elevated")

    # Bilirubin nuance (chronic, non-acute)
    if "Bilirubin" in canonical:
        b = canonical["Bilirubin"]["value"]
        if b is not None and b > 21:
            chronic_findings.append(
                "Mild isolated bilirubin elevation — benign causes possible"
            )

    # Trust-based downgrade
    if trust_flags.get("has_long_term_risk") and overall_severity == "normal":
        overall_severity = "mild"

    return {
        "overall_severity": overall_severity,
        "per_analyte": per_analyte,
        "acute_findings": acute_findings,
        "chronic_findings": chronic_findings,
    }
# =========================
# PART 6 — TRENDS + FINAL SUMMARY
# =========================

# -------------------------
# Trend analysis (numeric only)
# -------------------------

def trend_analysis(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compares current vs previous canonical values.
    FACTUAL deltas only.
    No clinical interpretation.
    """

    if not previous or "canonical" not in previous:
        return {
            "status": "no_previous",
            "changes": {}
        }

    prev_canon = previous.get("canonical", {})
    changes = {}

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

            changes[analyte] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": round(delta, 2),
                "percent_change": round(pct, 1) if pct is not None else None
            }
        except Exception:
            continue

    return {
        "status": "available" if changes else "no_numeric_overlap",
        "changes": changes
    }

# -------------------------
# Doctor-safe final summary
# -------------------------

def build_final_summary(
    route_info: Dict[str, Any],
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Final GP-facing wording.
    This is the ONLY place wording is generated.
    """

    meta = canonical.get("_meta", {})
    acute = route_info.get("acute_findings", [])
    chronic = route_info.get("chronic_findings", [])
    severity = route_info.get("overall_severity")

    summary_lines = []
    suggestions = []

    # -------------------------
    # CBC availability guard
    # -------------------------

    if not meta.get("cbc_present"):
        summary_lines.append(
            "CBC parameters were not available; interpretation is limited to chemistry results."
        )

    # -------------------------
    # Acute status
    # -------------------------

    if acute:
        summary_lines.append(
            "Acute laboratory abnormalities detected."
        )
    else:
        summary_lines.append(
            "No acute laboratory abnormalities detected."
        )

    # -------------------------
    # Chronic / long-term risk
    # -------------------------

    if chronic:
        summary_lines.append(
            "Mild to moderate long-term or metabolic abnormalities are present."
        )

    # -------------------------
    # Bilirubin nuance
    # -------------------------

    facts = canonical.get("_facts", [])
    for f in facts:
        if "bilirubin" in f.lower():
            summary_lines.append(
                "Isolated bilirubin elevation with normal liver enzymes may be benign."
            )

    # -------------------------
    # Soft suggestions (NON-DIRECTIVE)
    # -------------------------

    if any("LDL" in c or "Triglycerides" in c for c in chronic):
        suggestions.append(
            "Consider repeat fasting lipid profile if the sample was non-fasting."
        )

    if trust_flags.get("has_long_term_risk"):
        suggestions.append(
            "Lifestyle modification may be beneficial depending on overall cardiovascular risk."
        )

    if "Bilirubin" in canonical:
        suggestions.append(
            "Repeat bilirubin testing may be considered if clinically indicated."
        )

    # -------------------------
    # Final severity wording
    # -------------------------

    if severity == "normal" and trust_flags.get("has_long_term_risk"):
        severity_text = "clinically stable with long-term risk factors"
    elif severity == "normal":
        severity_text = "no acute abnormalities detected"
    else:
        severity_text = severity

    return {
        "severity_text": severity_text,
        "summary": " ".join(summary_lines),
        "suggested_follow_up": suggestions
    }
# =========================
# PART 7 — EXECUTION PIPELINE
# =========================

# -------------------------
# Save AI results
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
# Fetch previous completed report
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
            .eq("ai_status", "completed")
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
# Main report processor
# -------------------------

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id")
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    log.info("Processing report %s", report_id)

    try:
        # -------------------------
        # PDF ingestion
        # -------------------------

        pdf_bytes = download_pdf_from_supabase(job)
        text, scanned = extract_text_from_pdf(pdf_bytes)

        # -------------------------
        # Parse lab text
        # -------------------------

        parsed, comments = parse_lab_text(text)

        # -------------------------
        # Canonical mapping
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
        # Trust flags
        # -------------------------

        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False
        }

        apply_long_term_risk_flags(canonical, doctor_trust_flags)

        # -------------------------
        # Route engine
        # -------------------------

        route_info = route_engine(
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

        # -------------------------
        # Trend analysis
        # -------------------------

        previous = fetch_previous_results(job)
        trends = trend_analysis(canonical, previous)

        # -------------------------
        # Final summary
        # -------------------------

        final_summary = build_final_summary(
            route_info=route_info,
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

        # -------------------------
        # Final payload
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
# Poll loop
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
                .limit(3)
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
# Entry point
# -------------------------

if __name__ == "__main__":
    poll_loop()
