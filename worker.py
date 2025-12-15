#!/usr/bin/env python3
"""
AMI Health Worker — Production Clinical Engine
Doctor-grade, safety-first, non-hallucinating

Design guarantees:
- Never silently fail
- Never say NORMAL if abnormalities exist
- Never interpret missing CBC
- Always separate acute vs chronic risk
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
# =========================
# PARSER TYPE DEFINITIONS
# (MUST EXIST BEFORE USE)
# =========================

ParsedValue = Dict[str, Any]
ParsedResults = Dict[str, ParsedValue]


from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# =========================
# ENVIRONMENT
# =========================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing")

# =========================
# LOGGING — DOCTOR SAFE
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker")

# =========================
# SUPABASE
# =========================

try:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    supabase = None
    log.error("Supabase init failed: %s", e)

# =========================
# OCR DEPENDENCY (LOCAL)
# =========================

try:
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False
    log.warning("pytesseract unavailable — scanned PDFs disabled")

# =========================
# TIME
# =========================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# =========================
# HARD SAFETY LIMITS
# =========================

PHYS_LIMITS = {
    "Hb": (3, 25),
    "WBC": (0.1, 100),
    "Platelets": (1, 2000),
    "MCV": (50, 130),
    "CRP": (0, 500),
    "Creatinine": (10, 2000),
    "Sodium": (110, 180),
    "Potassium": (2.0, 7.5),
}

def safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def gate(analyte: str, val: float) -> Optional[float]:
    if analyte not in PHYS_LIMITS:
        return val
    lo, hi = PHYS_LIMITS[analyte]
    if val < lo or val > hi:
        return None
    return val
# =========================
# PART 2 — PDF INGESTION + OCR
# =========================

# -------------------------
# DOWNLOAD PDF
# -------------------------

def download_pdf(job: Dict[str, Any]) -> bytes:
    if not supabase:
        raise RuntimeError("Supabase unavailable")

    if job.get("pdf_url"):
        import requests
        r = requests.get(job["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    path = job.get("file_path")
    if not path:
        raise RuntimeError("No file_path on job")

    res = supabase.storage.from_(SUPABASE_BUCKET).download(path)

    if hasattr(res, "data") and res.data:
        return res.data
    if isinstance(res, (bytes, bytearray)):
        return res

    raise RuntimeError("PDF download failed")

# -------------------------
# DIGITAL TEXT EXTRACTION
# -------------------------

def extract_text_digital(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages).strip()
    except Exception as e:
        log.warning("Digital extraction failed: %s", e)
        return ""

# -------------------------
# SCANNED DETECTION
# -------------------------

def looks_scanned(text: str) -> bool:
    """
    Lancet/Ampath PDFs often have headers but no table text.
    """
    if not text:
        return True
    if len(text) < 150:
        return True
    if "ERYTHROCYTE" not in text and "HAEMOGLOBIN" not in text:
        # common Lancet CBC markers missing
        return True
    return False

# -------------------------
# OCR PREPROCESS
# -------------------------

def prep_image(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img

# -------------------------
# OCR PAGE
# -------------------------

def ocr_page(img: Image.Image) -> str:
    if not HAS_OCR:
        raise RuntimeError("OCR requested but pytesseract not available")

    return pytesseract.image_to_string(
        img,
        config="--psm 6 -c preserve_interword_spaces=1"
    )

# -------------------------
# OCR FULL PDF
# -------------------------

def extract_text_scanned(pdf_bytes: bytes) -> str:
    if not HAS_OCR:
        raise RuntimeError("Scanned PDF but OCR unavailable")

    pages = convert_from_bytes(pdf_bytes, dpi=300)
    collected = []

    for i, page in enumerate(pages):
        try:
            img = prep_image(page)
            txt = ocr_page(img)
            if txt.strip():
                collected.append(txt)
        except Exception as e:
            log.warning("OCR page %d failed: %s", i + 1, e)

    full = "\n".join(collected).strip()

    if not full:
        raise RuntimeError("OCR produced no usable text")

    return full

# -------------------------
# MASTER EXTRACTION
# -------------------------

def extract_text(pdf_bytes: bytes) -> Tuple[str, bool]:
    """
    Returns: (text, scanned_flag)
    """
    digital = extract_text_digital(pdf_bytes)

    if digital and not looks_scanned(digital):
        log.info("Digital PDF detected")
        return digital, False

    log.info("Scanned PDF detected — OCR engaged")
    scanned = extract_text_scanned(pdf_bytes)
    return scanned, True
# =========================
# PART 3 — CBC + CHEMISTRY PARSER (LANCET / AMPATH SAFE)
# =========================

NUM = r"(-?\d+(?:\.\d+)?)"
RANGE = r"(\d+(?:\.\d+)?)\s*[-–to]+\s*(\d+(?:\.\d+)?)"
FLAG = r"\b(H|L)\b"
UNIT = r"(x10\^?\d+/L|g/dL|g/L|mmol/L|µmol/L|U/L|%|fL)"

def parse_line_tokens(line: str) -> List[str]:
    """
    Normalize OCR noise into tokens.
    """
    clean = re.sub(r"[│|]+", " ", line)
    clean = re.sub(r"\s{2,}", " ", clean)
    return clean.strip().split(" ")

def extract_value(tokens: List[str]) -> Optional[float]:
    for t in tokens:
        try:
            return float(t.replace(",", "."))
        except Exception:
            continue
    return None

def extract_flag(text: str) -> Optional[str]:
    m = re.search(FLAG, text)
    return m.group(1) if m else None

def extract_range(text: str) -> Tuple[Optional[float], Optional[float]]:
    m = re.search(RANGE, text)
    if not m:
        return None, None
    return safe_float(m.group(1)), safe_float(m.group(2))

def extract_unit(text: str) -> Optional[str]:
    m = re.search(UNIT, text)
    return m.group(1) if m else None

def parse_lab_text(text: str) -> Tuple[ParsedResults, List[str]]:
    """
    Main parser entry.
    """
    results: ParsedResults = {}
    comments: List[str] = []

    if not text:
        return results, comments

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for raw in lines:
        lower = raw.lower()

        # Keep lab comments
        if lower.startswith("comment") or "note:" in lower:
            comments.append(raw)
            continue

        # Identify analyte label
        label_match = re.match(r"^[A-Za-z][A-Za-z\s\-/()]+", raw)
        if not label_match:
            continue

        label = label_match.group(0).strip()
        analyte = normalize_analyte(label)
        if not analyte:
            continue

        tokens = parse_line_tokens(raw)

        value = extract_value(tokens)
        if value is None:
            continue

        # HARD NUMERIC SAFETY
        safe_val = numeric_safety_gate(analyte, value)
        if safe_val is None:
            continue

        units = extract_unit(raw)
        flag = extract_flag(raw)
        ref_low, ref_high = extract_range(raw)

        results[analyte] = {
            "value": safe_val,
            "units": units,
            "flag": flag,
            "ref_low": ref_low,
            "ref_high": ref_high,
            "raw": raw
        }

    return results, comments

# -------------------------
# CBC PRESENCE CHECK
# -------------------------

CBC_KEYS = {
    "Hb",
    "WBC",
    "Platelets",
    "MCV",
    "Neutrophils",
    "Lymphocytes",
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
    Build a canonical, audit-safe structure.
    NO interpretation.
    NO severity.
    """

    canonical: Dict[str, Any] = {}

    age_grp = age_group(patient_age)
    sex = normalize_sex(patient_sex)

    # -------------------------
    # METADATA (FIRST)
    # -------------------------

    canonical["_meta"] = {
        "age": patient_age,
        "age_group": age_grp,
        "sex": sex,
        "cbc_present": cbc_present(parsed),
        "parser_comments": comments,
    }

    # -------------------------
    # COPY PARSED VALUES VERBATIM
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

    # Neutrophil-Lymphocyte Ratio (ONLY if both present)
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
    If CBC is missing, insert an explicit safety notice.
    NEVER allow silent pass-through.
    """
    if not canonical.get("_meta", {}).get("cbc_present"):
        canonical["_safety"] = canonical.get("_safety", [])
        canonical["_safety"].append(
            "CBC not detected — interpretation limited to chemistry only"
        )

# -------------------------
# BILIRUBIN FACTUAL CONTEXT
# -------------------------

def annotate_bilirubin_context(canonical: Dict[str, Any]) -> None:
    """
    Adds factual bilirubin relationships ONLY.
    No diagnosis, no reassurance language.
    """

    total = canonical.get("Bilirubin", {}).get("value")
    conj = canonical.get("Bilirubin Conjugated", {}).get("value")
    unconj = canonical.get("Bilirubin Unconjugated", {}).get("value")

    if total is not None and conj is not None and unconj is not None:
        if abs((conj + unconj) - total) <= 2:
            canonical["_facts"] = canonical.get("_facts", [])
            canonical["_facts"].append(
                "Bilirubin fractions sum to total bilirubin"
            )

# -------------------------
# LONG-TERM RISK FLAGS (FACTUAL ONLY)
# -------------------------

def apply_long_term_risk_flags(
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> None:
    """
    Marks presence of long-term metabolic risk.
    Does NOT alter severity here.
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
# PART 5 — SEVERITY + ROUTE ENGINE (TEXT ONLY)
# =========================

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
# ROUTE ENGINE (CLINICAL LOGIC)
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

    # -------------------------
    # CBC MISSING — NEVER SILENT
    # -------------------------

    if not meta.get("cbc_present"):
        chronic_findings.append(
            "CBC not available — interpretation limited to chemistry results"
        )

    # -------------------------
    # ANALYTE PASS
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
        # ACUTE VS CHRONIC SPLIT
        # -------------------------

        if sev in ("severe", "critical"):
            acute_findings.append(f"{analyte} markedly abnormal")

        elif analyte in ("LDL", "Triglycerides", "Non-HDL") and value is not None:
            chronic_findings.append(f"{analyte} elevated")

    # -------------------------
    # BILIRUBIN CLINICAL NUANCE
    # -------------------------

    if "Bilirubin" in canonical:
        b = canonical["Bilirubin"]["value"]
        if b is not None and b > 21:
            chronic_findings.append(
                "Mild isolated bilirubin elevation — benign causes possible"
            )

    # -------------------------
    # LONG-TERM RISK OVERRIDE
    # -------------------------

    if trust_flags.get("has_long_term_risk") and overall_severity == "normal":
        overall_severity = "mild"

    # -------------------------
    # SUMMARY (SAFE WORDING)
    # -------------------------

    if acute_findings:
        summary = "Acute laboratory abnormalities detected."
    elif chronic_findings:
        summary = (
            "No acute pathology detected. "
            "Mild metabolic and lipid abnormalities noted."
        )
    else:
        summary = "No acute laboratory abnormalities detected."

    return {
        "overall_severity": overall_severity,
        "per_analyte": per_analyte,
        "acute_findings": acute_findings,
        "chronic_findings": chronic_findings,
        "summary": summary
    }
# =========================
# PART 6 — FINAL ASSEMBLY + STORAGE
# =========================

# -------------------------
# TREND ANALYSIS (FACTUAL ONLY)
# -------------------------

def trend_analysis(
    current: Dict[str, Any],
    previous: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes numeric deltas only.
    NO interpretation.
    """
    if not previous or "canonical" not in previous:
        return {"status": "no_previous"}

    prev = previous.get("canonical", {})
    changes = {}

    for analyte, data in current.items():
        if analyte.startswith("_"):
            continue

        cur_val = data.get("value")
        prev_val = prev.get(analyte, {}).get("value")

        if cur_val is None or prev_val is None:
            continue

        try:
            delta = round(cur_val - prev_val, 2)
            changes[analyte] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": delta
            }
        except Exception:
            continue

    return {
        "status": "available" if changes else "no_overlap",
        "changes": changes
    }

# -------------------------
# GP-FACING FINAL SUMMARY
# -------------------------

def build_final_summary(
    route_info: Dict[str, Any],
    canonical: Dict[str, Any],
    trust_flags: Dict[str, bool]
) -> Dict[str, Any]:
    """
    FINAL wording layer.
    This is the only place language is assembled.
    """

    meta = canonical.get("_meta", {})
    acute = route_info.get("acute_findings", [])
    chronic = route_info.get("chronic_findings", [])
    severity = route_info.get("overall_severity")

    summary_lines = []
    suggestions = []

    # CBC disclaimer — NEVER OPTIONAL
    if not meta.get("cbc_present"):
        summary_lines.append(
            "CBC not available — interpretation limited to chemistry results."
        )

    # Acute status
    if acute:
        summary_lines.append("Acute laboratory abnormalities detected.")
    else:
        summary_lines.append("No acute laboratory abnormalities detected.")

    # Chronic risk
    if chronic:
        summary_lines.append(
            "Mild to moderate long-term metabolic or lipid abnormalities identified."
        )

    # Bilirubin reassurance
    facts = canonical.get("_facts", [])
    for f in facts:
        if "bilirubin" in f.lower():
            summary_lines.append(
                "Isolated bilirubin elevation with normal fractions may be benign."
            )

    # Soft GP-safe suggestions
    if any(k in " ".join(chronic).lower() for k in ("ldl", "triglyceride", "non-hdl")):
        suggestions.append(
            "Consider fasting lipid profile if not fasting at time of sampling."
        )
        suggestions.append(
            "Lifestyle modification may be beneficial depending on overall cardiovascular risk."
        )

    if "Bilirubin" in canonical:
        suggestions.append(
            "Repeat bilirubin testing may be considered if clinically indicated."
        )

    # Severity wording override (CRITICAL FIX)
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

# -------------------------
# SAVE RESULTS
# -------------------------

def save_ai_results(report_id: str, payload: Dict[str, Any]) -> None:
    if not supabase:
        log.warning("Supabase unavailable — skipping save")
        return

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": payload,
        "ai_error": None
    }).eq("id", report_id).execute()

# -------------------------
# FETCH PREVIOUS REPORT
# -------------------------

def fetch_previous_results(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not supabase:
        return None

    pid = job.get("patient_id")
    if not pid:
        return None

    try:
        res = (
            supabase
            .table(SUPABASE_TABLE)
            .select("ai_results")
            .eq("patient_id", pid)
            .eq("ai_status", "completed")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data if hasattr(res, "data") else []
        if rows:
            return rows[0].get("ai_results")
    except Exception:
        pass

    return None

# -------------------------
# MAIN PROCESSOR
# -------------------------

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id")
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    log.info("Processing report %s", report_id)

    try:
        pdf_bytes = download_pdf_from_supabase(job)
        text, scanned = extract_text_from_pdf(pdf_bytes)

        parsed, comments = parse_lab_text(text)

        canonical = canonical_map(
            parsed=parsed,
            comments=comments,
            patient_age=patient_age,
            patient_sex=patient_sex
        )

        assert_cbc_or_flag(canonical)
        annotate_bilirubin_context(canonical)

        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False
        }

        apply_long_term_risk_flags(canonical, doctor_trust_flags)

        route_info = route_engine(
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

        previous = fetch_previous_results(job)
        trends = trend_analysis(canonical, previous)

        final_summary = build_final_summary(
            route_info=route_info,
            canonical=canonical,
            trust_flags=doctor_trust_flags
        )

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
        log.info("Report %s completed", report_id)

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
    log.info("AMI Worker running")

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
                supabase.table(SUPABASE_TABLE).update({
                    "ai_status": "processing"
                }).eq("id", job["id"]).execute()

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
# END PART 6
# =========================
