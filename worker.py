#!/usr/bin/env python3
"""
AMI Health Worker — Production Medical AI Pipeline
Safety-critical. Zero hallucinations. Doctor-trust first.
"""

import os
import io
import re
import json
import time
import traceback
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# =========================
# ENV + CLIENTS
# =========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

from supabase import create_client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# HARD SAFETY HELPERS
# =========================

def safe_float(v) -> Optional[float]:
    try:
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Physiological hard limits — NEVER GUESS
PHYS_LIMITS = {
    "Hb": (3, 25),
    "WBC": (0.1, 100),
    "Platelets": (1, 2000),
    "CRP": (0, 500),
    "Creatinine": (10, 2000),
    "ALT": (0, 5000),
    "AST": (0, 5000),
    "CK": (0, 200000),
    "Potassium": (2.0, 7.5),
    "Sodium": (110, 180),
}


def numeric_safety_gate(key: str, val: float) -> Optional[float]:
    if key not in PHYS_LIMITS:
        return val
    lo, hi = PHYS_LIMITS[key]
    if val < lo or val > hi:
        return None
    return val
# =========================
# PDF INGESTION
# =========================

def download_pdf_from_supabase(record: Dict[str, Any]) -> bytes:
    """
    Download PDF bytes from Supabase storage or pdf_url.
    """
    if record.get("pdf_url"):
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if not record.get("file_path"):
        raise RuntimeError("Missing file_path and pdf_url")

    res = supabase.storage.from_(SUPABASE_BUCKET).download(record["file_path"])

    if hasattr(res, "data") and res.data:
        return res.data

    if isinstance(res, (bytes, bytearray)):
        return res

    raise RuntimeError("Supabase PDF download failed")


# =========================
# DIGITAL VS SCANNED
# =========================

def extract_text_digital(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n".join(pages).strip()
    except Exception:
        return ""


def is_scanned_pdf(pdf_bytes: bytes, min_chars: int = 80) -> bool:
    text = extract_text_digital(pdf_bytes)
    return not text or len(text) < min_chars


# =========================
# OCR (PYTESSERACT ONLY)
# =========================

try:
    import pytesseract
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")  # grayscale
    w, h = img.size
    if w > 1600:
        img = img.resize((1600, int(h * 1600 / w)))
    return img


def ocr_pdf_pages(pdf_bytes: bytes) -> str:
    if not HAS_TESSERACT:
        raise RuntimeError("pytesseract not available")

    pages = convert_from_bytes(pdf_bytes, dpi=200)
    text_blocks: List[str] = []

    for p in pages:
        try:
            img = preprocess_for_ocr(p)
            t = pytesseract.image_to_string(img)
            if t:
                text_blocks.append(t)
        except Exception:
            continue

    return "\n".join(text_blocks).strip()


# =========================
# TEXT EXTRACTION ENTRYPOINT
# =========================

def extract_report_text(pdf_bytes: bytes) -> Dict[str, Any]:
    """
    Returns:
    {
      "text": extracted_text,
      "scanned": bool
    }
    """
    if is_scanned_pdf(pdf_bytes):
        text = ocr_pdf_pages(pdf_bytes)
        return {"text": text, "scanned": True}

    text = extract_text_digital(pdf_bytes)
    return {"text": text, "scanned": False}
# =========================
# PARSER — TABLE + INLINE
# =========================

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# -------------------------
# ANALYTE ALIASES
# -------------------------
ANALYTE_ALIASES = {
    # CBC
    "hb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "rbc": "RBC",
    "hct": "HCT",
    "haematocrit": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "plt": "Platelets",
    "platelets": "Platelets",
    "wbc": "WBC",
    "neut": "Neutrophils",
    "neutrophils": "Neutrophils",
    "lymph": "Lymphocytes",
    "lymphocytes": "Lymphocytes",
    "mono": "Monocytes",
    "eos": "Eosinophils",
    "baso": "Basophils",

    # Chemistry
    "crp": "CRP",
    "creat": "Creatinine",
    "creatinine": "Creatinine",
    "egfr": "eGFR",
    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ck-mb": "CK-MB",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "albumin": "Albumin",
    "bilirubin": "Bilirubin",
    "bilirubin total": "Bilirubin Total",
    "bilirubin conjugated": "Bilirubin Conjugated",
    "bilirubin unconjugated": "Bilirubin Unconjugated",
    "cholesterol": "Cholesterol",
    "ldl": "LDL",
    "hdl": "HDL",
    "triglycerides": "Triglycerides",
    "non-hdl": "Non-HDL",
}


def normalize_analyte(label: str) -> Optional[str]:
    if not label:
        return None
    key = _norm(label)
    key = re.sub(r"[^a-z0-9\-\s]", "", key)
    return ANALYTE_ALIASES.get(key)


# -------------------------
# TABLE ROW PARSER
# -------------------------
def parse_table_rows(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Matches rows like:
      HB 12.5 L
      PLT 53 L
      CRP 20 H
      CK 14028 H
      ALT 226 H
    """
    results: Dict[str, Dict[str, Any]] = {}

    lines = [_norm(l) for l in text.splitlines() if l.strip()]

    for line in lines:
        m = re.match(
            r"^([a-zA-Z\-\s\/]+)\s+(-?\d+(?:\.\d+)?)\s*([hHlL]{1,2})?$",
            line
        )
        if not m:
            continue

        raw_label, raw_val, flag = m.groups()
        analyte = normalize_analyte(raw_label)
        val = safe_float(raw_val)

        if not analyte or val is None:
            continue

        results[analyte] = {
            "value": val,
            "flag": flag.upper() if flag else None,
            "raw": line,
        }

    return results


# -------------------------
# INLINE / SENTENCE PARSER
# -------------------------
def parse_inline_values(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Catches:
      Creatinine 81 umol/L
      CRP < 5 mg/L
      LDL Chol 3.5 H
    """
    results: Dict[str, Dict[str, Any]] = {}

    pattern = re.compile(
        r"([A-Za-z\-\s\/]+)\s*[:=]?\s*(<|>)?\s*(-?\d+(?:\.\d+)?)\s*([a-zA-Z\/µ%]+)?\s*([hHlL])?",
        re.IGNORECASE
    )

    for m in pattern.finditer(text):
        raw_label, comp, raw_val, unit, flag = m.groups()
        analyte = normalize_analyte(raw_label)
        val = safe_float(raw_val)

        if not analyte or val is None:
            continue

        results.setdefault(analyte, {})
        results[analyte].update({
            "value": val,
            "unit": unit,
            "flag": flag.upper() if flag else None,
            "raw": m.group(0),
        })

    return results


# -------------------------
# MASTER PARSER
# -------------------------
def parse_report_text(text: str) -> Dict[str, Dict[str, Any]]:
    """
    1. Table rows FIRST (highest accuracy)
    2. Inline fallback
    """
    parsed: Dict[str, Dict[str, Any]] = {}

    table_hits = parse_table_rows(text)
    parsed.update(table_hits)

    inline_hits = parse_inline_values(text)
    for k, v in inline_hits.items():
        parsed.setdefault(k, v)

    if not parsed:
        raise RuntimeError("Parser failure: no analytes detected")

    return parsed
# =========================
# CANONICAL MAP (HARD SAFETY)
# =========================

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Pure normalization layer.
    - No clinical logic
    - No severity logic
    - No early returns
    """

    canonical: Dict[str, Dict[str, Any]] = {}

    for key, item in parsed.items():
        raw_val = item.get("value")
        val = safe_float(raw_val)
        if val is None:
            continue

        safe_val = numeric_safety_gate(key, val)
        if safe_val is None:
            continue

        canonical[key] = {
            "value": safe_val,
            "unit": item.get("unit"),
            "flag": item.get("flag"),
            "raw": item.get("raw"),
        }

    # -------------------------
    # SAFE DERIVED VALUES
    # -------------------------

    try:
        neut = canonical.get("Neutrophils", {}).get("value")
        lymph = canonical.get("Lymphocytes", {}).get("value")
        if neut is not None and lymph is not None and lymph > 0:
            canonical["NLR"] = {
                "value": round(neut / lymph, 2),
                "unit": None,
                "flag": None,
                "raw": "derived NLR",
            }
    except Exception:
        pass

    # -------------------------
    # HARD FAIL IF EMPTY
    # -------------------------
    if not canonical:
        raise RuntimeError("Canonical map empty after safety filtering")

    return canonical


# =========================
# eGFR (CKD-EPI 2021)
# =========================

def calculate_egfr(
    creatinine_umol: float,
    age: float,
    sex: str
) -> Optional[float]:
    """
    CKD-EPI 2021, race-free.
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
# =========================
# ROUTE ENGINE
# =========================

SEVERITY_MAP = {
    1: "normal",
    2: "mild",
    3: "moderate",
    4: "severe",
    5: "critical",
}


def score_severity(analyte: str, value: Optional[float], flag: Optional[str]) -> int:
    """
    Conservative per-analyte severity.
    Flags (H/L) ALWAYS up-weight severity.
    """

    if value is None:
        return 1

    a = analyte.lower()

    try:
        # -------------------------
        # H/L FLAG OVERRIDE
        # -------------------------
        if flag == "L":
            base = 2
        elif flag == "H":
            base = 2
        else:
            base = 1

        # -------------------------
        # CRITICAL DOMAINS
        # -------------------------
        if a == "ck":
            if value >= 10000:
                return 5
            if value >= 5000:
                return 4
            if value >= 2000:
                return 3
            return max(base, 2)

        if a == "platelets":
            if value < 50:
                return 5
            if value < 100:
                return 4
            if value < 150:
                return 3
            return base

        if a == "crp":
            if value >= 100:
                return 5
            if value >= 50:
                return 4
            if value >= 10:
                return 3
            return base

        if a == "potassium":
            if value < 3.0 or value > 6.0:
                return 5
            if value < 3.5 or value > 5.5:
                return 3
            return base

        if a == "sodium":
            if value < 120 or value > 160:
                return 5
            if value < 125 or value > 155:
                return 3
            return base

        if a == "creatinine":
            if value >= 350:
                return 5
            if value >= 200:
                return 4
            if value >= 120:
                return 3
            return base

        if a == "hb":
            if value < 8:
                return 5
            if value < 10:
                return 4
            if value < 12:
                return 3
            return base

        # -------------------------
        # DEFAULT
        # -------------------------
        return base

    except Exception:
        return 1


def route_engine_all(
    canonical: Dict[str, Dict[str, Any]],
    patient_meta: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    doctor_trust_flags: Dict[str, bool],
) -> Dict[str, Any]:

    patterns: List[Dict[str, str]] = []
    routes: List[str] = []
    next_steps: List[str] = []
    differentials: List[str] = []
    per_analyte: Dict[str, Any] = {}

    severity_scores: List[int] = []

    # -------------------------
    # PER-ANALYTE SCORING
    # -------------------------
    for analyte, data in canonical.items():
        val = data.get("value")
        flag = data.get("flag")

        sev = score_severity(analyte, val, flag)
        severity_scores.append(sev)

        per_analyte[analyte] = {
            "value": val,
            "unit": data.get("unit"),
            "flag": flag,
            "severity": SEVERITY_MAP[sev],
        }

    # -------------------------
    # DOMINANT THREAT OVERRIDES
    # -------------------------
    ck = canonical.get("CK", {}).get("value")
    if ck is not None and ck >= 5000:
        patterns.insert(0, {
            "pattern": "rhabdomyolysis physiology",
            "reason": f"CK {ck}"
        })
        routes.insert(0, "Rhabdomyolysis route")
        next_steps.append(
            "Marked CK elevation suggests muscle injury or rhabdomyolysis; monitor renal function and electrolytes closely."
        )
        severity_scores.append(5)

    plate = canonical.get("Platelets", {}).get("value")
    if plate is not None and plate < 100:
        patterns.insert(0, {
            "pattern": "thrombocytopenia",
            "reason": f"Platelets {plate}"
        })
        routes.insert(0, "Thrombocytopenia route")
        next_steps.append(
            "Low platelet count warrants repeat testing and evaluation for bleeding or marrow suppression."
        )
        severity_scores.append(4)

    crp = canonical.get("CRP", {}).get("value")
    if crp is not None and crp >= 10:
        patterns.append({
            "pattern": "inflammatory response",
            "reason": f"CRP {crp}"
        })
        next_steps.append(
            "Elevated CRP suggests active inflammation or infection; correlate with clinical findings."
        )

    # -------------------------
    # OVERALL SEVERITY
    # -------------------------
    overall_numeric = max(severity_scores) if severity_scores else 1

    # Doctor-trust override (long-term risk must not appear normal)
    if doctor_trust_flags.get("has_long_term_risk") and overall_numeric == 1:
        overall_numeric = 2

    severity_text = SEVERITY_MAP[overall_numeric]

    # -------------------------
    # SUMMARY
    # -------------------------
    summary_lines: List[str] = []

    if patterns:
        summary_lines.append(
            "Patterns: " + "; ".join(p["pattern"] for p in patterns)
        )

    if routes:
        summary_lines.append(
            "Primary routes: " + "; ".join(routes)
        )

    if differentials:
        summary_lines.append(
            "Differentials: " + ", ".join(differentials[:5])
        )

    if next_steps:
        summary_lines.append(
            "Suggested follow-up: " + " | ".join(next_steps[:5])
        )

    if not summary_lines:
        summary_lines.append(
            "No acute abnormalities detected. Long-term risk assessment recommended."
        )

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differentials": differentials,
        "per_analyte": per_analyte,
        "overall_severity": overall_numeric,
        "severity_text": severity_text,
        "summary": "\n".join(summary_lines),
    }
# =========================
# PROCESS REPORT
# =========================

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id") or job.get("report_id")
    patient_id = job.get("patient_id")
    patient_age = job.get("age")
    patient_sex = job.get("sex")

    if not report_id:
        raise RuntimeError("Missing report id")

    print(f"Processing report {report_id}")

    try:
        # -------------------------
        # DOWNLOAD + EXTRACT
        # -------------------------
        pdf_bytes = download_pdf_from_supabase(job)
        extracted = extract_report_text(pdf_bytes)

        raw_text = extracted["text"]
        scanned = extracted["scanned"]

        if not raw_text:
            raise RuntimeError("No text extracted from PDF")

        # -------------------------
        # PARSE → CANONICAL
        # -------------------------
        parsed = parse_report_text(raw_text)
        canonical = canonical_map(parsed)

        # -------------------------
        # DOCTOR TRUST FLAGS
        # -------------------------
        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False,
        }

        # Long-term cardiovascular risk guardrails
        ldl = canonical.get("LDL", {}).get("value")
        tg = canonical.get("Triglycerides", {}).get("value")
        non_hdl = canonical.get("Non-HDL", {}).get("value")

        if ldl is not None and ldl >= 3.0:
            doctor_trust_flags["has_long_term_risk"] = True
        if tg is not None and tg >= 1.7:
            doctor_trust_flags["has_long_term_risk"] = True
        if non_hdl is not None and non_hdl >= 3.4:
            doctor_trust_flags["has_long_term_risk"] = True

        # -------------------------
        # eGFR (SAFE INSERT)
        # -------------------------
        creat = canonical.get("Creatinine", {}).get("value")
        if creat is not None and patient_age:
            egfr = calculate_egfr(creat, patient_age, patient_sex)
            if egfr is not None:
                canonical["eGFR"] = {
                    "value": egfr,
                    "unit": "mL/min/1.73m²",
                    "flag": "L" if egfr < 60 else None,
                    "raw": "calculated (CKD-EPI 2021)",
                }

        # -------------------------
        # FETCH PREVIOUS RESULTS
        # -------------------------
        previous = None
        if patient_id:
            try:
                prev_q = (
                    supabase
                    .table(SUPABASE_TABLE)
                    .select("ai_results,created_at")
                    .eq("patient_id", patient_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                rows = prev_q.data if hasattr(prev_q, "data") else prev_q
                if rows:
                    previous = rows[0].get("ai_results")
            except Exception:
                previous = None

        # -------------------------
        # ROUTE ENGINE
        # -------------------------
        route_info = route_engine_all(
            canonical=canonical,
            patient_meta={"age": patient_age, "sex": patient_sex},
            previous=previous,
            doctor_trust_flags=doctor_trust_flags,
        )

        # -------------------------
        # FINAL PAYLOAD
        # -------------------------
        ai_results = {
            "processed_at": now_iso(),
            "scanned": scanned,
            "canonical": canonical,
            "routes": route_info,
            "doctor_trust_flags": doctor_trust_flags,
        }

        save_ai_results_to_supabase(report_id, ai_results)

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
# =========================
# SAVE RESULTS TO SUPABASE
# =========================

def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    try:
        payload = {
            "ai_status": "completed",
            "ai_results": ai_results,
            "ai_error": None,
        }
        supabase.table(SUPABASE_TABLE).update(payload).eq("id", report_id).execute()
        print(f"Saved ai_results for report {report_id}")
    except Exception as e:
        print("⚠️ Failed to save ai_results:", e)
        try:
            supabase.table(SUPABASE_TABLE).update(
                {"ai_status": "failed", "ai_error": str(e)}
            ).eq("id", report_id).execute()
        except Exception:
            pass


# =========================
# POLL LOOP
# =========================

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))


def poll_loop() -> None:
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

            if not rows:
                time.sleep(POLL_INTERVAL)
                continue

            for job in rows:
                report_id = job.get("id")
                if not report_id:
                    continue

                try:
                    supabase.table(SUPABASE_TABLE).update(
                        {"ai_status": "processing"}
                    ).eq("id", report_id).execute()
                except Exception:
                    pass

                process_report(job)

        except Exception as e:
            print("Polling error:", e)
            traceback.print_exc()
            time.sleep(5)


# =========================
# CLI / LOCAL TEST HARNESS
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pdf", help="Path to local PDF for testing")
    parser.add_argument("--once", action="store_true", help="Process pending once and exit")
    args = parser.parse_args()

    if args.test_pdf:
        print("🧪 Running local PDF test")

        with open(args.test_pdf, "rb") as f:
            test_bytes = f.read()

        # Override downloader for local test
        def _local_download(_):
            return test_bytes

        globals()["download_pdf_from_supabase"] = _local_download

        dummy_job = {
            "id": "local-test",
            "patient_id": "local-test",
            "file_path": "local.pdf",
            "age": 45,
            "sex": "male",
        }

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
                for r in rows:
                    process_report(r)
            else:
                print("No pending reports.")
        else:
            poll_loop()
