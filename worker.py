#!/usr/bin/env python3
"""
AMI Health Worker — FULL CLINICAL ENGINE V7.2 (STABLE)

- Digital PDF parsing (chemistry tables)
- CBC column-table parsing (CRITICAL FIX)
- OCR fallback (pytesseract only)
- Route Engine V4 (ER-grade dominance)
- Built-in parser unit tests
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import io
import re
import sys
import time
import json
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# ======================================================
# ENV
# ======================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker")

# ======================================================
# SUPABASE (OPTIONAL FOR TEST MODE)
# ======================================================

supabase = None
if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ======================================================
# OCR
# ======================================================

try:
    import pytesseract
    HAS_OCR = True
except Exception:
    HAS_OCR = False

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    return img

def ocr_page(img: Image.Image) -> str:
    return pytesseract.image_to_string(
        img,
        config="--psm 6 -c preserve_interword_spaces=1"
    )

# ======================================================
# PDF EXTRACTION
# ======================================================

def extract_text_digital(pdf: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf))
        return "\n".join([p.extract_text() or "" for p in reader.pages]).strip()
    except Exception:
        return ""

def looks_scanned(text: str) -> bool:
    if not text or len(text) < 150:
        return True
    anchors = ("hb", "wbc", "platelet", "creatinine", "crp", "cholesterol")
    return not any(a in text.lower() for a in anchors)

def extract_text_scanned(pdf: bytes) -> str:
    if not HAS_OCR:
        raise RuntimeError("OCR required but pytesseract unavailable")

    pages = convert_from_bytes(pdf, dpi=300)
    out = []
    for p in pages:
        txt = ocr_page(preprocess_for_ocr(p))
        if txt.strip():
            out.append(txt)
    if not out:
        raise RuntimeError("OCR produced no text")
    return "\n".join(out)

def extract_text(pdf: bytes) -> Tuple[str, bool]:
    text = extract_text_digital(pdf)
    if text and not looks_scanned(text):
        return text, False
    return extract_text_scanned(pdf), True

# ======================================================
# ANALYTE NORMALISATION
# ======================================================

ANALYTE_MAP = {
    "hb": "Hb", "haemoglobin": "Hb", "hemoglobin": "Hb",
    "rbc": "RBC", "hct": "HCT",
    "wbc": "WBC", "white cell count": "WBC",
    "plt": "Platelets", "platelet": "Platelets",
    "crp": "CRP",
    "k": "Potassium", "potassium": "Potassium",
    "creat": "Creatinine", "creatinine": "Creatinine",
    "alt": "ALT", "ast": "AST",
    "ck": "CK", "ck-mb": "CK-MB",
    "albumin": "Albumin",
    "calcium": "Calcium", "ca": "Calcium",
    "ca adj": "Calcium_adj"
}

def normalize_label(label: str) -> Optional[str]:
    key = re.sub(r"[^a-z\- ]", "", label.lower()).strip()
    return ANALYTE_MAP.get(key)

# ======================================================
# PARSERS
# ======================================================

def parse_inline_tables(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    for ln in text.splitlines():
        ln = re.sub(r"\s{2,}", " ", ln.strip())
        m = re.match(r"^([A-Za-z\- ]+)\s+(\d+(?:\.\d+)?)\s*([HL])?$", ln)
        if not m:
            continue
        analyte = normalize_label(m.group(1))
        if analyte:
            results[analyte] = {
                "value": float(m.group(2)),
                "flag": "high" if m.group(3) == "H" else "low" if m.group(3) == "L" else None,
                "raw": ln
            }
    return results

def parse_chemistry_tables(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    for ln in text.splitlines():
        label_match = re.match(r"^[A-Za-z][A-Za-z \-/()]+", ln)
        if not label_match:
            continue
        analyte = normalize_label(label_match.group(0))
        if not analyte:
            continue
        nums = re.findall(r"\d+(?:\.\d+)?", ln.replace(",", "."))
        if not nums:
            continue
        results[analyte] = {
            "value": float(nums[-1]),
            "flag": "high" if " H" in ln else "low" if " L" in ln else None,
            "raw": ln
        }
    return results

def parse_cbc_column_table(text: str) -> Dict[str, Dict[str, Any]]:
    """
    FIX FOR YOUR EXACT FAILURE CASE
    TEST | RESULT | FLAG split across lines
    """
    results = {}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines) - 1:
        analyte = normalize_label(lines[i])
        if analyte:
            try:
                value = float(lines[i + 1])
            except:
                value = None
            flag = None
            if i + 2 < len(lines) and lines[i + 2] in ("H", "L"):
                flag = "high" if lines[i + 2] == "H" else "low"
            if value is not None:
                results[analyte] = {
                    "value": value,
                    "flag": flag,
                    "raw": f"{lines[i]} {value} {flag or ''}".strip()
                }
                i += 3
                continue
        i += 1
    return results

def parse_labs(text: str) -> Dict[str, Dict[str, Any]]:
    labs = {}
    labs.update(parse_chemistry_tables(text))
    labs.update(parse_cbc_column_table(text))
    labs.update(parse_inline_tables(text))
    return labs

# ======================================================
# ROUTE ENGINE V4 (ER DOMINANT)
# ======================================================

SEVERITY_ORDER = ["normal", "mild", "moderate", "severe", "critical"]

def max_severity(a, b):
    return SEVERITY_ORDER[max(SEVERITY_ORDER.index(a), SEVERITY_ORDER.index(b))]

def route_engine(labs: Dict[str, Any]) -> Dict[str, Any]:
    patterns, routes, steps, diffs = [], [], [], []
    severity, urgency = "normal", "low"

    def escalate(s, u=None):
        nonlocal severity, urgency
        severity = max_severity(severity, s)
        if u:
            urgency = max(urgency, u, key=lambda x: ["low", "moderate", "high"].index(x))

    v = lambda k: labs.get(k, {}).get("value")

    if v("CK") and v("CK") > 5000:
        patterns.append("Rhabdomyolysis physiology")
        routes.append("High-risk muscle injury")
        steps.append("Check renal function, urine myoglobin")
        escalate("critical", "high")

    if v("Platelets") and v("Platelets") < 50:
        patterns.append("Severe thrombocytopenia")
        routes.append("Bleeding risk")
        escalate("severe", "high")

    if v("Potassium") and v("Potassium") < 3.0:
        patterns.append("Hypokalaemia")
        routes.append("Arrhythmia risk")
        escalate("severe", "high")

    if v("CRP") and v("CRP") > 10:
        patterns.append("Inflammation")
        escalate("moderate", "moderate")

    if v("ALT") and v("ALT") > 200:
        patterns.append("Transaminitis")
        escalate("severe", "high")

    if not patterns:
        patterns.append("No acute abnormalities")

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": steps,
        "differentials": diffs,
        "severity_text": severity,
        "urgency": urgency
    }

# ======================================================
# WORKER CORE
# ======================================================

def process_job(job: Dict[str, Any]):
    pdf = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"]).data
    text, scanned = extract_text(pdf)
    labs = parse_labs(text)
    analysis = route_engine(labs)

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": {
            "scanned": scanned,
            "labs": labs,
            "analysis": analysis
        }
    }).eq("id", job["id"]).execute()

# ======================================================
# UNIT TESTS
# ======================================================

def run_tests():
    print("\n🧪 Running parser tests...\n")

    cbc_text = """
    HB
    12.5
    L
    PLT
    53
    L
    CRP
    20
    H
    CK
    14028
    H
    """

    labs = parse_labs(cbc_text)
    assert labs["Hb"]["value"] == 12.5
    assert labs["Platelets"]["value"] == 53
    assert labs["CK"]["value"] == 14028
    assert labs["CRP"]["flag"] == "high"

    print("✅ CBC column table test passed")

    chem_text = "Creatinine 88 H\nALT 226 H"
    labs = parse_labs(chem_text)
    assert labs["Creatinine"]["value"] == 88
    assert labs["ALT"]["value"] == 226

    print("✅ Chemistry table test passed")

    inline_text = "HB 10.2 L"
    labs = parse_labs(inline_text)
    assert labs["Hb"]["value"] == 10.2

    print("✅ Inline text test passed\n")
    print("🎉 ALL TESTS PASSED\n")

# ======================================================
# MAIN
# ======================================================

def main():
    if "--test" in sys.argv:
        run_tests()
        return

    if not supabase:
        raise RuntimeError("Supabase not configured")

    log.info("AMI Worker running")
    while True:
        res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending").limit(1).execute()
        jobs = res.data or []
        if not jobs:
            time.sleep(POLL_INTERVAL)
            continue

        job = jobs[0]
        supabase.table(SUPABASE_TABLE).update({"ai_status": "processing"}).eq("id", job["id"]).execute()
        try:
            process_job(job)
        except Exception as e:
            traceback.print_exc()
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", job["id"]).execute()

if __name__ == "__main__":
    main()
