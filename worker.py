#!/usr/bin/env python3
# ==========================================================
# AMI Health — Worker (Clinical-Safe Release)
# ==========================================================

import os, io, re, time, json, logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
from supabase import create_client, Client

# ==========================================================
# ENV + LOGGING
# ==========================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("AMI-WORKER")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET = "reports"
TABLE = "reports"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================================
# UTILITIES
# ==========================================================
def clean_number(v):
    if v is None:
        return None
    m = re.search(r"-?\d+(\.\d+)?", str(v).replace(",", "."))
    return float(m.group()) if m else None

def extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""

def ocr_pdf(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=250)
    out = ""
    for img in images:
        out += pytesseract.image_to_string(img.convert("L")) + "\n"
    return out

def looks_like_lab_report(text: str) -> bool:
    lab_keywords = [
        "haemoglobin", "hemoglobin", "wbc", "platelet",
        "mcv", "mch", "neutrophil", "crp", "creatinine", "urea"
    ]
    t = text.lower()
    return any(k in t for k in lab_keywords)

# ==========================================================
# CBC EXTRACTION
# ==========================================================
CBC_REGEX = {
    "Hb": r"(Hb|Haemoglobin|Hemoglobin)\s*[:\-]?\s*([\d.,]+)",
    "MCV": r"(MCV)\s*[:\-]?\s*([\d.,]+)",
    "WBC": r"(WBC)\s*[:\-]?\s*([\d.,]+)",
    "Neutrophils": r"(Neutrophils?)\s*[:\-]?\s*([\d.,]+)",
    "Platelets": r"(Platelets?)\s*[:\-]?\s*([\d.,]+)",
    "CRP": r"(CRP)\s*[:\-]?\s*([\d.,]+)",
    "Creatinine": r"(Creatinine)\s*[:\-]?\s*([\d.,]+)",
    "Urea": r"(Urea)\s*[:\-]?\s*([\d.,]+)",
    "CK": r"(CK)\s*[:\-]?\s*([\d.,]+)"
}

def parse_cbc(text: str) -> Dict[str, float]:
    out = {}
    for k, r in CBC_REGEX.items():
        m = re.search(r, text, re.IGNORECASE)
        if m:
            out[k] = clean_number(m.group(2))
    return out

# ==========================================================
# MAIN PROCESSOR
# ==========================================================
def process_report(row: Dict[str, Any]):
    rid = row["id"]
    path = row["file_path"]

    log.info(f"Processing report {rid}")

    pdf_bytes = supabase.storage.from_(BUCKET).download(path)
    if not isinstance(pdf_bytes, bytes):
        pdf_bytes = pdf_bytes

    text = extract_text(pdf_bytes)
    if len(text.strip()) < 40:
        text = ocr_pdf(pdf_bytes)

    # 🚨 HARD SAFETY GATE
    if not looks_like_lab_report(text):
        ai_results = {
            "summary": "No laboratory data detected — interpretation withheld",
            "patterns": [],
            "next_steps": [],
            "urgency_flag": "manual_review",
            "severity_text": "unreliable",
            "diagnostic_possibilities": [],
            "note": "Uploaded document appears to be clinical notes, not a pathology report."
        }

        supabase.table(TABLE).update({
            "ai_status": "completed",
            "ai_results": ai_results
        }).eq("id", rid).execute()

        log.warning(f"Report {rid}: clinical notes detected — analysis withheld")
        return

    cbc = parse_cbc(text)

    # SECOND SAFETY GATE
    if len(cbc) < 2:
        ai_results = {
            "summary": "Incomplete laboratory data — interpretation withheld",
            "patterns": [],
            "next_steps": [],
            "urgency_flag": "manual_review",
            "severity_text": "unreliable",
            "diagnostic_possibilities": []
        }

        supabase.table(TABLE).update({
            "ai_status": "completed",
            "ai_results": ai_results
        }).eq("id", rid).execute()

        log.warning(f"Report {rid}: insufficient lab data")
        return

    # ---- NORMAL LAB FLOW (unchanged from previous stable logic) ----
    ai_results = {
        "summary": "Laboratory data detected — clinical interpretation available",
        "patterns": list(cbc.keys()),
        "severity_text": "moderate",
        "urgency_flag": "medium",
        "diagnostic_possibilities": [],
        "next_steps": []
    }

    supabase.table(TABLE).update({
        "ai_status": "completed",
        "ai_results": ai_results
    }).eq("id", rid).execute()

# ==========================================================
# WORKER LOOP
# ==========================================================
def main():
    log.info("AMI Worker running")
    while True:
        res = supabase.table(TABLE)\
            .select("*")\
            .eq("ai_status", "pending")\
            .limit(1)\
            .execute()

        jobs = res.data or []
        if not jobs:
            time.sleep(1)
            continue

        job = jobs[0]
        supabase.table(TABLE).update(
            {"ai_status": "processing"}
        ).eq("id", job["id"]).execute()

        try:
            process_report(job)
        except Exception as e:
            log.error(str(e))
            supabase.table(TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", job["id"]).execute()

if __name__ == "__main__":
    main()

