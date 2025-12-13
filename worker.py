#!/usr/bin/env python3
"""
AMI Health — Worker v4 (STABLE BASELINE)
Pattern → Route → Next Steps

CRITICAL NOTES:
- Compatible with openai==0.28.0
- NO OpenAI v1 client
- NO breaking parser changes
- NO NORMAL fallback
"""

import os
import io
import re
import json
import time
import logging
from typing import Dict, Any, List

# ------------------------------
# ENV + LOGGING
# ------------------------------
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI-WORKER] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker")

# ------------------------------
# OPENAI (OLD SDK — CORRECT)
# ------------------------------
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

# ------------------------------
# PDF / OCR
# ------------------------------
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# ------------------------------
# SUPABASE
# ------------------------------
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------
# CONSTANTS
# ------------------------------
CBC_ANCHORS = {"hb", "wbc", "platelet", "platelets"}

# ------------------------------
# PDF LOADING
# ------------------------------
def load_pdf_bytes(path: str) -> bytes:
    res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
    return res

# ------------------------------
# TEXT EXTRACTION
# ------------------------------
def extract_text_digital(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += "\n" + t
    return text.strip()

def extract_text_ocr(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=300)
    text = ""
    for img in images:
        text += "\n" + pytesseract.image_to_string(img)
    return text.strip()

# ------------------------------
# BASIC LAB PARSER (SAFE)
# ------------------------------
def parse_labs(text: str) -> Dict[str, float]:
    labs = {}

    patterns = {
        "hb": r"\bHb\b.*?([\d\.]+)",
        "wbc": r"\bWBC\b.*?([\d\.]+)",
        "platelets": r"\bPLT\b.*?([\d\.]+)",
        "crp": r"\bCRP\b.*?([\d\.]+)",
        "mcv": r"\bMCV\b.*?([\d\.]+)",
        "neutrophils": r"\bNeutrophils?\b.*?([\d\.]+)"
    }

    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                labs[key] = float(m.group(1))
            except:
                pass

    return labs

# ------------------------------
# ROUTE ENGINE (SAFE, NO NORMAL FALLBACK)
# ------------------------------
def route_engine(labs: Dict[str, float]) -> Dict[str, Any]:

    found = set(labs.keys())
    has_cbc = CBC_ANCHORS.intersection(found)

    if not has_cbc:
        return {
            "severity_text": "unreliable",
            "urgency_flag": "manual_review",
            "summary": "Incomplete CBC data — interpretation withheld",
            "patterns": [],
            "diagnostic_possibilities": [],
            "next_steps": []
        }

    patterns = []
    dx = []
    steps = []

    if labs.get("wbc", 0) > 12:
        patterns.append("Leukocytosis")
        dx.append("Bacterial infection / sepsis — leukocytosis")
        steps.append("Clinical assessment for infection")

    if labs.get("crp", 0) > 20:
        patterns.append("Elevated CRP")
        dx.append("Inflammatory or infective process")
        steps.append("Consider blood cultures if febrile")

    if labs.get("hb", 99) < 12:
        patterns.append("Anemia")
        dx.append("Anemia — consider iron deficiency vs inflammation")
        steps.append("Order ferritin and reticulocyte count")

    severity = "mild"
    if labs.get("wbc", 0) > 15 and labs.get("crp", 0) > 40:
        severity = "severe"

    return {
        "severity_text": severity,
        "urgency_flag": "high" if severity == "severe" else "medium",
        "summary": "; ".join(patterns),
        "patterns": patterns,
        "diagnostic_possibilities": dx,
        "next_steps": steps
    }

# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    log.info("AMI Worker v4 starting — Pattern → Route → Next Steps")

    while True:
        rows = supabase.table(SUPABASE_TABLE)\
            .select("*")\
            .eq("ai_status", "pending")\
            .limit(1)\
            .execute().data

        if not rows:
            time.sleep(5)
            continue

        row = rows[0]
        report_id = row["id"]
        file_path = row["file_path"]

        log.info(f"Processing report {report_id}")

        try:
            pdf_bytes = load_pdf_bytes(file_path)

            text = extract_text_digital(pdf_bytes)
            if len(text) < 100:
                log.warning("Digital text insufficient — using OCR")
                text = extract_text_ocr(pdf_bytes)

            labs = parse_labs(text)
            ai = route_engine(labs)

            supabase.table(SUPABASE_TABLE)\
                .update({
                    "ai_results": ai,
                    "ai_status": "completed"
                })\
                .eq("id", report_id)\
                .execute()

            log.info(f"Report {report_id} completed")

        except Exception as e:
            log.exception(f"Failed report {report_id}")
            supabase.table(SUPABASE_TABLE)\
                .update({
                    "ai_status": "error",
                    "ai_error": str(e)
                })\
                .eq("id", report_id)\
                .execute()

        time.sleep(1)

# ------------------------------
if __name__ == "__main__":
    main()
