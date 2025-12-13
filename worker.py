#!/usr/bin/env python3
# ==========================================================
# AMI Health — Worker (Stable Clinical Edition)
# Purpose: Deterministic CBC extraction + clinical pattern synthesis
# ==========================================================

import os
import io
import re
import time
import json
import logging
from typing import Dict, Any, List, Optional

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
SUPABASE_TABLE = "reports"
SUPABASE_BUCKET = "reports"

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==========================================================
# UTILITIES
# ==========================================================
def clean_number(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None

def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""

def is_scanned(text: str) -> bool:
    return len(text.strip()) < 40

def ocr_pdf(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=250)
    text = ""
    for img in images:
        gray = img.convert("L")
        text += pytesseract.image_to_string(gray) + "\n"
    return text

# ==========================================================
# CBC PARSER (STRICT — NO HALLUCINATION)
# ==========================================================
CBC_PATTERNS = {
    "Hb": r"(Hb|Haemoglobin|Hemoglobin)\s*[:\-]?\s*([\d.,]+)",
    "MCV": r"(MCV)\s*[:\-]?\s*([\d.,]+)",
    "WBC": r"(WBC|White cell count)\s*[:\-]?\s*([\d.,]+)",
    "Neutrophils": r"(Neutrophils?)\s*[:\-]?\s*([\d.,]+)",
    "Platelets": r"(Platelets?)\s*[:\-]?\s*([\d.,]+)",
    "CRP": r"(CRP)\s*[:\-]?\s*([\d.,]+)",
    "CK": r"(CK)\s*[:\-]?\s*([\d.,]+)",
    "Creatinine": r"(Creatinine)\s*[:\-]?\s*([\d.,]+)",
    "Urea": r"(Urea)\s*[:\-]?\s*([\d.,]+)",
}

def parse_cbc(text: str) -> Dict[str, float]:
    values = {}
    for key, pattern in CBC_PATTERNS.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            values[key] = clean_number(m.group(2))
    return values

# ==========================================================
# CLINICAL INTELLIGENCE (DR-GRADE)
# ==========================================================
def generate_report(cbc: Dict[str, float]) -> Dict[str, Any]:
    patterns = []
    differentials = []
    actions = []

    Hb = cbc.get("Hb")
    MCV = cbc.get("MCV")
    WBC = cbc.get("WBC")
    Neut = cbc.get("Neutrophils")
    CRP = cbc.get("CRP")
    CK = cbc.get("CK")
    Cr = cbc.get("Creatinine")
    Urea = cbc.get("Urea")

    # ---- ANAEMIA ----
    if Hb and Hb < 12:
        patterns.append(f"anemia: Hb {Hb} g/dL")
        if MCV and MCV < 80:
            patterns.append(f"microcytic anemia: MCV {MCV} fL")
            differentials += [
                "Iron deficiency anemia",
                "Thalassaemia trait",
                "Chronic blood loss"
            ]
            actions.append("Order ferritin, reticulocyte count, and peripheral smear")
            actions.append("Assess menstrual history or occult blood loss")

    # ---- INFECTION / SEPSIS ----
    if WBC and WBC > 12 and Neut and Neut > 75:
        patterns.append(f"leukocytosis: WBC {WBC} x10^9/L")
        patterns.append(f"neutrophilic predominance: Neutrophils {Neut}%")
        if CRP and CRP > 10:
            patterns.append(f"elevated CRP: CRP {CRP} mg/L")
            differentials += ["Bacterial infection", "Sepsis"]
            actions.append("Clinical assessment for sepsis; consider blood cultures")
            actions.append("Consider IV fluids and empiric antibiotics if unstable")

    # ---- KIDNEY ----
    if Cr and Urea and Cr > 120:
        differentials.append("Acute kidney injury (prerenal?)")
        actions.append("Assess volume status and hydration")

    # ---- CK ----
    if CK and CK > 1000:
        patterns.append(f"rhabdomyolysis signal: CK {CK} U/L")
        actions.append("Assess muscle pain and urine colour")
        actions.append("Check renal function and electrolytes; consider urgent fluids")

    severity = "HIGH" if CRP and CRP > 40 else "MODERATE"

    return {
        "severity": severity,
        "patterns": patterns,
        "differential": list(dict.fromkeys(differentials)),
        "actions": list(dict.fromkeys(actions))
    }

# ==========================================================
# MAIN PROCESSOR
# ==========================================================
def process_report(row: Dict[str, Any]):
    report_id = row["id"]
    path = row["file_path"]

    log.info(f"Processing report {report_id}")

    pdf_resp = supabase.storage.from_(SUPABASE_BUCKET).download(path)
    pdf_bytes = pdf_resp if isinstance(pdf_resp, bytes) else pdf_resp

    text = extract_pdf_text(pdf_bytes)
    if is_scanned(text):
        text = ocr_pdf(pdf_bytes)

    cbc = parse_cbc(text)
    if not cbc:
        raise RuntimeError("No CBC extracted")

    report = generate_report(cbc)

    ai_results = {
        "severity": report["severity"],
        "patterns": report["patterns"],
        "top_differential_diagnoses": report["differential"],
        "immediate_suggested_actions": report["actions"]
    }

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": ai_results
    }).eq("id", report_id).execute()

# ==========================================================
# WORKER LOOP
# ==========================================================
def main():
    log.info("AMI Worker started")
    while True:
        res = supabase.table(SUPABASE_TABLE)\
            .select("*")\
            .eq("ai_status", "pending")\
            .limit(1)\
            .execute()

        jobs = res.data or []
        if not jobs:
            time.sleep(1)
            continue

        job = jobs[0]
        supabase.table(SUPABASE_TABLE).update(
            {"ai_status": "processing"}
        ).eq("id", job["id"]).execute()

        try:
            process_report(job)
        except Exception as e:
            log.error(str(e))
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", job["id"]).execute()

if __name__ == "__main__":
    main()
