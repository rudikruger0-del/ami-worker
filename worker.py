#!/usr/bin/env python3
# AMI Health — Worker V8 FINAL (LOCKED)
# Stable OCR + deterministic clinical reasoning
# Matches UI + Doctor feedback exactly

import os
import io
import re
import time
import json
import logging
from typing import Dict, Any, List

from dotenv import load_dotenv
from supabase import create_client
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract

# ===============================
# ENV + LOGGING
# ===============================
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("AMI-WORKER")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET = "reports"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

log.info("AMI Worker V8 starting — OCR → Parse → Clinical Reasoning")

# ===============================
# UTILITIES
# ===============================
def clean_num(x):
    if x is None:
        return None
    s = str(x).replace(",", ".")
    m = re.search(r"-?\d+(\.\d+)?", s)
    return float(m.group()) if m else None

# ===============================
# OCR
# ===============================
def ocr_pdf(pdf_bytes: bytes) -> str:
    pages = convert_from_bytes(pdf_bytes, dpi=300)
    text = ""
    for img in pages:
        img = img.convert("L")
        text += pytesseract.image_to_string(img) + "\n"
    return text

# ===============================
# PARSER
# ===============================
ANALYTES = {
    "Hb": r"(hb|haemoglobin|hemoglobin)",
    "MCV": r"(mcv)",
    "MCH": r"(mch)",
    "WBC": r"(wbc|white)",
    "Neutrophils": r"(neut)",
    "Platelets": r"(platelet|plt)",
    "CRP": r"(crp)",
    "Creatinine": r"(creatinine)",
    "Urea": r"(urea)",
    "CO2": r"(co2|bicarbonate)",
    "CK": r"(ck[^-]|creatine kinase)",
}

def parse(text: str) -> Dict[str, float]:
    out = {}
    for line in text.splitlines():
        for key, rx in ANALYTES.items():
            if re.search(rx, line, re.I):
                val = clean_num(line)
                if val is not None:
                    out[key] = val
    return out

# ===============================
# DECORATION
# ===============================
RANGES = {
    "Hb": (12, 16),
    "MCV": (80, 100),
    "WBC": (4, 11),
    "Platelets": (150, 450),
    "CRP": (0, 5),
    "Creatinine": (45, 90),
    "CO2": (22, 29),
}

def decorate(vals):
    deco = {}
    for k, v in vals.items():
        low, high = RANGES.get(k, (None, None))
        if low and v < low:
            flag = "low"
        elif high and v > high:
            flag = "high"
        else:
            flag = "normal"
        deco[k] = {
            "value": v,
            "severity_text": flag
        }
    return deco

# ===============================
# CLINICAL ENGINE (DR STYLE)
# ===============================
def clinical(vals):
    patterns = []
    ddx = []
    steps = []
    severity = "normal"

    if vals.get("WBC", 0) > 12 and vals.get("Neutrophils", 0) > 70:
        patterns.append("Neutrophilic leukocytosis")
        ddx.append("Sepsis / bacterial infection")
        severity = "severe"

    if vals.get("CRP", 0) > 50:
        patterns.append("Marked inflammation")
        ddx.append("Acute inflammatory or infectious process")
        severity = "severe"

    if vals.get("Hb", 99) < 11:
        patterns.append("Anaemia")
        ddx.append("Iron deficiency / blood loss")

    if vals.get("Creatinine", 0) > 120 and vals.get("CO2", 99) < 20:
        patterns.append("Possible acute kidney injury")
        ddx.append("Pre-renal AKI — dehydration / sepsis")

    if severity == "severe":
        steps.extend([
            "Urgent clinical assessment",
            "Blood cultures if septic picture",
            "IV fluids and renal monitoring",
        ])

    return {
        "severity_text": severity,
        "summary": "Laboratory data detected — clinical interpretation available",
        "patterns": patterns,
        "differential": ddx,
        "next_steps": steps
    }

# ===============================
# PROCESS REPORT
# ===============================
def process(job):
    rid = job["id"]
    path = job["file_path"]

    log.info(f"Processing report {rid}")

    pdf = supabase.storage.from_(BUCKET).download(path)
    if isinstance(pdf, bytes):
        pdf_bytes = pdf
    else:
        pdf_bytes = pdf

    text = ocr_pdf(pdf_bytes)
    vals = parse(text)
    decorated = decorate(vals)
    routes = clinical(vals)

    ai_results = {
        "decorated": decorated,
        "routes": routes
    }

    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_results
    }).eq("id", rid).execute()

    log.info(f"Completed report {rid}")

# ===============================
# LOOP
# ===============================
while True:
    res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
    jobs = res.data or []
    if not jobs:
        time.sleep(1)
        continue

    job = jobs[0]
    supabase.table("reports").update({"ai_status": "processing"}).eq("id", job["id"]).execute()
    process(job)
