#!/usr/bin/env python3
# ============================================================
# AMI Health — Worker v4 (RESTORED & FIXED)
# Compatible with openai==0.28.0
# ============================================================

print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

import os
import io
import re
import time
import json
import datetime
import traceback
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from supabase import create_client
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import openai

# ============================================================
# ENV
# ============================================================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "reports"

# ============================================================
# HELPERS
# ============================================================

def now():
    return datetime.datetime.utcnow().isoformat() + "Z"

def clean_number(x):
    if x is None:
        return None
    s = str(x).replace(",", ".")
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None

# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_text(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def is_scanned(text: str) -> bool:
    return len(text.strip()) < 40

def ocr_pdf(pdf_bytes: bytes) -> str:
    images = convert_from_bytes(pdf_bytes, dpi=200)
    text = []
    for img in images:
        g = img.convert("L")
        g = g.resize((g.width * 2, g.height * 2))
        text.append(pytesseract.image_to_string(g))
    return "\n".join(text)

# ============================================================
# CBC PARSER (RESTORED STYLE)
# ============================================================

CBC_MAP = {
    "hb": "Hb",
    "hemoglobin": "Hb",
    "haemoglobin": "Hb",
    "wbc": "WBC",
    "white": "WBC",
    "platelet": "Platelets",
    "plt": "Platelets",
    "neut": "Neutrophils",
    "lymph": "Lymphocytes",
    "monocyte": "Monocytes",
    "eosin": "Eosinophils",
    "baso": "Basophils",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "crp": "CRP",
    "alt": "ALT",
    "ast": "AST",
    "ck-mb": "CK_MB",
    "ck": "CK",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",
    "calcium": "Calcium",
}

def parse_cbc(text: str) -> Dict[str, float]:
    out = {}
    for line in text.splitlines():
        low = line.lower()
        for k, canon in CBC_MAP.items():
            if k in low:
                val = clean_number(line)
                if val is not None:
                    out[canon] = val
    return out

# ============================================================
# ROUTE ENGINE V4 (SIMPLIFIED & SAFE)
# ============================================================

def route_engine(cbc: Dict[str, float]) -> Dict[str, Any]:
    patterns = []
    routes = []
    next_steps = []
    diff = []
    severity = "normal"
    urgency = "low"

    Hb = cbc.get("Hb")
    WBC = cbc.get("WBC")
    Neut = cbc.get("Neutrophils")
    CRP = cbc.get("CRP")
    Cr = cbc.get("Creatinine")

    if Hb and Hb < 11:
        patterns.append("anaemia")
        routes.append("Anaemia workup")
        diff += ["Iron deficiency", "Chronic disease"]

    if WBC and WBC > 15 and Neut and Neut > 70:
        patterns.append("leukocytosis with neutrophilia")
        routes.append("Infection / sepsis route")
        diff.append("Bacterial infection")
        severity = "severe"
        urgency = "high"

    if CRP and CRP > 100:
        patterns.append("very high CRP")
        routes.append("Severe inflammation / sepsis")
        urgency = "high"

    if Cr and Cr > 150:
        patterns.append("renal impairment")
        routes.append("AKI route")
        diff.append("Acute kidney injury")

    if not patterns:
        patterns.append("no major acute abnormalities")

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": list(set(diff)),
        "severity_text": severity,
        "urgency_flag": urgency
    }

# ============================================================
# MAIN PROCESSOR
# ============================================================

def process_report(job):
    report_id = job["id"]
    path = job["file_path"]

    print(f"Processing report {report_id}")

    pdf = supabase.storage.from_(BUCKET).download(path).data
    text = extract_text(pdf)

    if is_scanned(text):
        text = ocr_pdf(pdf)

    cbc = parse_cbc(text)

    if not cbc:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": "No CBC extracted"
        }).eq("id", report_id).execute()
        return

    routes = route_engine(cbc)

    ai_results = {
        "cbc": cbc,
        "patterns": routes["patterns"],
        "routes": routes["routes"],
        "differential": routes["differential"],
        "severity_text": routes["severity_text"],
        "urgency_flag": routes["urgency_flag"],
        "generated_at": now()
    }

    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_results
    }).eq("id", report_id).execute()

# ============================================================
# WORKER LOOP
# ============================================================

def main():
    print("Worker running...")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
            jobs = res.data or []
            if not jobs:
                time.sleep(1)
                continue
            job = jobs[0]
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", job["id"]).execute()
            process_report(job)
        except Exception as e:
            print("ERROR:", e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
