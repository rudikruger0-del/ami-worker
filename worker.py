#!/usr/bin/env python3
# =========================================================
# AMI HEALTH — WORKER V8 (HYBRID TABLE PARSER)
# Fixes: digital tables, scanned PDFs, OCR fallback
# Mode: BALANCED (GP-safe)
# =========================================================

import os
import io
import re
import time
import logging
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

import fitz  # PyMuPDF

try:
    import pytesseract
    HAS_OCR = True
except:
    pytesseract = None
    HAS_OCR = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except:
    create_client = None
    HAS_SUPABASE = False

# ---------------------------------------------------------
# ENV
# ---------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
PDF_DPI = int(os.getenv("PDF_RENDER_DPI", "220"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI-V8] %(levelname)s: %(message)s"
)
log = logging.getLogger("worker-v8")

supabase = None
if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------
# ANALYTE MAP
# ---------------------------------------------------------
LABEL_MAP = {
    "hb": "Hb",
    "hemoglobin": "Hb",
    "haemoglobin": "Hb",
    "rbc": "RBC",
    "hct": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "plt": "Platelets",
    "platelet": "Platelets",
    "wbc": "WBC",
    "white": "WBC",
    "neut": "Neutrophils",
    "lymph": "Lymphocytes",
    "mono": "Monocytes",
    "eos": "Eosinophils",
    "baso": "Basophils",
    "crp": "CRP",
    "urea": "Urea",
    "creat": "Creatinine",
    "creatinine": "Creatinine",
    "na": "Sodium",
    "sodium": "Sodium",
    "k": "Potassium",
    "potassium": "Potassium",
    "cl": "Chloride",
    "alt": "ALT",
    "ast": "AST",
    "ck-mb": "CK_MB",
    "ck": "CK",
    "albumin": "Albumin",
    "calcium": "Calcium",
    "ca": "Calcium",
    "co2": "CO2",
}

NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")

# ---------------------------------------------------------
# PDF DOWNLOAD
# ---------------------------------------------------------
def download_pdf(record: Dict[str, Any]) -> bytes:
    if "pdf_url" in record and record["pdf_url"]:
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if "file_path" in record and supabase:
        res = supabase.storage.from_(SUPABASE_BUCKET).download(record["file_path"])
        return res.data if hasattr(res, "data") else res

    raise ValueError("No PDF source")

# ---------------------------------------------------------
# SAFE FLOAT
# ---------------------------------------------------------
def safe_float(txt: str) -> Optional[float]:
    if not txt:
        return None
    txt = txt.replace(",", ".")
    m = NUM_RE.findall(txt)
    if not m:
        return None
    try:
        return float(m[-1])
    except:
        return None

# ---------------------------------------------------------
# 1️⃣ PYMuPDF TABLE PARSER (PRIMARY)
# ---------------------------------------------------------
def parse_tables_pymupdf(pdf_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    out = {}

    for page in doc:
        blocks = page.get_text("blocks")
        rows = {}

        for x0, y0, x1, y1, text, *_ in blocks:
            y_key = round(y0 / 5) * 5
            rows.setdefault(y_key, []).append((x0, text.strip()))

        for y in rows:
            cells = sorted(rows[y], key=lambda c: c[0])
            line = " ".join(c[1] for c in cells if c[1])

            lower = line.lower()
            for lbl, key in LABEL_MAP.items():
                if lbl in lower:
                    val = safe_float(line)
                    if val is not None:
                        out[key] = {"value": val, "raw": line}

    return out

# ---------------------------------------------------------
# OCR ENGINE (FALLBACK)
# ---------------------------------------------------------
def preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def parse_ocr(pdf_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    if not HAS_OCR:
        return {}

    images = convert_from_bytes(pdf_bytes, dpi=PDF_DPI)
    out = {}

    for img in images:
        txt = pytesseract.image_to_string(preprocess(img), config="--psm 6")
        for ln in txt.split("\n"):
            low = ln.lower()
            for lbl, key in LABEL_MAP.items():
                if lbl in low:
                    val = safe_float(ln)
                    if val is not None:
                        out[key] = {"value": val, "raw": ln}

    return out

# ---------------------------------------------------------
# FALLBACK TEXT PARSER
# ---------------------------------------------------------
def parse_text_fallback(pdf_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return {}

    out = {}
    for ln in text.split("\n"):
        low = ln.lower()
        for lbl, key in LABEL_MAP.items():
            if lbl in low:
                val = safe_float(ln)
                if val is not None:
                    out[key] = {"value": val, "raw": ln}
    return out

# ---------------------------------------------------------
# MASTER EXTRACTION
# ---------------------------------------------------------
def extract_all(pdf_bytes: bytes) -> Tuple[Dict[str, Dict[str, Any]], str]:
    parsed = parse_tables_pymupdf(pdf_bytes)
    if len(parsed) >= 5:
        return parsed, "pymupdf"

    log.warning("PyMuPDF insufficient — forcing OCR")
    parsed = parse_ocr(pdf_bytes)
    if len(parsed) >= 5:
        return parsed, "ocr"

    log.warning("OCR insufficient — using fallback text")
    parsed = parse_text_fallback(pdf_bytes)
    return parsed, "fallback"

# ---------------------------------------------------------
# SIMPLE BALANCED INTERPRETATION (SAFE)
# ---------------------------------------------------------
def interpret(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    patterns = []
    actions = []
    severity = "normal"

    def abnormal(k, v, low=None, high=None):
        nonlocal severity
        if v is None:
            return False
        if low is not None and v < low:
            severity = "moderate"
            return True
        if high is not None and v > high:
            severity = "moderate"
            return True
        return False

    hb = parsed.get("Hb", {}).get("value")
    plt = parsed.get("Platelets", {}).get("value")
    ck = parsed.get("CK", {}).get("value")
    ast = parsed.get("AST", {}).get("value")
    alt = parsed.get("ALT", {}).get("value")
    crp = parsed.get("CRP", {}).get("value")

    if abnormal("Hb", hb, low=12):
        patterns.append("Low hemoglobin")
    if abnormal("Platelets", plt, low=100):
        patterns.append("Thrombocytopenia")
        actions.append("Urgent platelet review")
        severity = "severe"
    if abnormal("CRP", crp, high=10):
        patterns.append("Inflammation")
    if abnormal("AST", ast, high=100) or abnormal("ALT", alt, high=100):
        patterns.append("Marked transaminitis")
        severity = "severe"
    if abnormal("CK", ck, high=1000):
        patterns.append("Possible rhabdomyolysis")
        actions.append("Assess muscle injury, renal risk")
        severity = "severe"

    if not parsed:
        severity = "insufficient_data"

    return {
        "severity_text": severity,
        "patterns": patterns,
        "next_steps": actions,
        "summary": " | ".join(patterns) if patterns else "No significant abnormalities detected"
    }

# ---------------------------------------------------------
# PROCESS REPORT
# ---------------------------------------------------------
def process_report(record: Dict[str, Any]):
    rid = record.get("id")
    pdf = download_pdf(record)

    parsed, method = extract_all(pdf)
    interpretation = interpret(parsed)

    result = {
        "parsed": parsed,
        "routes": interpretation,
        "extraction_method": method,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    if supabase:
        supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": result,
            "ai_error": None
        }).eq("id", rid).execute()

# ---------------------------------------------------------
# POLL
# ---------------------------------------------------------
def poll():
    if not supabase:
        log.error("Supabase not configured")
        return

    while True:
        rows = supabase.table(SUPABASE_TABLE)\
            .select("*").eq("ai_status", "pending").limit(5).execute().data

        for r in rows:
            supabase.table(SUPABASE_TABLE).update({"ai_status": "processing"})\
                .eq("id", r["id"]).execute()
            process_report(r)

        time.sleep(POLL_INTERVAL)

# ---------------------------------------------------------
if __name__ == "__main__":
    poll()
