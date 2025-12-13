#!/usr/bin/env python3
# ============================================================
# AMI HEALTH — WORKER V8.1 (PRODUCTION)
# Pattern-first, doctor-grade CBC interpretation
# Supports DIGITAL PDFs + SCANNED PDFs (OCR fallback)
# ============================================================

import os, io, re, time, json, logging
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

# OCR
import pytesseract

# Supabase
from supabase import create_client

# PyMuPDF (table-aware extraction)
import fitz  # pymupdf

# ------------------------------------------------------------
# ENV + LOGGING
# ------------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI-V8.1] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker-v8.1")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ------------------------------------------------------------
# HARD LIMITS (SAFETY — NEVER ROUTE IMPOSSIBLE VALUES)
# ------------------------------------------------------------
PHYSIO_LIMITS = {
    "Potassium": (1.5, 7.5),
    "Sodium": (110, 170),
    "Creatinine": (20, 2000),
    "WBC": (0.1, 200),
    "Platelets": (1, 5000),
    "CRP": (0, 1000),
    "CK": (0, 200000),
}

# ------------------------------------------------------------
# PDF DOWNLOAD
# ------------------------------------------------------------
def download_pdf(record: Dict[str, Any]) -> bytes:
    if record.get("file_path"):
        res = supabase.storage.from_(SUPABASE_BUCKET).download(record["file_path"])
        return res.data if hasattr(res, "data") else res
    raise RuntimeError("Missing file_path")

# ------------------------------------------------------------
# DIGITAL PDF — PyMuPDF TABLE EXTRACTION
# ------------------------------------------------------------
def extract_with_pymupdf(pdf_bytes: bytes) -> Dict[str, float]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    results = {}

    for page in doc:
        blocks = page.get_text("blocks")
        rows = {}

        for b in blocks:
            text = b[4].strip()
            y = round(b[1], 1)
            if text:
                rows.setdefault(y, []).append(text)

        for y in rows:
            row = " ".join(rows[y])
            m = re.match(r"([A-Za-z \-/]+)\s+([0-9.,]+)\s*([LH]?)", row)
            if m:
                label = normalize_label(m.group(1))
                value = safe_float(m.group(2))
                if label and value is not None:
                    results[label] = value

    return results

# ------------------------------------------------------------
# OCR PIPELINE (SCANNED PDFs)
# ------------------------------------------------------------
def preprocess(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.MedianFilter(3))
    return img

def ocr_pdf(pdf_bytes: bytes) -> Dict[str, float]:
    images = convert_from_bytes(pdf_bytes, dpi=300)
    text = ""
    for img in images:
        img = preprocess(img)
        text += pytesseract.image_to_string(img)

    return parse_text_lines(text)

# ------------------------------------------------------------
# FALLBACK TEXT PARSER
# ------------------------------------------------------------
def parse_text_lines(text: str) -> Dict[str, float]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"([A-Za-z \-/]+)\s*[:\-]?\s*([0-9.,]+)", line)
        if m:
            label = normalize_label(m.group(1))
            value = safe_float(m.group(2))
            if label and value is not None:
                out[label] = value
    return out

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------
def safe_float(x: str) -> Optional[float]:
    try:
        x = x.replace(",", ".")
        return float(re.sub(r"[^\d.]", "", x))
    except:
        return None

def normalize_label(lbl: str) -> Optional[str]:
    lbl = lbl.lower().strip()
    MAP = {
        "hb": "Hb", "hemoglobin": "Hb",
        "rbc": "RBC",
        "hct": "HCT",
        "mcv": "MCV",
        "mch": "MCH",
        "mchc": "MCHC",
        "rdw": "RDW",
        "wbc": "WBC",
        "neut": "Neutrophils",
        "lymph": "Lymphocytes",
        "mono": "Monocytes",
        "eos": "Eosinophils",
        "baso": "Basophils",
        "plt": "Platelets",
        "platelets": "Platelets",
        "crp": "CRP",
        "creat": "Creatinine",
        "urea": "Urea",
        "na": "Sodium",
        "k": "Potassium",
        "cl": "Chloride",
        "alt": "ALT",
        "ast": "AST",
        "ck": "CK",
        "ck-mb": "CK_MB",
        "albumin": "Albumin",
        "co2": "CO2",
        "total co2": "CO2",
        "calcium": "Calcium",
    }
    for k in MAP:
        if k in lbl:
            return MAP[k]
    return None

# ------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------
def validate_values(values: Dict[str, float]) -> Dict[str, float]:
    valid = {}
    for k, v in values.items():
        if k in PHYSIO_LIMITS:
            lo, hi = PHYSIO_LIMITS[k]
            if not (lo <= v <= hi):
                log.warning(f"Rejected impossible value: {k}={v}")
                continue
        valid[k] = v
    return valid

# ------------------------------------------------------------
# PATTERN-FIRST CLINICAL ENGINE (DOCTOR STYLE)
# ------------------------------------------------------------
def clinical_engine(vals: Dict[str, float]) -> Dict[str, Any]:
    patterns = []
    dx = []
    actions = []

    Hb = vals.get("Hb")
    MCV = vals.get("MCV")
    WBC = vals.get("WBC")
    Neut = vals.get("Neutrophils")
    CRP = vals.get("CRP")
    Plt = vals.get("Platelets")
    Creat = vals.get("Creatinine")
    CO2 = vals.get("CO2")
    CK = vals.get("CK")

    # --- ANEMIA ---
    if Hb and Hb < 12:
        if MCV and MCV < 80:
            patterns.append("Microcytic anemia")
            dx.append("Iron deficiency anemia — low Hb with microcytosis")
            actions.append("Order ferritin and reticulocyte count")
        else:
            patterns.append("Anemia")
            dx.append("Anemia — low hemoglobin")

    # --- INFECTION / SEPSIS ---
    if WBC and WBC > 12 and Neut and Neut > 75 and CRP and CRP > 50:
        patterns.append("Sepsis pattern")
        dx.append("Sepsis — neutrophilia with leukocytosis and markedly elevated CRP")
        actions += [
            "Urgent clinical assessment for sepsis",
            "Blood cultures if febrile",
            "IV fluids and empiric antibiotics if unstable"
        ]

    # --- AKI ---
    if Creat and Creat > 120:
        patterns.append("Acute kidney injury")
        dx.append("Acute kidney injury — elevated creatinine")
        actions += [
            "Repeat creatinine urgently",
            "Assess urine output and volume status"
        ]
        if CO2 and CO2 < 22:
            dx.append("Possible prerenal AKI — low CO2, check fluid status")

    # --- RHABDOMYOLYSIS ---
    if CK and CK > 1000:
        patterns.append("Marked transaminitis / rhabdomyolysis")
        dx.append("Possible rhabdomyolysis — markedly elevated CK")
        actions.append("Assess muscle injury; aggressive IV hydration")

    # --- SEVERITY ---
    severity = "normal"
    if len(patterns) >= 3 or ("Sepsis pattern" in patterns):
        severity = "severe"
    if CK and CK > 10000 or (Creat and Creat > 250):
        severity = "critical"

    return {
        "severity_text": severity,
        "patterns": patterns,
        "diagnostic_possibilities": dx,
        "immediate_actions": actions
    }

# ------------------------------------------------------------
# MAIN PROCESS
# ------------------------------------------------------------
def process_record(rec: Dict[str, Any]):
    pdf = download_pdf(rec)

    # 1. Digital first
    vals = extract_with_pymupdf(pdf)

    # 2. Fallback OCR
    if len(vals) < 5:
        log.warning("PyMuPDF insufficient — forcing OCR")
        vals = ocr_pdf(pdf)

    vals = validate_values(vals)

    # 3. Safety check
    if not {"Hb", "WBC", "Platelets"} & vals.keys():
        result = {
            "severity_text": "unreliable",
            "summary": "Insufficient CBC data — manual review required",
            "parsed": vals
        }
    else:
        result = clinical_engine(vals)
        result["parsed"] = vals

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": result
    }).eq("id", rec["id"]).execute()

# ------------------------------------------------------------
# POLLER
# ------------------------------------------------------------
def poll():
    while True:
        rows = supabase.table(SUPABASE_TABLE)\
            .select("*").eq("ai_status", "pending").limit(3).execute().data
        for r in rows:
            supabase.table(SUPABASE_TABLE)\
                .update({"ai_status": "processing"})\
                .eq("id", r["id"]).execute()
            process_record(r)
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    poll()
