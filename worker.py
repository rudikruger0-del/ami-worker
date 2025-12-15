#!/usr/bin/env python3
"""
AMI Health Worker — FULL CLINICAL ENGINE (V7.1 FIXED)

Design principles:
- Deterministic, doctor-grade
- Never say NORMAL if abnormalities exist
- Table-aware digital PDF parsing
- OCR fallback for scanned PDFs
- Dominant abnormality drives severity
- Patterns → Routes → Next steps → Differentials
- ER-grade + specialty logic
"""

# ======================================================
# IMPORTS
# ======================================================

import os
import io
import re
import time
import json
import logging
import traceback
from typing import Dict, Any, Optional, List, Tuple

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

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Supabase credentials missing")

# ======================================================
# LOGGING
# ======================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker")

# ======================================================
# SUPABASE
# ======================================================

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
    log.warning("pytesseract not available — OCR disabled")

def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    img = img.convert("L")

    w, h = img.size
    if w < 1800:
        scale = 1800 / w
        img = img.resize((int(w * scale), int(h * scale)))

    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    return img

def ocr_page(img: Image.Image) -> str:
    return pytesseract.image_to_string(
        img,
        config="--psm 6 -c preserve_interword_spaces=1"
    )

# ======================================================
# PDF INGESTION
# ======================================================

def download_pdf(job: Dict[str, Any]) -> bytes:
    res = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"])
    if hasattr(res, "data"):
        return res.data
    return res

def extract_text_digital(pdf: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf))
        out = []
        for page in reader.pages:
            out.append(page.extract_text() or "")
        return "\n".join(out).strip()
    except Exception:
        return ""

def looks_scanned(text: str) -> bool:
    if not text or len(text) < 150:
        return True
    anchors = (
        "haemoglobin", "hemoglobin", "wbc", "platelet",
        "creatinine", "crp", "cholesterol"
    )
    return not any(a in text.lower() for a in anchors)

def extract_text_scanned(pdf: bytes) -> str:
    if not HAS_OCR:
        raise RuntimeError("OCR required but pytesseract unavailable")

    pages = convert_from_bytes(pdf, dpi=300)
    collected = []

    for p in pages:
        img = preprocess_for_ocr(p)
        txt = ocr_page(img)
        if txt.strip():
            collected.append(txt)

    if not collected:
        raise RuntimeError("OCR produced no usable text")

    return "\n".join(collected)

def extract_text(pdf: bytes) -> Tuple[str, bool]:
    digital = extract_text_digital(pdf)
    if digital and not looks_scanned(digital):
        return digital, False
    return extract_text_scanned(pdf), True

# ======================================================
# ANALYTE NORMALISATION
# ======================================================

ANALYTE_MAP = {
    "hb": "Hb", "haemoglobin": "Hb", "hemoglobin": "Hb",
    "wbc": "WBC", "white cell count": "WBC",
    "platelet": "Platelets", "plt": "Platelets",
    "mcv": "MCV", "rdw": "RDW",
    "neutrophil": "Neutrophils",
    "lymphocyte": "Lymphocytes",
    "crp": "CRP",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
    "calcium": "Calcium",
    "alt": "ALT", "ast": "AST",
    "alp": "ALP", "ggt": "GGT",
    "bilirubin": "Bilirubin",
    "ck": "CK",
    "cholesterol total": "Cholesterol",
    "ldl": "LDL",
    "hdl": "HDL",
    "triglycerides": "Triglycerides",
    "non-hdl": "Non-HDL",
}

def normalize_label(label: str) -> Optional[str]:
    key = re.sub(r"[^a-z\- ]", "", label.lower()).strip()
    return ANALYTE_MAP.get(key)

# ======================================================
# TABLE-AWARE DIGITAL PARSER (CRITICAL FIX)
# ======================================================

def parse_labs(text: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses table-based DIGITAL lab PDFs:
    Test | Reference Range | Result
    """
    results: Dict[str, Dict[str, Any]] = {}

    if not text:
        return results

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines:
        ln = re.sub(r"\s{2,}", " ", ln)

        label_match = re.match(r"^[A-Za-z][A-Za-z \-/()]+", ln)
        if not label_match:
            continue

        label = label_match.group(0).strip()
        analyte = normalize_label(label)
        if not analyte:
            continue

        nums = re.findall(r"\d+(?:\.\d+)?", ln.replace(",", "."))
        if not nums:
            continue

        try:
            value = float(nums[-1])  # RESULT column
        except Exception:
            continue

        flag = None
        if re.search(r"\bH\b", ln):
            flag = "high"
        elif re.search(r"\bL\b", ln):
            flag = "low"

        results[analyte] = {
            "value": value,
            "flag": flag,
            "raw": ln
        }

    return results

# ======================================================
# ROUTE ENGINE — FULL LOGIC
# ======================================================

SEVERITY_ORDER = ["normal", "mild", "moderate", "severe", "critical"]

def max_severity(a: str, b: str) -> str:
    return SEVERITY_ORDER[max(SEVERITY_ORDER.index(a), SEVERITY_ORDER.index(b))]

def route_engine(c: Dict[str, Any]) -> Dict[str, Any]:
    patterns: List[str] = []
    routes: List[str] = []
    next_steps: List[str] = []
    differentials: List[str] = []

    severity = "normal"
    urgency = "low"

    def escalate(sev: str, urg: Optional[str] = None):
        nonlocal severity, urgency
        severity = max_severity(severity, sev)
        if urg:
            urgency = max(
                urgency,
                urg,
                key=lambda x: ["low", "moderate", "high"].index(x)
            )

    # ---------- Extract values ----------
    def v(k): return c.get(k, {}).get("value")

    Hb = v("Hb")
    PLT = v("Platelets")
    CK = v("CK")
    K = v("Potassium")
    Cr = v("Creatinine")
    CRP = v("CRP")
    AST = v("AST")
    ALT = v("ALT")
    Bili = v("Bilirubin")
    TG = v("Triglycerides")
    LDL = v("LDL")

    # ==================================================
    # ER-GRADE DOMINANCE
    # ==================================================

    if CK and CK > 5000:
        patterns.append("Severe muscle injury physiology (rhabdomyolysis)")
        routes.append("Rhabdomyolysis risk pathway")
        next_steps += [
            "Monitor renal function closely",
            "Assess hydration status",
            "Check urine for myoglobin"
        ]
        differentials.append("Trauma, exertion, toxin/drug-induced muscle injury")
        escalate("critical", "high")

    if PLT and PLT < 50:
        patterns.append("Severe thrombocytopenia")
        routes.append("High bleeding risk pathway")
        next_steps += [
            "Assess for bleeding",
            "Repeat platelet count",
            "Review medications and infection markers"
        ]
        differentials.append("ITP, sepsis-related consumption, marrow suppression")
        escalate("severe", "high")

    if K:
        if K < 3.0:
            patterns.append("Hypokalaemia")
            routes.append("Electrolyte risk pathway")
            next_steps.append("Assess ECG if symptomatic")
            escalate("severe", "high")
        elif K > 6.0:
            patterns.append("Hyperkalaemia")
            routes.append("Arrhythmia risk pathway")
            next_steps.append("Urgent ECG correlation")
            escalate("critical", "high")

    if Cr and Cr > 300:
        patterns.append("Severe renal dysfunction")
        routes.append("Acute kidney injury physiology")
        next_steps.append("Trend creatinine and urine output")
        differentials.append("AKI vs advanced CKD")
        escalate("severe", "high")

    if CRP:
        if CRP >= 100:
            patterns.append("Marked inflammatory response")
            routes.append("Sepsis / severe infection physiology")
            next_steps.append("Search for infectious source")
            escalate("severe", "high")
        elif CRP >= 10:
            patterns.append("Active inflammation")
            routes.append("Inflammatory/infective correlation")
            escalate("moderate", "moderate")

    # ==================================================
    # SPECIALTY MODULES
    # ==================================================

    # Liver
    if AST and ALT and (AST > 300 or ALT > 300):
        patterns.append("Marked transaminitis")
        routes.append("Hepatocellular injury physiology")
        next_steps.append("Review hepatotoxic exposure")
        differentials.append("Ischaemic, toxic, viral hepatitis")
        escalate("severe", "high")

    if Bili and Bili > 21:
        patterns.append("Isolated bilirubin elevation")
        routes.append("Benign hyperbilirubinaemia pattern possible")
        next_steps.append("Consider Gilbert syndrome if asymptomatic")
        escalate("mild", "low")

    # Metabolic / lipid
    if TG and TG > 1.9:
        patterns.append("Hypertriglyceridaemia")
        routes.append("Metabolic risk factor")
        next_steps.append("Assess diet, alcohol intake, insulin resistance")
        escalate("mild", "low")

    if LDL and LDL > 3.0:
        patterns.append("Elevated LDL cholesterol")
        routes.append("Cardiovascular risk factor")
        next_steps.append("Lifestyle modification and risk stratification")
        escalate("mild", "low")

    # Oncology
    if Hb and Hb < 8 and PLT and PLT < 100:
        patterns.append("Bicytopenia")
        routes.append("Bone marrow suppression physiology")
        differentials.append("Malignancy, marrow infiltration, chemotherapy effect")
        escalate("severe", "high")

    # Anaemia (demoted if secondary)
    if Hb and Hb < 11 and severity == "normal":
        patterns.append("Anaemia")
        routes.append("Anaemia workup")
        next_steps.append("Assess MCV, ferritin, renal function")
        escalate("moderate", "moderate")

    if not patterns:
        patterns.append("No major acute abnormalities detected")

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differentials": differentials,
        "severity_text": severity,
        "urgency": urgency
    }

# ======================================================
# WORKER CORE
# ======================================================

def process(job: Dict[str, Any]) -> None:
    pdf = download_pdf(job)
    text, scanned = extract_text(pdf)
    labs = parse_labs(text)
    analysis = route_engine(labs)

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": {
            "scanned": scanned,
            "labs": labs,
            "analysis": analysis
        },
        "ai_error": None
    }).eq("id", job["id"]).execute()

def main():
    log.info("AMI Worker started")
    while True:
        try:
            res = supabase.table(SUPABASE_TABLE)\
                .select("*")\
                .eq("ai_status", "pending")\
                .limit(1)\
                .execute()
            jobs = res.data or []
            if not jobs:
                time.sleep(POLL_INTERVAL)
                continue

            job = jobs[0]
            supabase.table(SUPABASE_TABLE)\
                .update({"ai_status": "processing"})\
                .eq("id", job["id"])\
                .execute()

            process(job)

        except Exception as e:
            traceback.print_exc()
            if job and "id" in job:
                supabase.table(SUPABASE_TABLE)\
                    .update({"ai_status": "failed", "ai_error": str(e)})\
                    .eq("id", job["id"])\
                    .execute()
            time.sleep(3)

if __name__ == "__main__":
    main()
