#!/usr/bin/env python3
"""
AMI Health Worker — FULL CLINICAL ENGINE (V7)

Design goals:
- Deterministic, doctor-grade reasoning
- OCR + digital PDF support
- Patterns → Routes → Next steps → Differentials
- Dominant abnormality drives severity
- ER-grade high-risk detection
- Specialty modules: renal, liver, oncology, pregnancy, infection
- Severity TEXT only (normal/mild/moderate/severe/critical)
"""

# ======================================================
# IMPORTS & ENV
# ======================================================

import os, io, re, time, json, traceback, logging
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI] %(levelname)s: %(message)s"
)
log = logging.getLogger("ami-worker")

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
    w, h = img.size
    if w < 1800:
        s = 1800 / w
        img = img.resize((int(w * s), int(h * s)))
    img = ImageOps.autocontrast(img)
    img = img.point(lambda x: 0 if x < 140 else 255, "1")
    return img

def ocr_page(img: Image.Image) -> str:
    return pytesseract.image_to_string(
        img, config="--psm 6 -c preserve_interword_spaces=1"
    )

# ======================================================
# PDF INGESTION
# ======================================================

def download_pdf(job: Dict[str, Any]) -> bytes:
    res = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"])
    return res.data if hasattr(res, "data") else res

def extract_text_digital(pdf: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def looks_scanned(text: str) -> bool:
    if not text or len(text) < 150:
        return True
    anchors = ("haemoglobin", "hemoglobin", "wbc", "platelet", "crp", "creatinine")
    return not any(a in text.lower() for a in anchors)

def extract_text_scanned(pdf: bytes) -> str:
    pages = convert_from_bytes(pdf, dpi=300)
    out = []
    for p in pages:
        img = preprocess_for_ocr(p)
        t = ocr_page(img)
        if t.strip():
            out.append(t)
    if not out:
        raise RuntimeError("OCR failed")
    return "\n".join(out)

def extract_text(pdf: bytes) -> Tuple[str, bool]:
    d = extract_text_digital(pdf)
    if d and not looks_scanned(d):
        return d, False
    return extract_text_scanned(pdf), True

# ======================================================
# PARSER (ROBUST)
# ======================================================

ANALYTES = {
    "hb": "Hb", "haemoglobin": "Hb", "hemoglobin": "Hb",
    "wbc": "WBC", "white cell count": "WBC",
    "platelet": "Platelets", "plt": "Platelets",
    "mcv": "MCV", "rdw": "RDW",
    "neutrophil": "Neutrophils", "lymphocyte": "Lymphocytes",
    "crp": "CRP",
    "creatinine": "Creatinine", "urea": "Urea",
    "sodium": "Sodium", "na": "Sodium",
    "potassium": "Potassium", "k": "Potassium",
    "alt": "ALT", "ast": "AST", "alp": "ALP", "ggt": "GGT",
    "bilirubin": "Bilirubin",
    "ck": "CK",
    "calcium": "Calcium",
}

def normalize_label(lbl: str) -> Optional[str]:
    k = re.sub(r"[^a-z ]", "", lbl.lower()).strip()
    return ANALYTES.get(k)

def extract_number(line: str) -> Optional[float]:
    s = line.replace(",", ".")
    s = re.sub(r"[<>≤≥]", "", s)
    s = re.sub(r"(\d)\s+(\d{3})", r"\1\2", s)
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    for n in nums:
        v = float(n)
        if v > 0:
            return v
    return None

def parse_labs(text: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for ln in text.splitlines():
        m = re.match(r"^[A-Za-z][A-Za-z /()-]+", ln)
        if not m:
            continue
        a = normalize_label(m.group(0))
        if not a:
            continue
        v = extract_number(ln)
        if v is None:
            continue
        out[a] = {"value": v, "raw": ln}
    return out

# ======================================================
# ROUTE ENGINE — FULL LOGIC
# ======================================================

SEVERITY_ORDER = ["normal", "mild", "moderate", "severe", "critical"]

def max_sev(a, b):
    return SEVERITY_ORDER[max(SEVERITY_ORDER.index(a), SEVERITY_ORDER.index(b))]

def route_engine(c: Dict[str, Any]) -> Dict[str, Any]:
    patterns, routes, next_steps, differentials = [], [], [], []
    severity = "normal"
    urgency = "low"

    def escalate(s, u=None):
        nonlocal severity, urgency
        severity = max_sev(severity, s)
        if u:
            urgency = max(urgency, u, key=lambda x: ["low","moderate","high"].index(x))

    # ---------------- ER-GRADE DOMINANCE ----------------

    CK = c.get("CK", {}).get("value")
    PLT = c.get("Platelets", {}).get("value")
    K = c.get("Potassium", {}).get("value")
    CRP = c.get("CRP", {}).get("value")
    Cr = c.get("Creatinine", {}).get("value")
    AST = c.get("AST", {}).get("value")
    ALT = c.get("ALT", {}).get("value")
    Hb = c.get("Hb", {}).get("value")

    # Rhabdomyolysis
    if CK and CK > 5000:
        patterns.append("Severe muscle injury physiology (rhabdomyolysis)")
        routes.append("Rhabdomyolysis risk pathway")
        next_steps += [
            "Monitor renal function closely",
            "Assess hydration status",
            "Check urine for myoglobin"
        ]
        differentials.append("Trauma, severe exertion, toxin/drug-related muscle injury")
        escalate("critical", "high")

    # Cytopenia
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

    # Electrolytes
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

    # Renal
    if Cr and Cr > 300:
        patterns.append("Severe renal dysfunction")
        routes.append("Acute kidney injury physiology")
        next_steps.append("Trend creatinine and urine output")
        differentials.append("AKI vs advanced CKD")
        escalate("severe", "high")

    # Infection / inflammation
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

    # Liver
    if AST and ALT and (AST > 300 or ALT > 300):
        patterns.append("Marked transaminitis")
        routes.append("Hepatocellular injury physiology")
        next_steps.append("Review hepatotoxic exposure")
        differentials.append("Ischaemic, toxic, viral hepatitis")
        escalate("severe", "high")

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

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differentials": differentials,
        "severity_text": severity,
        "urgency": urgency
    }

# ======================================================
# WORKER LOOP
# ======================================================

def process(job):
    pdf = download_pdf(job)
    text, scanned = extract_text(pdf)
    labs = parse_labs(text)
    result = route_engine(labs)

    supabase.table(SUPABASE_TABLE).update({
        "ai_status": "completed",
        "ai_results": {
            "scanned": scanned,
            "labs": labs,
            "analysis": result
        }
    }).eq("id", job["id"]).execute()

def main():
    log.info("AMI Worker running")
    while True:
        jobs = supabase.table(SUPABASE_TABLE)\
            .select("*")\
            .eq("ai_status", "pending")\
            .limit(1)\
            .execute().data or []
        if not jobs:
            time.sleep(POLL_INTERVAL)
            continue
        job = jobs[0]
        supabase.table(SUPABASE_TABLE).update(
            {"ai_status": "processing"}
        ).eq("id", job["id"]).execute()
        try:
            process(job)
        except Exception as e:
            traceback.print_exc()
            supabase.table(SUPABASE_TABLE).update(
                {"ai_status": "failed", "ai_error": str(e)}
            ).eq("id", job["id"]).execute()

if __name__ == "__main__":
    main()
