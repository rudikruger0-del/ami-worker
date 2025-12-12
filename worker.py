#!/usr/bin/env python3
# AMI Health Worker V7 — FULL REBUILD (Balanced Clinical Mode)
# PART 1 — PDF Engine, OCR Engine, Ultra Parser V7

import os
import io
import re
import time
import json
import math
import logging
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps

# Optional imports
try:
    import pytesseract
    HAS_PYTESSERACT = True
except:
    pytesseract = None
    HAS_PYTESSERACT = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except:
    create_client = None
    HAS_SUPABASE = False


# ---------------------------------------------
# ENV + LOGGING
# ---------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "220"))
OCR_LANG = os.getenv("OCR_LANG", "eng")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AMI-V7] %(levelname)s: %(message)s"
)
logger = logging.getLogger("ami-worker-v7")

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None
    logger.warning("Supabase disabled — worker will run local only.")


# ---------------------------------------------
# PDF DOWNLOADER
# ---------------------------------------------
def download_pdf_from_record(record: Dict[str, Any]) -> bytes:
    """Fetch PDF from Supabase or direct URL."""
    if "pdf_url" in record and record["pdf_url"]:
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if "file_path" in record and supabase:
        try:
            res = supabase.storage.from_(SUPABASE_BUCKET).download(record["file_path"])
            return res.data if hasattr(res, "data") else res
        except Exception as e:
            logger.exception("Supabase download failed: %s", e)
            raise

    raise ValueError("No valid PDF location in record.")


# ---------------------------------------------
# DIGITAL TEXT EXTRACTION
# ---------------------------------------------
def extract_text_pypdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except:
        return ""

    out = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except:
            t = ""
        out.append(t)

    return "\n".join(out)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    txt = extract_text_pypdf(pdf_bytes)
    return len(txt.strip()) < TEXT_LENGTH_THRESHOLD


# ---------------------------------------------
# OCR ENGINE (Improved)
# ---------------------------------------------
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Noise removal + grayscale + enlarge for OCR."""
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        w, h = img.size
        if max(w, h) < 1800:
            scale = 1800 / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img
    except:
        return img


def ocr_image(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        return ""
    img2 = preprocess_for_ocr(img)
    try:
        txt = pytesseract.image_to_string(img2, lang=OCR_LANG, config="--oem 3 --psm 6")
    except:
        txt = ""
    return "".join(ch if (31 < ord(ch) < 127 or ch in "\n\r\t") else " " for ch in txt)


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=PDF_RENDER_DPI)


def extract_text_ocr(pdf_bytes: bytes) -> str:
    images = pdf_to_images(pdf_bytes)
    out = []
    for i, img in enumerate(images):
        t = ocr_image(img)
        logger.info(f"OCR page {i+1}: {len(t)} chars")
        out.append(t)
    return "\n".join(out)


# ---------------------------------------------
# ULTRA CBC PARSER V7 (MOST IMPORTANT FIX)
# ---------------------------------------------
CBC_LABEL_MAP = {
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "hb": "Hb",
    "rbc": "RBC",
    "erythrocyte": "RBC",
    "hct": "HCT",
    "hematocrit": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",
    "white cell": "WBC",
    "wbc": "WBC",
    "neut": "Neutrophils",
    "neutrophils": "Neutrophils",
    "lymph": "Lymphocytes",
    "lymphocytes": "Lymphocytes",
    "mono": "Monocytes",
    "monocytes": "Monocytes",
    "eos": "Eosinophils",
    "eosinophils": "Eosinophils",
    "baso": "Basophils",
    "basophils": "Basophils",
    "platelet": "Platelets",
    "plt": "Platelets",
    "crp": "CRP",
    "creatinine": "Creatinine",
    "urea": "Urea",
    "sodium": "Sodium",
    "na": "Sodium",
    "potassium": "Potassium",
    "k": "Potassium",
    "chloride": "Chloride",
    "cl": "Chloride",
    "calcium": "Calcium",
    "albumin": "Albumin",
    "co2": "CO2",
    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ck-mb": "CK_MB"
}

NUMERIC_RE = re.compile(r"([-+]?\d*\.\d+|\d+)")
PERCENT_RE = re.compile(r"(\d+(\.\d+)?)\s*%")

def extract_number(text: str) -> Optional[float]:
    """Extract numeric or percent values."""
    p = PERCENT_RE.search(text)
    if p:
        return float(p.group(1))
    m = NUMERIC_RE.findall(text)
    if m:
        return float(m[-1][0])  # last numeric in the line
    return None


def find_values_in_text(text: str) -> Dict[str, Dict[str, Any]]:
    """Reads ANY CBC table — PathCare/Lancet/NHLS — scanned or digital."""
    out = {}

    lines = [ln.strip().lower() for ln in text.split("\n") if ln.strip()]

    for ln in lines:
        clean = re.sub(r"\s+", " ", ln)

        for lbl, key in CBC_LABEL_MAP.items():
            if lbl in clean:
                val = extract_number(clean)
                if val is not None:
                    out[key] = {"value": val, "raw_line": ln}
                break

    return out
# ---------------------------------------------
# CANONICAL KEYS
# ---------------------------------------------
CANONICAL_KEYS = [
    "Hb", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW",
    "WBC", "Neutrophils", "Lymphocytes", "Monocytes",
    "Eosinophils", "Basophils", "NLR",
    "Platelets", "CRP", "Creatinine", "Urea",
    "Sodium", "Potassium", "Chloride", "Calcium",
    "Albumin", "CO2", "ALT", "AST", "CK", "CK_MB"
]


# ---------------------------------------------
# CANONICAL MAPPING ENGINE
# ---------------------------------------------
def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}

    for k, v in parsed.items():
        if k not in CANONICAL_KEYS:
            continue
        val = v.get("value")
        out[k] = {"value": val, "raw": v.get("raw_line")}

    # Compute NLR if possible
    neut = out.get("Neutrophils", {}).get("value")
    lymph = out.get("Lymphocytes", {}).get("value")
    if neut is not None and lymph is not None and lymph != 0:
        out["NLR"] = {"value": round(neut / lymph, 2)}

    return out


# ---------------------------------------------
# AGE GROUP CLASSIFIER
# ---------------------------------------------
def age_group_from_age(age: Optional[float]) -> str:
    if age is None:
        return "adult"

    try:
        a = float(age)
    except:
        return "adult"

    if a < (1 / 12): return "neonate"
    if a < 1: return "infant"
    if a < 13: return "child"
    if a < 18: return "teen"
    if a < 65: return "adult"
    return "elderly"


# ---------------------------------------------
# BALANCED SEVERITY SYSTEM (MODE B)
# ---------------------------------------------
def score_severity_v7(key: str, value: Optional[float], age_group: str, sex: str) -> int:
    """
    Balanced clinical thresholds:
    - Uses softer scoring than ICU-level engines
    - Avoids aggressive sepsis labeling
    - Produces realistic GP-facing severity
    """
    if value is None:
        return 1

    try:
        v = float(value)
    except:
        return 1

    key = key.lower()
    s = 1

    # Hemoglobin
    if key == "hb":
        low = 12.0 if sex.lower() == "female" else 13.0
        if age_group in ["neonate", "infant"]:
            low = 14.0

        if v < low - 3: s = 4
        elif v < low - 1: s = 3
        elif v < low: s = 2

    # WBC
    elif key == "wbc":
        if v > 25: s = 4
        elif v > 15: s = 3
        elif v > 11: s = 2

    # Neutrophils % / NLR
    elif key in ("neutrophils", "nlr"):
        if v > 10: s = 4
        elif v > 6: s = 3
        elif v > 3: s = 2

    # CRP
    elif key == "crp":
        if v > 150: s = 4
        elif v > 75: s = 3
        elif v > 20: s = 2

    # Creatinine
    elif key == "creatinine":
        if v > 300: s = 4
        elif v > 150: s = 3
        elif v > 120: s = 2

    # Electrolytes
    elif key == "sodium":
        if v < 120 or v > 160: s = 4
        elif v < 130 or v > 150: s = 3

    elif key == "potassium":
        if v < 2.8 or v > 6.2: s = 4
        elif v < 3.3 or v > 5.7: s = 3

    # Platelets
    elif key == "platelets":
        if v < 30: s = 4
        elif v < 100: s = 3
        elif v < 150: s = 2
        elif v > 600: s = 3

    return s


# ---------------------------------------------
# SEVERITY → TEXT
# ---------------------------------------------
def severity_text_for_score(s: int) -> str:
    if s <= 1: return "normal"
    if s == 2: return "mild"
    if s == 3: return "moderate"
    return "severe"  # Balanced mode does not use "critical"


# ---------------------------------------------
# PATTERN ENGINE (Balanced Mode)
# ---------------------------------------------
def add_pattern(patterns, per_key_scores, name, reason, score):
    patterns.append({"pattern": name, "reason": reason})
    per_key_scores[name] = max(per_key_scores.get(name, 1), score)


# ---------------------------------------------
# ROUTE ENGINE V7 (BALANCED)
# ---------------------------------------------
def route_engine_v7(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any]) -> Dict[str, Any]:
    age = patient_meta.get("age")
    sex = patient_meta.get("sex", "unknown")
    ag = age_group_from_age(age)

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    per_key_scores = {}

    # Extract values
    hb = canonical.get("Hb", {}).get("value")
    wbc = canonical.get("WBC", {}).get("value")
    crp = canonical.get("CRP", {}).get("value")
    neut = canonical.get("Neutrophils", {}).get("value")
    nlr = canonical.get("NLR", {}).get("value")
    plate = canonical.get("Platelets", {}).get("value")
    sodium = canonical.get("Sodium", {}).get("value")
    potassium = canonical.get("Potassium", {}).get("value")
    creat = canonical.get("Creatinine", {}).get("value")
    mcv = canonical.get("MCV", {}).get("value")

    # 1. Anemia patterns
    if hb is not None:
        s = score_severity_v7("Hb", hb, ag, sex)
        if s > 1:
            if mcv is not None and mcv < 80:
                add_pattern(patterns, per_key_scores, "microcytic anemia", f"Hb {hb}, MCV {mcv}", s)
                routes.append("Iron-deficiency evaluation")
                ddx.extend(["Iron deficiency", "Chronic blood loss"])
                next_steps.append("Order ferritin + CRP-adjusted interpretation.")
            elif mcv is not None and mcv > 100:
                add_pattern(patterns, per_key_scores, "macrocytic anemia", f"Hb {hb}, MCV {mcv}", s)
                routes.append("B12/folate evaluation")
                next_steps.append("Check B12 and folate.")
            else:
                add_pattern(patterns, per_key_scores, "anemia", f"Hb {hb}", s)
                ddx.append("Anemia of inflammation")

    # 2. Infection / Inflammation indicators (Balanced wording)
    infection_flag = False

    if wbc is not None and wbc > 11:
        s = score_severity_v7("WBC", wbc, ag, sex)
        add_pattern(patterns, per_key_scores, "elevated WBC", f"WBC {wbc}", s)
        routes.append("Infection/inflammation evaluation")
        infection_flag = True

    if neut is not None and neut > 75:
        s = score_severity_v7("Neutrophils", neut, ag, sex)
        add_pattern(patterns, per_key_scores, "neutrophil shift", f"Neutrophils {neut}%", s)
        infection_flag = True

    if nlr is not None and nlr > 6:
        s = score_severity_v7("NLR", nlr, ag, sex)
        add_pattern(patterns, per_key_scores, "elevated NLR", f"NLR {nlr}", s)
        infection_flag = True

    if crp is not None and crp > 20:
        s = score_severity_v7("CRP", crp, ag, sex)
        add_pattern(patterns, per_key_scores, "raised CRP", f"CRP {crp}", s)
        infection_flag = True

    if infection_flag:
        routes.append("Evaluate for possible infection indicators")
        ddx.extend(["Possible bacterial infection", "Inflammatory response"])
        next_steps.append("Clinical correlation recommended (temperature, symptoms).")

    # 3. Electrolyte issues
    if potassium is not None:
        s = score_severity_v7("Potassium", potassium, ag, sex)
        if s > 1:
            add_pattern(patterns, per_key_scores, "potassium imbalance", f"K {potassium}", s)
            next_steps.append("Repeat potassium urgently; assess ECG if low or elevated.")
            ddx.append("Electrolyte imbalance")

    if sodium is not None:
        s = score_severity_v7("Sodium", sodium, ag, sex)
        if s > 1:
            add_pattern(patterns, per_key_scores, "sodium imbalance", f"Na {sodium}", s)
            next_steps.append("Assess hydration status and repeat.")

    # 4. Kidney function
    if creat is not None:
        s = score_severity_v7("Creatinine", creat, ag, sex)
        if s > 1:
            add_pattern(patterns, per_key_scores, "renal signal", f"Creatinine {creat}", s)
            ddx.append("Possible renal impairment")
            next_steps.append("Repeat creatinine; check electrolytes.")

    # Severity aggregation
    final_score = max(per_key_scores.values()) if per_key_scores else 1
    severity_text = severity_text_for_score(final_score)

    # Build summary text
    summary_lines = []

    if patterns:
        summary_lines.append("Patterns identified: " +
                             "; ".join([p["pattern"] for p in patterns]))

    if routes:
        summary_lines.append("Evaluation paths: " + "; ".join(routes))

    if ddx:
        summary_lines.append("Possible considerations: " + ", ".join(ddx))

    if next_steps:
        summary_lines.append("Suggested actions: " + " | ".join(next_steps))

    summary = "\n".join(summary_lines) if summary_lines else "No significant abnormalities detected."

    return {
        "patterns": patterns,
        "routes": routes,
        "differential": ddx,
        "next_steps": next_steps,
        "severity_text": severity_text,
        "urgency_flag": "high" if final_score == 4 else "medium" if final_score == 3 else "low",
        "summary": summary,
        "final_score": final_score,
        "age_group": ag
    }
# ---------------------------------------------
# RISK BAR CALCULATOR (UI SUPPORT)
# ---------------------------------------------
def risk_bar_v7(key: str, value: Optional[float]) -> Tuple[int, str]:
    """Returns (percentage, color hex). Balanced version."""
    if value is None:
        return (0, "#10b981")

    try:
        v = float(value)
    except:
        return (0, "#10b981")

    pct = 0

    k = key.lower()

    # CRP
    if k == "crp":
        pct = min(100, int((v / 200) * 100))

    # WBC
    elif k == "wbc":
        if v <= 11: pct = int((v / 11) * 30)
        elif v <= 25: pct = 40 + int(((v - 11) / 14) * 40)
        else: pct = 80 + min(20, int((v - 25) * 2))

    # NLR / Neutrophils
    elif k in ("neutrophils", "nlr"):
        if v <= 3: pct = int((v / 3) * 20)
        elif v <= 6: pct = 25 + int(((v - 3) / 3) * 25)
        elif v <= 10: pct = 55 + int(((v - 6) / 4) * 25)
        else: pct = 80 + min(20, int((v - 10) * 2))

    # Creatinine
    elif k == "creatinine":
        if v <= 120:
            pct = int((v / 120) * 30)
        elif v <= 300:
            pct = 40 + int(((v - 120) / 180) * 40)
        else:
            pct = 80 + min(20, int((v - 300) / 5))

    # Hb
    elif k == "hb":
        if v >= 12:
            pct = 5
        elif v >= 10:
            pct = 20
        elif v >= 8:
            pct = 40
        else:
            pct = 70

    # Platelets
    elif k == "platelets":
        if v < 150:
            pct = min(100, int((150 - v) * 0.4))
        elif v > 600:
            pct = min(100, int((v - 600) * 0.1))
        else:
            pct = 5

    # fallback
    pct = min(max(pct, 0), 100)

    # color scale
    if pct >= 80: color = "#b91c1c"
    elif pct >= 60: color = "#ef4444"
    elif pct >= 40: color = "#f59e0b"
    elif pct >= 20: color = "#facc15"
    else: color = "#10b981"

    return (pct, color)


# ---------------------------------------------
# TREND ANALYSIS
# ---------------------------------------------
def trend_analysis_v7(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]):
    if not previous:
        return {"trend": "no_previous"}

    diffs = {}

    for key, cur in current.items():
        cur_val = cur.get("value")
        prev_val = None

        if isinstance(previous, dict):
            prev_val = previous.get("canonical", {}).get(key, {}).get("value")

        if cur_val is None or prev_val is None:
            continue

        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[key] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": delta,
                "pct_change": pct
            }
        except:
            pass

    return {"trend": diffs or "no_change"}


# ---------------------------------------------
# SUPABASE SAVE
# ---------------------------------------------
def save_ai_results(report_id: str, results: Dict[str, Any]):
    if not supabase:
        return
    try:
        supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": results,
            "ai_error": None
        }).eq("id", report_id).execute()
    except Exception as e:
        logger.exception("Supabase save failed: %s", e)


# ---------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------
def process_report(record: Dict[str, Any]):
    report_id = record.get("id", "<no-id>")

    # Load PDF
    try:
        pdf_bytes = download_pdf_from_record(record)
    except Exception as e:
        if supabase:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", report_id).execute()
        return

    # Determine extraction method
    scanned = is_scanned_pdf(pdf_bytes)
    if scanned and HAS_PYTESSERACT:
        text = extract_text_ocr(pdf_bytes)
    else:
        text = extract_text_pypdf(pdf_bytes)

    # Parse text
    parsed = find_values_in_text(text)
    canonical = canonical_map(parsed)

    if not canonical:
        save_ai_results(report_id, {
            "error": "No analytes extracted",
            "parsed": parsed,
            "raw_text_excerpt": text[:4000],
            "scanned": scanned
        })
        return

    # Load previous report for trends (optional)
    previous = None
    if supabase:
        try:
            pid = record.get("patient_id")
            if pid:
                q = supabase.table(SUPABASE_TABLE).select("ai_results").eq("patient_id", pid)\
                    .order("created_at", desc=True).limit(1).execute()
                rows = q.data if hasattr(q, "data") else q
                if rows and rows[0].get("ai_results"):
                    previous = rows[0]["ai_results"]
        except:
            pass

    # Clinical engine
    routes = route_engine_v7(canonical, record)

    # Trends
    trends = trend_analysis_v7(canonical, previous)

    # Decorated analytes for UI
    sex = record.get("sex", "unknown")
    ag = age_group_from_age(record.get("age"))

    decorated = {}
    for key in CANONICAL_KEYS:
        val = canonical.get(key, {})
        value = val.get("value")
        s = score_severity_v7(key, value, ag, sex)
        s_text = severity_text_for_score(s)

        pct, col = risk_bar_v7(key, value)

        decorated[key] = {
            "value": value,
            "raw": val.get("raw"),
            "severity_text": s_text,
            "risk_bar": {"percentage": pct, "color": col}
        }

    # Final result
    results = {
        "canonical": canonical,
        "parsed": parsed,
        "routes": routes,
        "decorated": decorated,
        "trends": trends,
        "raw_text_excerpt": text[:4000],
        "scanned": scanned,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    save_ai_results(report_id, results)


# ---------------------------------------------
# POLLING LOOP
# ---------------------------------------------
def poll_loop():
    if not supabase:
        logger.error("Supabase disabled — cannot poll.")
        return

    while True:
        try:
            q = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status", "pending")\
                .limit(10).execute()
            rows = q.data if hasattr(q, "data") else q

            if not rows:
                time.sleep(POLL_INTERVAL)
                continue

            for r in rows:
                try:
                    supabase.table(SUPABASE_TABLE).update({"ai_status": "processing"})\
                        .eq("id", r["id"]).execute()
                    process_report(r)
                except Exception as e:
                    supabase.table(SUPABASE_TABLE).update({
                        "ai_status": "failed",
                        "ai_error": str(e)
                    }).eq("id", r["id"]).execute()

        except Exception as e:
            logger.exception("Polling error: %s", e)
            time.sleep(POLL_INTERVAL)


# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMI Health Worker V7 — Full Rebuild")
    parser.add_argument("--test-pdf", help="Local PDF path for testing")
    parser.add_argument("--once", action="store_true", help="Run polling loop once")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdfb = f.read()

        dummy = {
            "id": "local-test",
            "patient_id": "local",
            "age": 30,
            "sex": "male",
            "file_path": args.test_pdf
        }

        def dl_override(_): return pdfb
        globals()["download_pdf_from_record"] = dl_override

        process_report(dummy)
        print("\n✔ Test run complete.\n")
    else:
        if args.once:
            poll_loop()
        else:
            poll_loop()
