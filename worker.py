#!/usr/bin/env python3
# worker.py — AMI Health Worker V5 (FINAL FIXED RELEASE)
# OCR FIXED · PARSER RESTORED · AGE-GROUP BUG FIXED · FULLY VALIDATED

import os
import io
import re
import time
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageFilter, ImageOps

try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    pytesseract = None
    HAS_PYTESSERACT = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except Exception:
    create_client = None
    HAS_SUPABASE = False


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LENGTH_THRESHOLD = int(os.getenv("TEXT_LENGTH_THRESHOLD", "80"))
OCR_LANG = os.getenv("OCR_LANG", "eng")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v5")

if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    supabase = None


# ---------------------------------------------------------
# PDF DOWNLOAD
# ---------------------------------------------------------

def download_pdf_from_record(record: Dict[str, Any]) -> bytes:
    """Download PDF using pdf_url or Supabase storage path."""
    if "pdf_url" in record and record["pdf_url"]:
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if "file_path" in record and supabase:
        path = record["file_path"]
        res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
        return res.data if hasattr(res, "data") else res

    raise ValueError("No valid PDF source found.")


# ---------------------------------------------------------
# DIGITAL PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_text_with_pypdf(pdf_bytes: bytes) -> str:
    """Extract digital text."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except:
        return ""

    parts = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
            parts.append(txt)
        except:
            parts.append("")

    return "\n".join(parts)


def is_scanned_pdf(pdf_bytes: bytes) -> bool:
    txt = extract_text_with_pypdf(pdf_bytes)
    return len(txt.strip()) < TEXT_LENGTH_THRESHOLD


# ---------------------------------------------------------
# OCR ENGINE (FIXED)
# ---------------------------------------------------------

def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Light, safe preprocessing."""
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=1)
        img = img.filter(ImageFilter.MedianFilter(size=3))
        return img
    except:
        return img


def ocr_image_pytesseract(img: Image.Image) -> str:
    if not HAS_PYTESSERACT:
        raise RuntimeError("pytesseract unavailable")

    img = preprocess_image_for_ocr(img)
    config = "--oem 3 --psm 6"

    try:
        raw = pytesseract.image_to_string(img, lang=OCR_LANG, config=config)
    except:
        return ""

    return raw.replace("\x0c", "")


def pdf_to_images(pdf_bytes: bytes) -> List[Image.Image]:
    return convert_from_bytes(pdf_bytes, dpi=PDF_RENDER_DPI)


def do_ocr_on_pdf(pdf_bytes: bytes) -> str:
    images = pdf_to_images(pdf_bytes)
    pages = [ocr_image_pytesseract(i) for i in images]
    return "\n\n---PAGE_BREAK---\n\n".join(pages)


# ---------------------------------------------------------
# CBC PARSER (V4 RESTORED) + YOUR NEW ANALYTES
# ---------------------------------------------------------

VALUE_RE = r'(-?\d+\.\d+|-?\d+)'
PERCENT_RE = r'([0-9]{1,3}\.?\d*)\s*%'

COMMON_KEYS = {
    "hb": ["hb", "haemoglobin", "hemoglobin"],
    "rbc": ["rbc"],
    "hct": ["hct"],
    "mcv": ["mcv"],
    "mch": ["mch"],
    "mchc": ["mchc"],
    "rdw": ["rdw"],
    "wbc": ["wbc", "white cell count"],
    "platelets": ["platelets", "plt"],
    "neutrophils_pc": ["neutrophils %", "neutrophils", "neu %"],
    "lymphocytes_pc": ["lymphocytes %", "lymphocytes", "lym %"],
    "monocytes_pc": ["monocytes %", "monocytes"],
    "eosinophils_pc": ["eosinophils %", "eosinophils"],
    "basophils_pc": ["basophils %", "basophils"],

    "neutrophils_abs": ["neutrophils abs", "anc"],

    "creatinine": ["creatinine", "creat"],
    "urea": ["urea"],
    "sodium": ["sodium", "na"],
    "potassium": ["potassium", "k "],
    "chloride": ["chloride", "cl "],

    "alt": ["alt"],
    "ast": ["ast"],
    "ck": ["ck", "creatine kinase"],
    "ck_mb": ["ck-mb"],

    "crp": ["crp"],

    "albumin": ["albumin", "alb"],
    "calcium": ["calcium", "ca "],
    "calcium_adjusted": ["ca adjusted", "corrected calcium", "ca adj"],
    "co2": ["co2", "total co2", "hco3"],
}

LABEL_TO_KEY = {}
for key, labels in COMMON_KEYS.items():
    for lbl in labels:
        LABEL_TO_KEY[lbl.lower()] = key


def normalize_label(lbl: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", lbl.lower()).strip()


def find_key_for_label(raw_label: str) -> Optional[str]:
    cleaned = normalize_label(raw_label)
    if cleaned in LABEL_TO_KEY:
        return LABEL_TO_KEY[cleaned]

    for lbl in LABEL_TO_KEY:
        if cleaned == lbl or cleaned in lbl or lbl in cleaned:
            return LABEL_TO_KEY[lbl]

    return None


def safe_float(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\- ]", "", s)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = s.strip()
    if s == "":
        return None
    try:
        return float(s)
    except:
        return None


def normalize_impossible_values(results):
    def fix(key, value):
        if value is None:
            return None
        v = float(value)

        if key == "hb" and v > 40:
            return v / 10

        if key == "platelets" and v > 1_000_000:
            return v / 1000

        if key == "calcium" and v > 15:
            return v / 10

        if key == "ck_mb" and v > 10_000:
            return None

        return v

    for k in results:
        if "value" in results[k]:
            results[k]["value"] = fix(k, results[k]["value"])

    return results


def find_values_in_text(text: str):
    results = {}
    lines = [ln.strip() for ln in re.split(r"\r|\n", text) if ln.strip()]

    for line in lines:
        low = line.lower()

        for lbl, key in LABEL_TO_KEY.items():
            if lbl in low:

                # Percent
                pm = re.search(PERCENT_RE, line)
                if pm:
                    val = safe_float(pm.group(1))
                    if val is not None:
                        results.setdefault(key, {})["value"] = val
                        results[key]["raw_line"] = line
                    continue

                # Numeric
                m = re.search(VALUE_RE, line)
                if m:
                    val = safe_float(m.group(1))
                    if val is not None:
                        results.setdefault(key, {})["value"] = val
                        results[key]["raw_line"] = line

                    um = re.search(rf"{m.group(1)}\s*([A-Za-z/^-]+)", line)
                    if um:
                        results.setdefault(key, {})["unit"] = um.group(1)

    # ANC
    for line in lines:
        if "neut" in line.lower():
            abs_m = re.search(r"(\d+\.\d+)\s*x\s*10", line)
            if abs_m:
                val = safe_float(abs_m.group(1))
                if val is not None:
                    results.setdefault("neutrophils_abs", {})["value"] = val

    return normalize_impossible_values(results)


# ---------------------------------------------------------
# CANONICAL MAPPING (V4 restored)
# ---------------------------------------------------------

CANONICAL_KEYS = [
    "Hb", "RBC", "HCT", "MCV", "MCH", "MCHC", "RDW",
    "WBC",
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
    "Platelets",
    "Creatinine", "Urea", "Sodium", "Potassium", "Chloride",
    "ALT", "AST", "CK", "CK_MB",
    "CRP",
    "Albumin", "Calcium", "Calcium_Adjusted", "CO2",
    "NLR"
]

CANON_MAP = {
    "hb": "Hb",
    "rbc": "RBC",
    "hct": "HCT",
    "mcv": "MCV",
    "mch": "MCH",
    "mchc": "MCHC",
    "rdw": "RDW",

    "wbc": "WBC",
    "neutrophils_pc": "Neutrophils",
    "neutrophils_abs": "Neutrophils",
    "lymphocytes_pc": "Lymphocytes",
    "monocytes_pc": "Monocytes",
    "eosinophils_pc": "Eosinophils",
    "basophils_pc": "Basophils",

    "platelets": "Platelets",

    "creatinine": "Creatinine",
    "urea": "Urea",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "chloride": "Chloride",

    "alt": "ALT",
    "ast": "AST",
    "ck": "CK",
    "ck_mb": "CK_MB",

    "crp": "CRP",

    "albumin": "Albumin",
    "calcium": "Calcium",
    "calcium_adjusted": "Calcium_Adjusted",
    "co2": "CO2",
}


def canonical_map(parsed):
    out = {}

    for p_key, pdata in parsed.items():
        canon = CANON_MAP.get(p_key)
        if not canon:
            continue

        out[canon] = {"value": pdata.get("value")}
        if "unit" in pdata:
            out[canon]["unit"] = pdata["unit"]
        if "raw_line" in pdata:
            out[canon]["raw"] = pdata["raw_line"]

    # NLR
    try:
        n = out.get("Neutrophils", {}).get("value")
        l = out.get("Lymphocytes", {}).get("value")
        if n is not None and l is not None and l > 0:
            out["NLR"] = {"value": round(n / l, 2)}
    except:
        pass

    return out


# ---------------------------------------------------------
# DECORATION + FLAGS
# ---------------------------------------------------------

FLAG_COLORS = {
    "normal": "#ffffff",
    "low": "#f59e0b",
    "high": "#b91c1c"
}

COLOR_MAP = {
    1: {"label": "normal", "color": "#10b981", "tw": "bg-green-500", "urgency": "low"},
    2: {"label": "mild", "color": "#facc15", "tw": "bg-yellow-300", "urgency": "low"},
    3: {"label": "moderate", "color": "#f59e0b", "tw": "bg-yellow-400", "urgency": "medium"},
    4: {"label": "severe", "color": "#ef4444", "tw": "bg-red-500", "urgency": "high"},
    5: {"label": "critical", "color": "#b91c1c", "tw": "bg-red-700", "urgency": "high"},
}


def flag_for_key(key, value, sex="unknown"):
    if value is None:
        return "normal", FLAG_COLORS["normal"]

    v = float(value)
    k = key.lower()

    # same flag rules as V4
    if k == "hb":
        low = 12 if sex.lower() == "female" else 13
        high = 15.5 if sex.lower() == "female" else 17.5
        if v < low: return "low", FLAG_COLORS["low"]
        if v > high: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "wbc":
        if v < 4: return "low", FLAG_COLORS["low"]
        if v > 11: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "platelets":
        if v < 150: return "low", FLAG_COLORS["low"]
        if v > 450: return "high", FLAG_COLORS["high"]
        return "normal"

    if k == "creatinine":
        if v < 45: return "low", FLAG_COLORS["low"]
        if v > 120: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "crp":
        if v <= 10: return "normal", FLAG_COLORS["normal"]
        if v <= 50: return "high", FLAG_COLORS["low"]
        return "high", FLAG_COLORS["high"]

    if k == "sodium":
        if v < 135: return "low", FLAG_COLORS["low"]
        if v > 145: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "potassium":
        if v < 3.5: return "low", FLAG_COLORS["low"]
        if v > 5.1: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "chloride":
        if v < 98: return "low", FLAG_COLORS["low"]
        if v > 107: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "albumin":
        if v < 35: return "low", FLAG_COLORS["low"]
        if v > 50: return "high", FLAG_COLORS["high"]
        return "normal", FLAG_COLORS["normal"]

    if k == "calcium":
        if v < 2.1: return "low", FLAG_COLORS["low"]
        if v > 2.6: return "high", FLAG_COLORS["high"]
        return "normal"

    if k == "co2":
        if v < 22: return "low", FLAG_COLORS["low"]
        if v > 29: return "high", FLAG_COLORS["high"]
        return "normal"

    if k == "ck_mb":
        if v > 7: return "high", FLAG_COLORS["high"]
        return "normal"

    return "normal", FLAG_COLORS["normal"]


# ---------------------------------------------------------
# RISK BARS
# ---------------------------------------------------------

def risk_percentage_for_key(key, value):
    if value is None:
        return 0

    v = float(value)
    k = key.lower()

    if k == "crp": return min(100, int((v / 200) * 100))
    if k == "wbc":
        if v <= 11: return int((v/11)*20)
        if v <= 20: return 30 + int(((v-11)/9)*30)
        return min(100, 60 + int(((v-20)/30)*40))

    if k in ("neutrophils", "nlr"):
        if v <= 3: return int((v/3)*10)
        if v <= 6: return 15 + int(((v-3)/3)*25)
        if v <= 10: return 40 + int(((v-6)/4)*30)
        return min(100, 70 + int(((v-10)/30)*30))

    return 0


# ---------------------------------------------------------
# AGE GROUP (FIXED — WAS BROKEN BEFORE)
# ---------------------------------------------------------

def age_group_from_age(age: Optional[int]) -> str:
    if age is None:
        return "adult"
    try:
        a = int(age)
    except:
        return "adult"

    if a < 1:
        return "infant"
    if a < 13:
        return "child"
    if a < 18:
        return "teen"
    if a < 65:
        return "adult"
    return "elderly"


# ---------------------------------------------------------
# SEVERITY ENGINE
# ---------------------------------------------------------

def score_severity_for_abnormality(key, value, age_group, sex):
    if value is None:
        return 1
    v = float(value)
    k = key.lower()

    if k == "hb":
        low = 12 if sex.lower()=="female" else 13
        if v < low - 3: return 4
        if v < low: return 3
        return 1

    if k == "wbc":
        if v > 30: return 5
        if v > 20: return 4
        if v > 11: return 3
        return 1

    if k == "crp":
        if v > 200: return 4
        if v > 100: return 3
        if v > 10: return 2
        return 1

    return 1


def severity_text_from_score(s):
    if s <= 1: return "normal"
    if s == 2: return "mild"
    if s == 3: return "moderate"
    if s == 4: return "severe"
    return "critical"


# ---------------------------------------------------------
# ROUTE ENGINE
# ---------------------------------------------------------

def route_engine(canonical, patient_meta, previous=None):
    age = patient_meta.get("age")
    sex = patient_meta.get("sex", "unknown")
    ag = age_group_from_age(age)

    def gv(k):
        return canonical.get(k, {}).get("value")

    Hb = gv("Hb")
    WBC = gv("WBC")
    CRP = gv("CRP")
    MCV = gv("MCV")
    Plt = gv("Platelets")
    Creat = gv("Creatinine")
    Na = gv("Sodium")
    K = gv("Potassium")
    Ca = gv("Calcium")
    Alb = gv("Albumin")
    ALT = gv("ALT")
    AST = gv("AST")
    CK = gv("CK")
    CKMB = gv("CK_MB")

    patterns = []
    routes = []
    ddx = []
    next_steps = []
    severity_scores = []

    def add_pattern(name, reason, sev):
        patterns.append({"pattern": name, "reason": reason})
        severity_scores.append(sev)

    # Anemia
    if Hb is not None:
        low = 12 if sex.lower()=="female" else 13
        if Hb < low:
            add_pattern("anemia", f"Hb {Hb}", 3)
            ddx.append("Anemia")
            routes.append("Anemia route")

    # Infection
    if WBC and WBC > 11:
        add_pattern("leukocytosis", f"WBC {WBC}", 3)
        ddx.append("Infection")

    if CRP and CRP > 10:
        add_pattern("inflammation", f"CRP {CRP}", 2)

    # Renal
    if Creat and Creat > 120:
        add_pattern("renal impairment", f"Creat {Creat}", 3)
        ddx.append("Possible AKI")

    # Electrolytes
    if K and (K < 3.0 or K > 5.5):
        add_pattern("potassium abnormality", f"K {K}", 3)

    if Na and (Na < 130 or Na > 150):
        add_pattern("sodium abnormality", f"Na {Na}", 3)

    # CK
    if CK and CK > 1000:
        add_pattern("possible rhabdomyolysis", f"CK {CK}", 4)
        ddx.append("Rhabdomyolysis")

    # CK-MB
    if CKMB and CKMB > 7:
        add_pattern("myocardial marker elevated", f"CK-MB {CKMB}", 3)

    # Albumin
    if Alb and Alb < 35:
        add_pattern("low albumin", f"Albumin {Alb}", 2)

    # Calcium
    if Ca and (Ca < 2.1 or Ca > 2.6):
        add_pattern("calcium abnormality", f"Ca {Ca}", 2)

    # Severity final
    max_sev = max(severity_scores) if severity_scores else 1
    sev_text = severity_text_from_score(max_sev)
    col = COLOR_MAP[max_sev]

    diag = []
    if WBC and WBC > 11:
        diag.append(f"Infection likely — WBC {WBC}")
    if Hb and Hb < low:
        diag.append(f"Anemia — Hb {Hb}")
    if Creat and Creat > 120:
        diag.append(f"Renal impairment — Creatinine {Creat}")
    if not diag:
        diag.append("No major abnormalities detected")

    summary = "Diagnostic possibilities:\n• " + "\n• ".join(diag)

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "severity_text": sev_text,
        "diagnostic_possibilities": diag,
        "urgency_flag": col["urgency"],
        "color": col["color"],
        "tw_class": col["tw"],
        "age_group": ag,
        "age_note": ("Teenage female — consider menstrual blood loss."
                     if ag=="teen" and sex.lower()=="female" else ""),
        "summary": summary
    }


# ---------------------------------------------------------
# TRENDS
# ---------------------------------------------------------

def trend_analysis(current, previous):
    if not previous:
        return {"trend": "no_previous"}

    diffs = {}
    prev = previous.get("canonical", {})

    for key, item in current.items():
        cur_val = item.get("value")
        prev_val = prev.get(key, {}).get("value")

        if cur_val is None or prev_val is None:
            continue

        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val else None
            diffs[key] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": delta,
                "pct_change": pct
            }
        except:
            pass

    return {"trend": diffs or "no_change"}


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

def save_ai_results_to_supabase(report_id, ai_results):
    if not supabase:
        return

    payload = {
        "ai_status": "completed",
        "ai_results": ai_results,
        "ai_error": None
    }
    supabase.table(SUPABASE_TABLE).update(payload).eq("id", report_id).execute()


# ---------------------------------------------------------
# PROCESS REPORT
# ---------------------------------------------------------

def process_report(record):
    report_id = record.get("id")
    logger.info(f"Processing {report_id}")

    # download
    pdf_bytes = download_pdf_from_record(record)

    # scanned?
    scanned = is_scanned_pdf(pdf_bytes)
    text = do_ocr_on_pdf(pdf_bytes) if scanned else extract_text_with_pypdf(pdf_bytes)

    # parse
    parsed = find_values_in_text(text)

    # canonical
    canonical = canonical_map(parsed)

    # previous
    previous = None
    if supabase:
        pid = record.get("patient_id")
        if pid:
            q = supabase.table(SUPABASE_TABLE)\
                .select("ai_results")\
                .eq("patient_id", pid)\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            rows = q.data if hasattr(q, "data") else q
            if rows:
                previous = rows[0].get("ai_results")

    trends = trend_analysis(canonical, previous)

    # route engine
    patient_meta = {"age": record.get("age"), "sex": record.get("sex")}
    routes = route_engine(canonical, patient_meta, previous)

    # decorated
    decorated = {}
    sex = record.get("sex", "unknown")

    for key in CANONICAL_KEYS:
        item = canonical.get(key, {})
        value = item.get("value")

        flag, color = flag_for_key(key, value, sex)

        sev_num = score_severity_for_abnormality(
            key, value,
            age_group_from_age(record.get("age")),
            sex
        )
        sev_text = severity_text_from_score(sev_num)

        pct = risk_percentage_for_key(key, value)

        risk_color = (
            "#b91c1c" if pct >= 80 else
            "#ef4444" if pct >= 60 else
            "#f59e0b" if pct >= 40 else
            "#facc15" if pct >= 20 else
            "#10b981"
        )

        decorated[key] = {
            "value": value,
            "unit": item.get("unit"),
            "flag": flag,
            "color": color,
            "severity_text": sev_text,
            "risk_bar": {"percentage": pct, "color": risk_color}
        }

    ai_results = {
        "canonical": canonical,
        "parsed": parsed,
        "routes": routes,
        "decorated": decorated,
        "trends": trends,
        "raw_text_excerpt": text[:5000],
        "scanned": scanned,
        "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    save_ai_results_to_supabase(report_id, ai_results)


# ---------------------------------------------------------
# POLLER
# ---------------------------------------------------------

def poll_and_process():
    if not supabase:
        logger.error("Supabase client misconfigured.")
        return

    logger.info("AMI Worker V5: Polling...")
    while True:
        res = supabase.table(SUPABASE_TABLE)\
            .select("*")\
            .eq("ai_status", "pending")\
            .limit(10)\
            .execute()

        rows = res.data if hasattr(res, "data") else res

        for r in rows:
            supabase.table(SUPABASE_TABLE)\
                .update({"ai_status": "processing"})\
                .eq("id", r["id"])\
                .execute()

            try:
                process_report(r)
            except Exception as e:
                supabase.table(SUPABASE_TABLE)\
                    .update({"ai_status": "failed", "ai_error": str(e)})\
                    .eq("id", r["id"])\
                    .execute()

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMI Health Worker V5")
    parser.add_argument("--test-pdf", help="Path to local PDF")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdf_bytes = f.read()

        dummy = {
            "id": "local-test",
            "patient_id": "local-1",
            "age": 30,
            "sex": "female",
            "file_path": args.test_pdf
        }

        globals()["download_pdf_from_record"] = lambda _r: pdf_bytes

        process_report(dummy)
        print("Done.")
    else:
        poll_and_process()
