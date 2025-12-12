#!/usr/bin/env python3
# AMI Health Worker V6 — Production Build
# Digital-first → OCR fallback → Hybrid Parser → Route Engine V6
# Severity = text only, no admission logic. Pure Supabase polling worker.

import os, io, re, time, json, logging
from typing import Dict, Any, Optional, List, Tuple

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

# optional libs
try:
    import pytesseract
    HAS_OCR = True
except:
    HAS_OCR = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except:
    create_client = None
    HAS_SUPABASE = False

# ------------------------------------------------------------------------
# ENV + LOGGING
# ------------------------------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LEN_THRESHOLD = int(os.getenv("TEXT_LEN_THRESHOLD", "80"))
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WorkerV6] %(levelname)s — %(message)s"
)
logger = logging.getLogger("worker-v6")

supabase = None
if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized.")
    except Exception as e:
        logger.error("Supabase init failed: %s", e)
else:
    logger.warning("Supabase credentials missing or library unavailable.")


# ------------------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------------------
def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    s = str(x).replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s in ("", "-", ".", "-."):
        return None
    try:
        return float(s)
    except:
        return None

def merge_digits(s: str) -> str:
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r"(?<=\.)\s+(?=\d)", "", s)
    return s

def current_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


# ------------------------------------------------------------------------
# PDF EXTRACTION
# ------------------------------------------------------------------------
def extract_text_digital(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n".join(pages)
    except:
        return ""

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.MedianFilter(3))
    w,h = img.size
    if max(w,h) < 1600:
        scale = 1600 // max(w,h)
        img = img.resize((w*scale, h*scale), Image.LANCZOS)
    return img

def extract_text_ocr(pdf_bytes: bytes) -> str:
    if not HAS_OCR:
        return ""

    try:
        pages = convert_from_bytes(pdf_bytes, dpi=PDF_RENDER_DPI)
    except Exception as e:
        logger.error("OCR render failed: %s", e)
        return ""

    out = []
    for idx, img in enumerate(pages):
        try:
            clean = preprocess_image(img)
            txt = pytesseract.image_to_string(clean, lang="eng", config="--oem 3 --psm 6")
            out.append(txt)
        except Exception as e:
            logger.error("OCR failed on page %d: %s", idx, e)
            out.append("")
    return "\n".join(out)

def is_scanned(text: str) -> bool:
    if len(text.strip()) < TEXT_LEN_THRESHOLD:
        return True
    if len(re.findall(r"\d", text)) < 3:
        return True
    return False


# ------------------------------------------------------------------------
# LABEL MAP + CANONICAL KEYS
# ------------------------------------------------------------------------
LABEL_MAP = {
    "hb":"Hb","haemoglobin":"Hb","hemoglobin":"Hb","hgb":"Hb",
    "rbc":"RBC","erythrocyte":"RBC",
    "hct":"HCT","haematocrit":"HCT",
    "mcv":"MCV","mch":"MCH","mchc":"MCHC","rdw":"RDW",

    "wbc":"WBC","white cell":"WBC","white blood":"WBC","leukocyte":"WBC","leucocyte":"WBC",

    "neutrophil":"Neutrophils","neut":"Neutrophils",
    "lymphocyte":"Lymphocytes","lymph":"Lymphocytes",
    "monocyte":"Monocytes","eosinophil":"Eosinophils","basophil":"Basophils",

    "platelet":"Platelets","plt":"Platelets",

    "crp":"CRP","c-reactive":"CRP",
    "creatinine":"Creatinine","creat":"Creatinine",
    "sodium":"Sodium","na ":"Sodium",
    "potassium":"Potassium","k ":"Potassium",
    "chloride":"Chloride","cl ":"Chloride",

    "urea":"Urea",
    "alt":"ALT","ast":"AST",
    "ck":"CK","ck-mb":"CKMB","ckmb":"CKMB",

    "calcium":"Calcium","ca ":"Calcium",
    "calcium adj":"CalciumAdj","ca_adj":"CalciumAdj",

    "albumin":"Albumin",
    "co2":"CO2","bicarb":"CO2",
    "egfr":"eGFR",
}

CANONICAL_KEYS = [
    "Hb","MCV","MCH","MCHC","RDW","WBC","Neutrophils","Lymphocytes","Monocytes",
    "Eosinophils","Basophils","NLR","Platelets","RBC","HCT","Creatinine","CRP",
    "Sodium","Potassium","Chloride","Urea","ALT","AST","CK","CKMB",
    "Calcium","CalciumAdj","Albumin","CO2","eGFR"
]

def match_label(lbl: str) -> Optional[str]:
    s = lbl.lower()
    s = re.sub(r"[^a-z0-9\- ]", " ", s)
    for key, canon in LABEL_MAP.items():
        if key in s:
            return canon
    return None


# ------------------------------------------------------------------------
# PARSER
# ------------------------------------------------------------------------
VALUE_RE = r"(-?\d{1,7}\.\d+|-?\d{1,7})"
PERCENT_RE = r"([0-9]{1,3}\.?\d*)\s*%"

def extract_numbers(line: str) -> List[str]:
    nums = re.findall(VALUE_RE, line)
    if nums:
        return nums
    pct = re.findall(PERCENT_RE, line)
    if pct:
        return pct
    return []

def parse_table_first(text: str) -> Dict[str, Dict[str, Any]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = {}

    for ln in lines:
        ln = merge_digits(ln)
        parts = re.split(r"\s{2,}", ln)
        label = parts[0].strip() if parts else ""
        key = match_label(label)
        if not key:
            continue

        nums = extract_numbers(ln)
        if nums:
            out[key] = {"value": safe_float(nums[0]), "raw": ln}

    return out

def parse_free(text: str) -> Dict[str, Dict[str, Any]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out = {}
    for ln in lines:
        low = ln.lower()
        for k in LABEL_MAP:
            if k in low:
                key = LABEL_MAP[k]
                nums = extract_numbers(ln)
                if nums:
                    out[key] = {"value": safe_float(nums[0]), "raw": ln}
                break
    return out

def hybrid_parse(text: str) -> Dict[str, Dict[str, Any]]:
    text = merge_digits(text)
    table = parse_table_first(text)
    free  = parse_free(text)

    out = {}
    for k in CANONICAL_KEYS:
        if k in table:
            out[k] = {"value": table[k]["value"]}
        elif k in free:
            out[k] = {"value": free[k]["value"]}
        else:
            out[k] = {"value": None}

    try:
        neut = out["Neutrophils"]["value"]
        lymph = out["Lymphocytes"]["value"]
        if neut and lymph and lymph != 0:
            out["NLR"] = {"value": round(neut / lymph, 2)}
    except:
        pass

    return out


# ------------------------------------------------------------------------
# SEVERITY / FLAGS
# ------------------------------------------------------------------------
def reference_range(key: str, sex: str) -> Tuple[Optional[float], Optional[float]]:
    sex = sex.lower()

    if key == "Hb":
        return (12.0, 15.5) if sex == "female" else (13.0, 17.5)
    if key == "WBC": return (4,11)
    if key == "Platelets": return (150,450)
    if key == "CRP": return (0,10)
    if key == "Creatinine":
        return (45,120) if sex == "female" else (60,130)
    if key == "Sodium": return (135,145)
    if key == "Potassium": return (3.5,5.1)
    if key == "ALT": return (0,34 if sex=="female" else 40)
    if key == "AST": return (0,34 if sex=="female" else 40)
    if key == "CK": return (0,200 if sex=="female" else 250)
    if key == "CKMB": return (0,7)

    return (None,None)

def severity_text(key: str, value: Optional[float], sex: str) -> str:
    if value is None:
        return "normal"
    low,high = reference_range(key, sex)
    if low is None:
        return "normal"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


# ------------------------------------------------------------------------
# ROUTE ENGINE V6
# ------------------------------------------------------------------------
def route_engine_v6(c: Dict[str, Dict[str, Any]], meta: Dict[str, Any], previous=None):
    sex = meta.get("sex","unknown")

    patterns = []
    routes   = []
    next_steps = []
    ddx = []

    def p(x): 
        if x not in patterns: patterns.append(x)
    def r(x):
        if x not in routes: routes.append(x)
    def d(x):
        if x not in ddx: ddx.append(x)
    def step(x):
        if x not in next_steps: next_steps.append(x)

    Hb = c["Hb"]["value"]
    MCV = c["MCV"]["value"]
    WBC = c["WBC"]["value"]
    CRP = c["CRP"]["value"]
    Neut = c["Neutrophils"]["value"]
    NLR  = c["NLR"]["value"]
    CK = c["CK"]["value"]
    CKMB = c["CKMB"]["value"]
    Creat = c["Creatinine"]["value"]
    Plate = c["Platelets"]["value"]

    # anemia
    if Hb is not None:
        sev = severity_text("Hb", Hb, sex)
        if sev != "normal":
            p("anemia")
            if MCV and MCV < 80:
                p("microcytic anemia")
                r("Iron deficiency route")
                d("Iron deficiency anemia")
            elif MCV and MCV > 100:
                p("macrocytic anemia")
                r("Macrocytic route")
                d("B12/Folate deficiency")
            else:
                p("normocytic anemia")
                r("Normocytic anemia route")
                d("Anaemia of inflammation")

    # infection / inflammation
    if WBC and WBC > 11:
        p("leukocytosis")
        r("Bacterial infection / Sepsis route")
        d("Bacterial infection")
        step("Clinical assessment for sepsis; blood cultures if unwell.")

    if Neut and Neut >= 70:
        p("neutrophilic predominance")

    if CRP and CRP > 10:
        p("elevated CRP")
        if CRP > 50:
            r("Significant inflammatory response")
            d("Severe infection / inflammatory disease")

    if NLR and NLR > 10:
        p("very high NLR")
        r("High NLR route")
        step("Urgent clinical review for sepsis if clinically unwell.")

    # rhabdomyolysis
    if CK and CK > 1000:
        p("rhabdomyolysis signal")
        r("Rhabdomyolysis route")
        d("Rhabdomyolysis")
        step("Check creatinine/electrolytes; consider IV fluids.")

    # myocardial
    if CKMB and CKMB > 7:
        p("myocardial injury signal")
        r("Myocardial injury route")
        d("Possible myocardial injury")
        step("ECG + troponin if chest symptoms.")

    # kidneys
    if Creat and Creat > 120:
        p("elevated creatinine")
        r("AKI route")
        d("Acute kidney injury")
        step("Repeat creatinine; check urine output; review meds.")

    # thrombocytopenia
    if Plate and Plate < 150:
        p("thrombocytopenia")
        d("ITP / marrow suppression / DIC")
        step("Repeat platelets; assess bleeding.")

    # severity summarisation
    numeric_scores = []
    for key in CANONICAL_KEYS:
        val = c[key]["value"]
        sev = severity_text(key, val, sex)
        if sev == "low" or sev == "high":
            numeric_scores.append(3)  # mild-mod default
    combined = max(numeric_scores) if numeric_scores else 1

    if combined <= 1:
        sev = "normal"
    elif combined == 2:
        sev = "mild"
    elif combined == 3:
        sev = "moderate"
    elif combined == 4:
        sev = "severe"
    else:
        sev = "critical"

    urgency = "low" if sev in ("normal","mild") else ("medium" if sev=="moderate" else "high")
    color = "#10b981" if sev=="normal" else ("#facc15" if sev=="mild" else ("#f59e0b" if sev=="moderate" else ("#ef4444" if sev=="severe" else "#b91c1c")))

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "severity_text": sev,
        "urgency_flag": urgency,
        "color": color
    }


# ------------------------------------------------------------------------
# TRENDS
# ------------------------------------------------------------------------
def trend_analysis(current, previous):
    if not previous:
        return {"trend": "no_previous"}

    diffs = {}
    for k,v in current.items():
        prev = previous.get("canonical", {}).get(k, {}).get("value") if isinstance(previous, dict) else None
        cur = v.get("value")
        if prev is None or cur is None:
            continue
        try:
            delta = cur - prev
            pct = (delta/prev)*100 if prev != 0 else None
            diffs[k] = {"previous": prev, "current": cur, "delta": delta, "pct_change": pct}
        except:
            pass

    return {"trend": diffs or "no_change"}


# ------------------------------------------------------------------------
# SUPABASE DOWNLOAD + SAVE
# ------------------------------------------------------------------------
def download_pdf(record) -> bytes:
    # Option 1: Direct URL
    if record.get("pdf_url"):
        import requests
        r = requests.get(record["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    # Option 2: Supabase storage
    if supabase and record.get("file_path"):
        path = record["file_path"]
        res = supabase.storage.from_(SUPABASE_BUCKET).download(path)
        if hasattr(res, "data"):
            return res.data
        return res

    raise ValueError("No PDF source available (pdf_url or file_path missing).")


def save_results(report_id: str, ai: Dict[str, Any]):
    if not supabase:
        logger.warning("Supabase not configured; skipping save.")
        return

    try:
        supabase.table(SUPABASE_TABLE).update({
            "ai_status": "completed",
            "ai_results": ai,
            "ai_error": None
        }).eq("id", report_id).execute()
    except Exception as e:
        logger.error("Supabase save failed: %s", e)


# ------------------------------------------------------------------------
# MAIN PROCESS FUNCTION
# ------------------------------------------------------------------------
def process_record(record):
    rid = record.get("id")
    logger.info(f"Processing report {rid}")

    try:
        pdf_bytes = download_pdf(record)
    except Exception as e:
        logger.error("PDF download failed: %s", e)
        if supabase:
            supabase.table(SUPABASE_TABLE).update({
                "ai_status": "failed",
                "ai_error": str(e)
            }).eq("id", rid).execute()
        return

    digital = extract_text_digital(pdf_bytes)
    scanned = is_scanned(digital)

    if scanned:
        logger.info("PDF appears scanned — using OCR.")
        ocr = extract_text_ocr(pdf_bytes)
        text = ocr if len(ocr.strip()) > 10 else digital
    else:
        logger.info("PDF appears digital — using digital text.")
        text = digital

    parsed = hybrid_parse(text)

    # fetch previous results
    previous = None
    if supabase and record.get("patient_id"):
        try:
            q = supabase.table(SUPABASE_TABLE)\
                .select("ai_results")\
                .eq("patient_id", record["patient_id"])\
                .order("created_at", desc=True)\
                .limit(1).execute()

            rows = q.data if hasattr(q, "data") else q
            if rows:
                previous = rows[0].get("ai_results")
        except:
            pass

    # trends
    trends = trend_analysis(parsed, previous)

    # route engine
    meta = {"age": record.get("age"), "sex": record.get("sex","unknown")}
    routes = route_engine_v6(parsed, meta, previous)

    # decorate
    decorated = {}
    sex = record.get("sex","unknown")
    for k in CANONICAL_KEYS:
        val = parsed[k]["value"]
        sev = severity_text(k, val, sex)
        flag = "normal"
        color = "#ffffff"
        if sev == "low":
            flag = "low"; color="#f59e0b"
        elif sev == "high":
            flag = "high"; color="#b91c1c"

        decorated[k] = {
            "value": val,
            "unit": None,
            "flag": flag,
            "color": color,
            "severity_text": sev
        }

    ai = {
        "canonical": parsed,
        "decorated": decorated,
        "routes": routes,
        "trends": trends,
        "raw_text_excerpt": text[:5000],
        "scanned": scanned,
        "processed_at": current_ts()
    }

    save_results(rid, ai)


# ------------------------------------------------------------------------
# POLLING LOOP (PRODUCTION)
# ------------------------------------------------------------------------
def poll_loop():
    if not supabase:
        logger.error("Supabase not configured — worker cannot run.")
        return

    logger.info(f"Starting polling loop (interval {POLL_INTERVAL}s)...")

    while True:
        try:
            q = supabase.table(SUPABASE_TABLE)\
                .select("*")\
                .eq("ai_status","pending")\
                .limit(5).execute()

            rows = q.data if hasattr(q,"data") else q
            if rows:
                for rec in rows:
                    rid = rec.get("id")
                    try:
                        supabase.table(SUPABASE_TABLE).update({
                            "ai_status": "processing"
                        }).eq("id", rid).execute()
                        process_record(rec)
                    except Exception as e:
                        logger.error("Error processing report %s: %s", rid, e)
                        supabase.table(SUPABASE_TABLE).update({
                            "ai_status": "failed",
                            "ai_error": str(e)
                        }).eq("id", rid).execute()
            else:
                logger.debug("No pending reports.")

        except Exception as e:
            logger.error("Polling error: %s", e)

        time.sleep(POLL_INTERVAL)


# ------------------------------------------------------------------------
# START WORKER
# ------------------------------------------------------------------------
if __name__ == "__main__":
    poll_loop()

