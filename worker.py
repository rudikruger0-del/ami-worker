#!/usr/bin/env python3
"""
AMI Worker v4

Features:
- handles scanned & digital PDFs
- prefers local OCR via pytesseract (if available)
- OpenAI fallback (chat completions) for OCR if pytesseract not available
- robust CBC + chemistry parsing (best-effort)
- Route Engine v4: patterns -> routes -> next_steps
- severity (1-5), urgency (low/med/high), simple differential trees, trend comparison
- defensive logging for debugging in Railway logs

Before using local OCR:
- Add pytesseract to requirements.txt
- Install tesseract-ocr in Dockerfile (apt-get install -y tesseract-ocr)
"""

import os
import time
import io
import json
import re
import traceback
import base64
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports (must be present in your runtime)
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

# Try local OCR via pytesseract if available
USE_PYTESSACT = False
try:
    import pytesseract  # type: ignore
    USE_PYTESSACT = True
    print(">>> pytesseract available — local OCR enabled")
except Exception as e:
    print(">>> pytesseract not available; will use OpenAI Vision fallback for OCR:", e)

# -------------------------
# Environment + Clients
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set — OpenAI fallback will fail if used")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

# -------------------------
# Utilities
# -------------------------
def debug(*args, **kwargs):
    print("DEBUG:", *args, **kwargs)

def clean_number(val: Any) -> Optional[float]:
    """Convert strings like '88.0%', '11,6 g/dL', '4.2*' to float, else None"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    if s.strip() == "":
        return None
    s = s.replace(",", ".")
    # look for -?digits[.digits]
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def severity_from_routes(routes: List[str]) -> int:
    """Simple severity grading: 1 (mild) ... 5 (critical)"""
    score = 1
    for r in routes:
        if "urgent" in r.lower() or "arrhythmia" in r.lower() or "rhabdomyolysis" in r.lower():
            score = max(score, 5)
        elif "renal" in r.lower() or "creatinine" in r.lower():
            score = max(score, 4)
        elif "infection" in r.lower() or "inflammation" in r.lower():
            score = max(score, 3)
        elif "anaemia" in r.lower() or "iron" in r.lower():
            score = max(score, 2)
    return score

def urgency_flag_from_severity(sev: int) -> str:
    if sev >= 5:
        return "high"
    if sev >= 3:
        return "medium"
    return "low"

# -------------------------
# PDF helpers
# -------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract selectable text from PDF using pypdf"""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        debug("pypdf parse error:", e)
        return ""

def is_scanned_pdf(pdf_text: str) -> bool:
    """Heuristic: if no text or too short -> scanned"""
    if not pdf_text:
        return True
    return len(pdf_text.strip()) < 30

# -------------------------
# Local OCR via pytesseract
# -------------------------
def ocr_image_local(img: Image.Image) -> str:
    """Use pytesseract to extract text from a PIL image"""
    try:
        # convert to grayscale for better OCR
        g = img.convert("L")
        # optionally resize if too small
        w, h = g.size
        if w < 800:
            g = g.resize((int(w * 2), int(h * 2)), Image.LANCZOS)
        text = pytesseract.image_to_string(g)
        return text or ""
    except Exception as e:
        debug("pytesseract OCR error:", e)
        return ""

# -------------------------
# OpenAI OCR fallback (chat completions)
# -------------------------
def ocr_image_openai_fallback(img: Image.Image) -> Optional[Dict[str, Any]]:
    """
    Fallback OCR that sends a resized/optimized base64 image to OpenAI chat completions.
    Returns parsed JSON dict { 'cbc': [ {analyte, value, units, reference_low, reference_high} ] } OR None
    NOTE: This is fallback and may be limited by token/use restrictions.
    """
    try:
        # Resize to reasonable size and encode jpeg to reduce base64 length
        MAX_DIM = 1600  # keep within limits
        w, h = img.size
        scale = min(1.0, float(MAX_DIM) / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        b = buf.getvalue()
        b64 = base64.b64encode(b).decode("utf-8")

        # Build compact system prompt
        system_prompt = (
            "You are an OCR assistant extracting laboratory analyte tables from a lab report image. "
            "Return STRICT JSON: { 'cbc': [ { 'analyte': '', 'value': '', 'units': '', 'reference_low':'', 'reference_high':'' } ] } "
            "Return only JSON."
        )

        # Use chat completions create (openai==1.51.0)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                # Use image_url with a data URI object (best-effort for this OpenAI lib/version).
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # "url" key expected to be a string URI; here we pass data:image/jpeg;base64,...
                            "url": f"data:image/jpeg;base64,{b64}"
                        }
                    }
                ],
            },
        ]

        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # In some returns the content JSON is in resp.choices[0].message.content
        raw = None
        try:
            raw = resp.choices[0].message.content
        except Exception:
            try:
                raw = resp.choices[0].message.parsed  # earlier patterns
            except Exception:
                # fallback to printing whole response for debugging
                debug("OpenAI OCR resp full:", getattr(resp, "__dict__", resp))
                return None

        # raw might be string or already JSON object; try to parse safely
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed
            except Exception:
                debug("Could not parse raw string OCR JSON:", raw[:200])
                return None
        if isinstance(raw, dict):
            return raw
        # if it's a list or other structure, try to locate the json field
        try:
            # Try common place
            parsed = getattr(resp.choices[0].message, "parsed", None)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        debug("OpenAI OCR returned unknown structure:", type(raw))
        return None

    except Exception as e:
        debug("OpenAI OCR exception:", e)
        return None

# -------------------------
# Table extraction heuristics
# -------------------------
def parse_cbc_rows_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Heuristic parser: tries to find common analytes and values from OCR/text.
    Returns list of dicts {analyte, value, units, reference_low, reference_high}
    This is intentionally permissive (we then clean numbers).
    """
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows

    # normalize whitespace
    text = text.replace("\r", "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # common analytes keywords we look for
    analyte_keywords = [
        "haemoglobin", "hemoglobin", "hb", "hematocrit", "hct",
        "rbc", "erythrocyte", "mcv", "mch", "mchc", "rdw",
        "wbc", "leucocyte", "leukocyte", "neutrophil", "lymphocyte",
        "monocyte", "eosinophil", "basophil", "platelet", "platelets",
        "sodium", "potassium", "chloride", "creatinine", "urea", "crp",
        "alt", "ast", "alkaline phosphatase", "alp", "ggt", "bilirubin",
        "ck-mb", "ck", "glucose", "bicarbonate", "co2"
    ]
    # Build regex to capture "name ... value (ref low - ref high)" patterns
    # Examples we try to match:
    # "Haemoglobin: 11.6 g/dL (ref: 12.4-16.7) — low"
    # "MCV 78.9 fL (ref: 79−100) — low"
    pattern = re.compile(
        r"(?P<name>[A-Za-z %/\-\+()0-9]+?)\s*(?:[:\-])?\s*(?P<value>-?\d+[\.,]?\d*\%?)\s*(?P<units>[a-zA-Z/^\.\%\d°]*)\s*(?:\(?ref[:=]?\s*(?P<ref_low>-?\d+[\.,]?\d*)\s*[-–]\s*(?P<ref_high>-?\d+[\.,]?\d*)\)?)?",
        re.IGNORECASE
    )

    # Try scanning each line for analytes or matches
    for ln in lines:
        # quick check whether common analyte referenced
        low = ln.lower()
        if any(k in low for k in analyte_keywords):
            # try pattern
            m = pattern.search(ln)
            if m:
                name = m.group("name").strip()
                value = m.group("value").strip()
                units = (m.group("units") or "").strip()
                ref_low = (m.group("ref_low") or "").strip()
                ref_high = (m.group("ref_high") or "").strip()
                rows.append({
                    "analyte": name,
                    "value": value,
                    "units": units,
                    "reference_low": ref_low,
                    "reference_high": ref_high
                })
                continue
            # fallback: split by spaces and try to find numeric token
            tokens = ln.split()
            numeric = None
            for t in tokens:
                if re.search(r"-?\d+[\.,]?\d*\%?", t):
                    numeric = t
                    break
            if numeric:
                # approximated analyte name is first few tokens
                name = " ".join(tokens[:min(3, len(tokens))])
                rows.append({
                    "analyte": name,
                    "value": numeric,
                    "units": "",
                    "reference_low": "",
                    "reference_high": ""
                })
    return rows

# -------------------------
# Build canonical CBC dict for routing
# -------------------------
def build_cbc_value_dict(extracted_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Map heterogeneous analyte names to canonical keys used in route engine:
    'Hb', 'MCV','MCH','WBC','Neutrophils','Lymphocytes','Platelets','Creatinine', etc.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for r in extracted_rows:
        name = (r.get("analyte") or "").lower()
        if not name:
            continue
        def put(k):
            if k not in out:
                out[k] = r

        if "haemoglobin" in name or "hemoglobin" in name or name.strip() == "hb":
            put("Hb")
        elif name.startswith("mcv"):
            put("MCV")
        elif name.startswith("mchc") or "mchc" in name:
            put("MCHC")
        elif name.startswith("mch"):
            put("MCH")
        elif "rdw" in name:
            put("RDW")
        elif "rbc" in name or "erythro" in name:
            put("RBC")
        elif "hematocrit" in name or "hct" in name:
            put("Hct")
        elif "white" in name or "wbc" in name or "leukocyte" in name or "leucocyte" in name:
            put("WBC")
        elif "neutroph" in name:
            put("Neutrophils")
        elif "lymph" in name:
            put("Lymphocytes")
        elif "monocyt" in name:
            put("Monocytes")
        elif "eosin" in name:
            put("Eosinophils")
        elif "basoph" in name:
            put("Basophils")
        elif "platelet" in name or "plt" in name:
            put("Platelets")
        elif "creatinine" in name or name.strip() == "crea":
            put("Creatinine")
        elif name.startswith("urea"):
            put("Urea")
        elif "sodium" in name or name.strip() == "na":
            put("Sodium")
        elif "potassium" in name or name.strip() == "k":
            put("Potassium")
        elif "chloride" in name:
            put("Chloride")
        elif "crp" in name:
            put("CRP")
        elif "alt" in name or "alanine" in name:
            put("ALT")
        elif "ast" in name or "aspartate" in name:
            put("AST")
        elif "alkaline" in name or "alp" in name:
            put("ALP")
        elif "ggt" in name:
            put("GGT")
        elif "bilirubin" in name:
            put("Bilirubin")
        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")
        elif name.strip() == "ck":
            put("CK")
        # else keep a general fallback by short name token
        else:
            # small normalization: first token
            tk = name.split()[0].upper()
            put(tk)
    return out

# -------------------------
# Route Engine v4
# -------------------------
def generate_clinical_routes_v4(cbc_values: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Returns dictionary:
    {
      "patterns": [...],
      "routes": [...],
      "next_steps": [...],
      "severity": int (1-5),
      "urgency": "low|medium|high",
      "differential": [...],
      "trend": "improving|worse|stable|no_data"
    }
    """
    def v(key):
        return clean_number(cbc_values.get(key, {}).get("value"))

    Hb = v("Hb")
    MCV = v("MCV")
    MCH = v("MCH")
    MCHC = v("MCHC")
    RDW = v("RDW")
    WBC = v("WBC")
    Neut = v("Neutrophils")
    Lymph = v("Lymphocytes")
    Plt = v("Platelets")
    Cr = v("Creatinine")
    Urea = v("Urea")
    CRP = v("CRP")
    Na = v("Sodium")
    K = v("Potassium")
    ALT = v("ALT")
    AST = v("AST")
    Bili = v("Bilirubin")
    CK = v("CK")
    CKMB = v("CK-MB")

    patterns = []
    routes = []
    next_steps = []
    differential = []

    # Anaemia patterns
    if Hb is not None:
        if Hb < 8:
            patterns.append("Severe anaemia")
            routes.append("Urgent anaemia route - consider transfusion thresholds, haemodynamic assessment")
            next_steps.append("Immediate clinical review; order repeat CBC, crossmatch if bleeding or symptomatic")
            differential.extend(["Acute blood loss", "Severe iron deficiency", "haemolysis", "bone marrow failure"])
        elif Hb < 11:
            patterns.append("Anaemia")
            if MCV is not None:
                if MCV < 80:
                    patterns.append("Microcytic anaemia")
                    routes.append("Microcytic anaemia route")
                    next_steps.append("Order ferritin, iron studies, reticulocyte count")
                    differential.extend(["Iron deficiency", "Thalassaemia trait", "Chronic blood loss"])
                elif 80 <= MCV <= 100:
                    patterns.append("Normocytic anaemia")
                    routes.append("Normocytic anaemia route")
                    next_steps.append("Consider inflammation, renal disease; check renal function and CRP")
                    differential.extend(["Anaemia of chronic disease", "early iron deficiency", "haemolysis"])
                else:
                    patterns.append("Macrocytic anaemia")
                    routes.append("Macrocytic anaemia route")
                    next_steps.append("Check B12, folate, review medications and liver function")
                    differential.extend(["B12/folate deficiency", "alcohol/liver disease", "drugs"])
            else:
                next_steps.append("MCV missing — request RBC indices (MCV, MCH, MCHC)")

    # Leukocyte patterns
    if WBC is not None:
        if WBC > 12:
            patterns.append("Leukocytosis")
            routes.append("Inflammatory / infective route")
            if Neut and Neut > 70:
                patterns.append("Neutrophil predominance")
                routes.append("Bacterial pattern more likely")
                differential.extend(["Bacterial infection", "stress response", "steroid effect"])
            if Lymph and Lymph > 45:
                patterns.append("Lymphocytosis")
                routes.append("Consider viral infection or recovery phase")
                differential.extend(["Viral infection", "pertussis", "chronic lymphocytic processes"])
        elif WBC < 4:
            patterns.append("Leukopenia")
            routes.append("Consider bone marrow suppression, viral, drug causes")
            differential.extend(["Viral suppression", "bone marrow failure", "drug toxicity"])

    # Platelets
    if Plt is not None:
        if Plt < 50:
            patterns.append("Severe thrombocytopenia")
            routes.append("Urgent bleeding risk route")
            next_steps.append("Immediate clinical review; check for bleeding, repeat platelet count")
            differential.extend(["ITP", "DIC", "sepsis", "drug-induced"])
        elif Plt < 150:
            patterns.append("Mild thrombocytopenia")
            routes.append("Investigate for infection, consumptive or marrow causes")
        elif Plt > 450:
            patterns.append("Thrombocytosis")
            routes.append("Reactive thrombocytosis vs primary process")
            differential.extend(["Reactive (infection/inflammation/iron deficiency)", "myeloproliferative disorder"])

    # Inflammation & CRP
    if CRP is not None:
        if CRP > 100:
            patterns.append("Very high CRP")
            routes.append("High inflammatory burden — urgent assessment for sepsis/serious infection")
            next_steps.append("Blood cultures, start antibiotics if sepsis suspected")
            differential.extend(["Sepsis", "severe bacterial infection", "major inflammatory process"])
        elif CRP > 10:
            patterns.append("Raised CRP")
            routes.append("Active inflammation/infection likely — correlate with WBC/diff")
            next_steps.append("Correlate clinically; consider further infectious workup or imaging")

    # Kidney / Creatinine
    if Cr is not None:
        # creatinine reference varies; we treat >110 umol/L roughly as high for adult average
        if Cr > 120:
            patterns.append("Raised creatinine")
            routes.append("Renal impairment route")
            next_steps.append("Check U&E trend, urine output, medication review, consider nephrology if worsening")
            differential.extend(["AKI (pre-renal, renal, post-renal)", "CKD"])

    # LFTs
    if any(x is not None for x in (ALT, AST, Bili)):
        if (ALT and ALT > 120) or (AST and AST > 120):
            patterns.append("Marked transaminase elevation")
            routes.append("Hepatocellular injury route")
            next_steps.append("Assess for viral hepatitis, drugs, ischemia")
        elif (ALT and ALT > 40) or (AST and AST > 40):
            patterns.append("Mild-moderate LFT elevation")
            routes.append("Consider viral, drugs, alcohol, NAFLD")

    # Electrolytes
    if K is not None:
        if K < 3.3:
            patterns.append("Hypokalaemia")
            routes.append("Risk of arrhythmia — replace potassium if symptomatic")
            next_steps.append("ECG if symptomatic or severe; replace potassium carefully")
        elif K > 5.5:
            patterns.append("Hyperkalaemia")
            routes.append("High arrhythmia risk — urgent ECG and management")
            next_steps.append("Immediate ECG and hyperkalaemia protocol if present")

    if Na is not None:
        if Na < 130:
            patterns.append("Significant hyponatraemia")
            routes.append("Assess fluid status, medications, endocrine causes")
            next_steps.append("Consider urgent review if severe or symptomatic")

    # CK
    if CK is not None and CK > 1000:
        patterns.append("Raised CK")
        routes.append("Rhabdomyolysis concern")
        next_steps.append("Check renal function, urine myoglobin, aggressive fluids")
        differential.extend(["Trauma", "seizure", "exertional rhabdomyolysis", "statin/myopathy"])

    # Compose summary severity & urgency
    sev = severity_from_routes(routes)
    urg = urgency_flag_from_severity(sev)

    # Deduplicate lists, keep order
    def uniq(seq):
        outl = []
        for s in seq:
            if s not in outl:
                outl.append(s)
        return outl

    patterns = uniq(patterns)
    routes = uniq(routes)
    next_steps = uniq(next_steps)
    differential = uniq(differential)

    # Trend comparison (if previous provided — expects previous dict of canonical values)
    trend = "no_data"
    try:
        if previous and isinstance(previous, dict):
            # quick heuristic: if previous has numeric Hb / Cr etc, compare
            diffs = []
            for k in ("Hb", "Creatinine", "CRP", "WBC", "Platelets"):
                prev_val = clean_number(previous.get(k, {}).get("value") if previous.get(k) else None)
                cur_val = clean_number(cbc_values.get(k, {}).get("value") if cbc_values.get(k) else None)
                if prev_val is None or cur_val is None:
                    continue
                if cur_val > prev_val * 1.15:
                    diffs.append((k, "worse"))
                elif cur_val < prev_val * 0.85:
                    diffs.append((k, "improved"))
                else:
                    diffs.append((k, "stable"))
            # simple majority rule
            worse = sum(1 for d in diffs if d[1] == "worse")
            improved = sum(1 for d in diffs if d[1] == "improved")
            if worse > improved:
                trend = "worse"
            elif improved > worse:
                trend = "improving"
            elif diffs:
                trend = "stable"
    except Exception:
        trend = "no_data"

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": differential,
        "severity": sev,
        "urgency": urg,
        "trend": trend
    }

# -------------------------
# Top-level processor
# -------------------------
def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    job is a dict from supabase table 'reports', must contain id and file_path.
    Optionally may contain 'previous_results' for trend comparison.
    """
    report_id = job.get("id")
    file_path = job.get("file_path") or ""
    previous = job.get("previous_results")  # optional canonical dict

    debug(f"Processing report {report_id} file_path={file_path}")

    if not file_path:
        err = "Missing file_path"
        supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Download PDF bytes from Supabase storage
    try:
        pdf_bytes_resp = supabase.storage.from_(BUCKET).download(file_path)
        # supabase client returns an object; the bytes may be in .data
        pdf_bytes = getattr(pdf_bytes_resp, "data", pdf_bytes_resp)
        if isinstance(pdf_bytes, dict) and pdf_bytes.get("error"):
            raise RuntimeError(f"Supabase storage download error: {pdf_bytes}")
        if isinstance(pdf_bytes, memoryview):
            pdf_bytes = pdf_bytes.tobytes()
    except Exception as e:
        err = f"Failed to download file: {e}"
        debug(err)
        supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Attempt text extraction
    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)
    debug(f"scanned={scanned}, text_length={len(text) if text else 0}")

    extracted_rows: List[Dict[str, Any]] = []

    # If scanned -> convert to images and OCR each page
    if scanned:
        debug("SCANNED PDF detected — converting to images")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
            debug(f"Converted scanned PDF to {len(images)} images")
        except Exception as e:
            debug("pdf2image conversion error:", e)
            images = []

        # For each page: try local OCR first; then OpenAI fallback
        for i, img in enumerate(images, start=1):
            debug(f"OCR page {i} — local ptysr? {USE_PYTESSACT}")
            page_text = ""
            page_rows: List[Dict[str, Any]] = []
            if USE_PYTESSACT:
                try:
                    page_text = ocr_image_local(img)
                    debug(f"Local OCR page {i}: {len(page_text)} chars")
                except Exception as e:
                    debug("Local OCR page error:", e)
                    page_text = ""

                # parse rows
                if page_text:
                    page_rows = parse_cbc_rows_from_text(page_text)
                    debug(f"Parsed {len(page_rows)} rows from page {i} (local OCR)")
            else:
                # fallback: try OpenAI OCR for the page (best-effort)
                try:
                    ocr_json = ocr_image_openai_fallback(img)
                    if ocr_json and isinstance(ocr_json, dict) and "cbc" in ocr_json:
                        page_rows = ocr_json.get("cbc", [])
                        debug(f"OpenAI OCR page {i} returned {len(page_rows)} CBC rows")
                    else:
                        debug("OpenAI OCR page returned no CBC rows or invalid JSON")
                except Exception as e:
                    debug("OpenAI fallback OCR exception:", e)

            # If still nothing, attempt to run OCR via pytesseract on resized grayscale even if module absent -> catch exception
            if not page_rows and not page_text and USE_PYTESSACT:
                try:
                    page_text = ocr_image_local(img)
                    page_rows = parse_cbc_rows_from_text(page_text)
                    debug(f"Retry local OCR parsed {len(page_rows)} rows on page {i}")
                except Exception as e:
                    debug("Second local OCR attempt failed:", e)

            # If we have page_rows, extend
            if page_rows:
                extracted_rows.extend(page_rows)
            else:
                # try parse from page_text if present
                if page_text:
                    p_rows = parse_cbc_rows_from_text(page_text)
                    if p_rows:
                        extracted_rows.extend(p_rows)
                        debug(f"Parsed {len(p_rows)} rows from page {i} via text parsing")

        # end for pages

    else:
        # Digital PDF: parse the text directly for table rows
        debug("Digital PDF detected — parsing text")
        extracted_rows = parse_cbc_rows_from_text(text)
        debug(f"Parsed {len(extracted_rows)} rows from digital PDF text")

    # If still no extracted rows, fail gracefully and save error
    if not extracted_rows:
        err = "No usable CBC/chemistry data extracted"
        debug(err)
        supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Build canonical dict
    canonical = build_cbc_value_dict(extracted_rows)
    debug("Canonical values keys:", list(canonical.keys()))

    # Compose merged_text (we still call the text interpretation model for richer narrative)
    merged_text = json.dumps({"cbc": extracted_rows}, ensure_ascii=False)

    # Call AI interpretation (text only) — using chat completions with response_format json_object
    interpretation = {}
    try:
        sys_prompt = (
            "You are a concise clinical assistant. Given lab values (CBC and chemistry) produce "
            "strict JSON object with fields: patient (name, age, sex), cbc (list of analytes with value/units/refs/flag), summary (impression, suggested_follow_up). "
            "Do NOT invent numbers."
        )
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": merged_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        # resp.choices[0].message.content might be string or object; try to load
        raw = resp.choices[0].message.content
        if isinstance(raw, str):
            try:
                interpretation = json.loads(raw)
            except Exception:
                debug("Could not parse interpretation raw string")
                interpretation = {}
        elif isinstance(raw, dict):
            interpretation = raw
        else:
            # try parsed property
            interpretation = getattr(resp.choices[0].message, "parsed", {}) or {}
        debug("AI interpretation obtained keys:", list(interpretation.keys()))
    except Exception as e:
        debug("AI interpretation error:", e)
        interpretation = {"cbc": extracted_rows, "summary": {"impression": "", "suggested_follow_up": ""}}

    # Generate routes using canonical dict + optional previous
    try:
        routes_result = generate_clinical_routes_v4(canonical, previous)
    except Exception as e:
        debug("Route engine error:", e)
        traceback.print_exc()
        routes_result = {
            "patterns": [],
            "routes": [],
            "next_steps": [],
            "differential": [],
            "severity": 1,
            "urgency": "low",
            "trend": "no_data"
        }

    # Merge routes_result into interpretation
    interpretation["routes_v4"] = routes_result
    # Also include canonical values for UI consumption
    interpretation["canonical_values"] = canonical

    # Save back to supabase
    try:
        supabase.table("reports").update(
            {
                "ai_status": "completed",
                "ai_results": interpretation,
                "ai_error": None
            }
        ).eq("id", report_id).execute()
        debug(f"Report {report_id} processed successfully — saved AI results.")
    except Exception as e:
        debug("Failed saving results to supabase:", e)
        return {"error": f"Failed saving results: {e}"}

    return {"success": True, "data": interpretation}

# -------------------------
# Worker loop
# -------------------------
def main_loop():
    print("Entering main loop — watching for jobs...")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
            # supabase returns object with .data attribute
            jobs = getattr(res, "data", None) or []
            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            job_id = job.get("id")
            print(f"🔎 Found job: {job_id}")

            # mark processing
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", job_id).execute()

            # process
            out = process_report(job)
            debug("process_report result:", out)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main_loop()
