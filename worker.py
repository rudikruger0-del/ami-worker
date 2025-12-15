#!/usr/bin/env python3
"""
AMI Health Worker — Production Clinical Pipeline
PART 1: Foundation, Ingestion, Parsing, Canonical Layer

SAFETY:
• Zero hallucinations
• No inferred labs
• Canonical mapping is PURE
"""

# ==============================
# Imports
# ==============================

import os
import io
import re
import time
import json
import base64
import traceback
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# Optional OCR
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# ==============================
# Environment & Clients
# ==============================

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_KEY missing")

from supabase import create_client, Client

supabase: Optional[Client] = None
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print("⚠️ Supabase init failed:", e)

# OpenAI (Vision + mini interpreter later)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    openai_client = None

# ==============================
# Constants
# ==============================

MAX_OCR_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1000"))
JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "55"))

# ==============================
# Utilities
# ==============================

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def numeric_safety_gate(key: str, value: float) -> Optional[float]:
    """
    HARD physiological bounds.
    Values outside these ranges are silently discarded.
    """
    LIMITS = {
        "Hb": (3, 25),
        "MCV": (50, 130),
        "MCH": (15, 45),
        "WBC": (0.1, 100),
        "Neutrophils": (0, 100),
        "Lymphocytes": (0, 100),
        "Platelets": (1, 2000),
        "CRP": (0, 500),
        "Creatinine": (10, 2000),
        "Sodium": (110, 180),
        "Potassium": (2.0, 7.5),
        "Urea": (0.5, 60),
        "CK": (0, 200000),
        "ALT": (0, 5000),
        "AST": (0, 5000),
    }

    if key not in LIMITS:
        return value

    low, high = LIMITS[key]
    if value < low or value > high:
        return None

    return value

# ==============================
# PDF Ingestion
# ==============================

def download_pdf_from_supabase(job: Dict[str, Any]) -> bytes:
    """
    Download PDF bytes from Supabase storage or direct pdf_url.
    Hardened against SDK quirks.
    """
    if job.get("pdf_url"):
        import requests
        r = requests.get(job["pdf_url"], timeout=30)
        r.raise_for_status()
        return r.content

    if not supabase or not job.get("file_path"):
        raise RuntimeError("No PDF source available")

    res = supabase.storage.from_(SUPABASE_BUCKET).download(job["file_path"])

    if hasattr(res, "data") and res.data:
        return res.data
    if isinstance(res, (bytes, bytearray)):
        return res

    raise RuntimeError("Supabase PDF download failed")

# ==============================
# PDF Text Extraction
# ==============================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract selectable text using pypdf.
    Returns empty string on failure.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages: List[str] = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pass
        return "\n\n".join(pages).strip()
    except Exception:
        return ""


def is_scanned_pdf(pdf_bytes: bytes, threshold_chars: int = 80) -> bool:
    """
    Heuristic: scanned if very little selectable text.
    """
    text = extract_text_from_pdf(pdf_bytes)
    return not text or len(text.strip()) < threshold_chars

# ==============================
# OCR Image Pipeline
# ==============================

def preprocess_image_for_ocr(img: Image.Image) -> bytes:
    """
    Grayscale + resize + JPEG compress.
    """
    img = ImageOps.exif_transpose(img)
    w, h = img.size
    if w > MAX_OCR_WIDTH:
        new_h = int((MAX_OCR_WIDTH / float(w)) * h)
        img = img.resize((MAX_OCR_WIDTH, new_h), Image.LANCZOS)

    img = img.convert("L")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()


def ocr_image_with_openai(img_bytes: bytes) -> Dict[str, Any]:
    """
    Vision OCR → strict JSON.
    """
    if not openai_client:
        return {"cbc": []}

    b64 = base64.b64encode(img_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant specialised in lab reports. "
        "Extract numeric CBC and chemistry values. "
        "Return STRICT JSON only:\n"
        "{ \"cbc\": [ { \"analyte\": \"Hb\", \"value\": 11.6, \"units\": \"g/dL\" } ] }"
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract lab values."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        return raw if isinstance(raw, dict) else json.loads(raw)
    except Exception:
        return {"cbc": []}


def do_ocr_on_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception:
        return outputs

    for page in pages:
        try:
            img_bytes = preprocess_image_for_ocr(page)
            outputs.append(ocr_image_with_openai(img_bytes))
        except Exception:
            outputs.append({"cbc": []})

    return outputs

# ==============================
# Parsing
# ==============================

SYNONYMS = {
    "hb": "Hb",
    "haemoglobin": "Hb",
    "hemoglobin": "Hb",
    "wbc": "WBC",
    "white cell count": "WBC",
    "neutrophils": "Neutrophils",
    "lymphocytes": "Lymphocytes",
    "platelets": "Platelets",
    "crp": "CRP",
    "creatinine": "Creatinine",
    "sodium": "Sodium",
    "potassium": "Potassium",
    "urea": "Urea",
    "ck": "CK",
    "alt": "ALT",
    "ast": "AST",
}

def normalize_label(label: str) -> Optional[str]:
    if not label:
        return None
    l = re.sub(r"[^a-z0-9 ]", "", label.lower()).strip()
    if l in SYNONYMS:
        return SYNONYMS[l]
    for k, v in SYNONYMS.items():
        if k in l:
            return v
    return None


def parse_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not text:
        return out

    lines = [ln.strip() for ln in re.split(r"\r|\n", text) if ln.strip()]
    for line in lines:
        matches = re.findall(
            r"([A-Za-z\-\s]{2,30})[:\s]{1,4}(-?\d+\.?\d*)\s*([a-zA-Z/%]+)?",
            line,
        )
        for label, val, units in matches:
            key = normalize_label(label)
            v = safe_float(val)
            if key and v is not None:
                out[key] = {
                    "value": v,
                    "units": units,
                    "raw": line,
                }
    return out


def parse_values_from_ocr_json(ocr_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    items = ocr_json.get("cbc") or []
    if not isinstance(items, list):
        return out

    for it in items:
        if not isinstance(it, dict):
            continue
        key = normalize_label(it.get("analyte") or "")
        val = safe_float(it.get("value"))
        if key and val is not None:
            out[key] = {
                "value": val,
                "units": it.get("units"),
                "raw": it,
            }
    return out

# ==============================
# Canonical Map (PURE)
# ==============================

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    PURE canonical mapping.
    No interpretation, no staging, no risk flags.
    """
    out: Dict[str, Dict[str, Any]] = {}

    for k, v in parsed.items():
        val = safe_float(v.get("value"))
        if val is None:
            continue

        safe_val = numeric_safety_gate(k, val)
        if safe_val is None:
            continue

        out[k] = {
            "value": safe_val,
            "units": v.get("units"),
            "raw": v.get("raw"),
        }

    # Safe derived value: NLR
    try:
        n = out.get("Neutrophils", {}).get("value")
        l = out.get("Lymphocytes", {}).get("value")
        if n is not None and l is not None and l > 0:
            out["NLR"] = {
                "value": round(n / l, 2),
                "units": None,
                "raw": "computed NLR",
            }
    except Exception:
        pass

    return out

# ==============================
# eGFR (Defined here, applied later)
# ==============================

def calculate_egfr(creatinine_umol: float, age: float, sex: str) -> Optional[float]:
    """
    CKD-EPI 2021 (race-free)
    """
    try:
        scr = creatinine_umol / 88.4
        sex = (sex or "").lower()

        if sex == "female":
            k = 0.7
            a = -0.241
            sex_factor = 1.012
        else:
            k = 0.9
            a = -0.302
            sex_factor = 1.0

        egfr = (
            142
            * min(scr / k, 1) ** a
            * max(scr / k, 1) ** -1.200
            * (0.9938 ** age)
            * sex_factor
        )
        return round(egfr, 1)
    except Exception:
        return None
# ==============================
# ROUTE ENGINE — FULL
# ==============================

SEVERITY_MAP = {
    1: "normal",
    2: "mild",
    3: "moderate",
    4: "severe",
    5: "critical",
}

def age_group(age: Optional[float]) -> str:
    if age is None:
        return "adult"
    try:
        a = float(age)
    except Exception:
        return "adult"

    if a < (1 / 12):
        return "neonate"
    if a < 1:
        return "infant"
    if a < 13:
        return "child"
    if a < 18:
        return "teen"
    if a < 65:
        return "adult"
    return "elderly"


def score_severity_key(
    key: str,
    value: Optional[float],
    ag: str,
    sex: str,
) -> int:
    """
    Per-analyte conservative severity scoring.
    Returns 1–5 only.
    """
    if value is None:
        return 1

    k = key.lower()
    sex = (sex or "").lower()

    try:
        # Hemoglobin
        if k == "hb":
            low = 12.0 if sex == "female" else 13.0
            if ag in ("neonate", "infant"):
                low = 14.0
            if value < low - 4:
                return 5
            if value < low - 2:
                return 4
            if value < low:
                return 3
            return 1

        # WBC
        if k == "wbc":
            if value > 25:
                return 5
            if value > 15:
                return 4
            if value > 11:
                return 3
            return 1

        # CRP
        if k == "crp":
            if value > 200:
                return 5
            if value > 100:
                return 4
            if value > 50:
                return 3
            if value > 10:
                return 2
            return 1

        # Platelets
        if k == "platelets":
            if value < 20:
                return 5
            if value < 50:
                return 4
            if value < 100:
                return 3
            return 1

        # Creatinine
        if k == "creatinine":
            if value > 354:
                return 5
            if value > 200:
                return 4
            if value > 120:
                return 3
            return 1

        # NLR
        if k == "nlr":
            if value > 10:
                return 4
            if value > 5:
                return 3
            return 1

        # CK
        if k == "ck":
            if value > 10000:
                return 5
            if value > 5000:
                return 4
            if value > 2000:
                return 3
            return 1

        # Potassium
        if k in ("potassium", "k"):
            if value < 3.2 or value > 6.0:
                return 5
            if value < 3.5 or value > 5.5:
                return 3
            return 1

        # Sodium
        if k in ("sodium", "na"):
            if value < 120 or value > 160:
                return 5
            if value < 125 or value > 155:
                return 3
            return 1

    except Exception:
        return 1

    return 1


def route_engine_all(
    canonical: Dict[str, Dict[str, Any]],
    patient_meta: Dict[str, Any],
    previous: Optional[Dict[str, Any]],
    doctor_trust_flags: Dict[str, bool],
) -> Dict[str, Any]:
    """
    FULL route engine.
    Pattern recognition → routes → differentials → next steps
    """

    ag = age_group(patient_meta.get("age"))
    sex = patient_meta.get("sex") or "unknown"

    patterns: List[Dict[str, str]] = []
    routes: List[str] = []
    next_steps: List[str] = []
    differentials: List[str] = []

    per_key: Dict[str, Any] = {}
    severity_scores: List[int] = []

    # -----------------------------
    # Per-analyte severity scoring
    # -----------------------------
    for k, v in canonical.items():
        val = v.get("value")
        score = score_severity_key(k, val, ag, sex)
        per_key[k] = {
            "value": val,
            "units": v.get("units"),
            "severity": score,
            "severity_text": SEVERITY_MAP[score],
            "raw": v.get("raw"),
        }
        severity_scores.append(score)

    # Convenience aliases
    Hb = canonical.get("Hb", {}).get("value")
    MCV = canonical.get("MCV", {}).get("value")
    WBC = canonical.get("WBC", {}).get("value")
    Neut = canonical.get("Neutrophils", {}).get("value")
    Lymph = canonical.get("Lymphocytes", {}).get("value")
    NLR = canonical.get("NLR", {}).get("value")
    CRP = canonical.get("CRP", {}).get("value")
    Plate = canonical.get("Platelets", {}).get("value")
    Creat = canonical.get("Creatinine", {}).get("value")
    CK = canonical.get("CK", {}).get("value")
    K = canonical.get("Potassium", {}).get("value")
    Na = canonical.get("Sodium", {}).get("value")
    ALT = canonical.get("ALT", {}).get("value")
    AST = canonical.get("AST", {}).get("value")
    RDW = canonical.get("RDW", {}).get("value")

    # =============================
    # ANAEMIA ROUTES
    # =============================
    if Hb is not None:
        hb_score = score_severity_key("Hb", Hb, ag, sex)
        if hb_score > 1:
            patterns.append({"pattern": "anemia", "reason": f"Hb {Hb}"})

            if MCV is not None and MCV < 80:
                patterns.append({"pattern": "microcytic anemia", "reason": f"MCV {MCV}"})
                routes.append("Iron deficiency evaluation route")
                differentials += [
                    "Iron deficiency anemia",
                    "Chronic blood loss",
                    "Thalassaemia trait",
                ]
                next_steps.append(
                    "Consider ferritin, iron studies, and reticulocyte count"
                )

            elif MCV is not None and MCV > 100:
                patterns.append({"pattern": "macrocytic anemia", "reason": f"MCV {MCV}"})
                routes.append("Macrocytic anemia evaluation route")
                differentials += [
                    "Vitamin B12 deficiency",
                    "Folate deficiency",
                    "Alcohol-related anemia",
                    "Myelodysplasia",
                ]
                next_steps.append(
                    "Consider vitamin B12, folate, and medication review"
                )

            else:
                patterns.append(
                    {"pattern": "normocytic anemia", "reason": "MCV normal or unavailable"}
                )
                routes.append("Normocytic anemia evaluation route")
                differentials += [
                    "Anaemia of chronic disease",
                    "Renal disease",
                    "Acute blood loss",
                    "Haemolysis",
                ]
                next_steps.append(
                    "Consider reticulocyte count, LDH, and peripheral smear"
                )

    # =============================
    # INFECTION / INFLAMMATION
    # =============================
    if WBC is not None and WBC > 11:
        patterns.append({"pattern": "leukocytosis", "reason": f"WBC {WBC}"})

        if Neut is not None and Neut > 70:
            patterns.append(
                {"pattern": "neutrophilic predominance", "reason": f"Neutrophils {Neut}%"}
            )
            routes.append("Bacterial infection consideration")
            differentials += ["Bacterial infection", "Severe inflammation"]
            next_steps.append(
                "Clinical correlation for infectious source is advised"
            )

        elif Lymph is not None and Lymph > 50:
            patterns.append(
                {"pattern": "lymphocytosis", "reason": f"Lymphocytes {Lymph}%"}
            )
            routes.append("Viral / lymphoid response consideration")
            differentials += ["Viral infection", "Reactive lymphocytosis"]
            next_steps.append("Consider viral causes and clinical context")

    if CRP is not None:
        if CRP > 50:
            patterns.append({"pattern": "marked inflammation", "reason": f"CRP {CRP}"})
            routes.append("Significant inflammatory response")
            differentials += [
                "Severe infection",
                "Systemic inflammatory condition",
            ]
            doctor_trust_flags["has_acute_risk"] = True
            next_steps.append(
                "Prompt clinical review and correlation with symptoms is recommended"
            )
        elif CRP > 10:
            patterns.append({"pattern": "elevated CRP", "reason": f"CRP {CRP}"})
            next_steps.append("Clinical correlation for inflammation is suggested")

    # =============================
    # PLATELETS
    # =============================
    if Plate is not None:
        if Plate < 100:
            patterns.append(
                {"pattern": "thrombocytopenia", "reason": f"Platelets {Plate}"}
            )
            routes.append("Thrombocytopenia evaluation")
            differentials += [
                "Immune thrombocytopenia",
                "Bone marrow suppression",
                "Sepsis-related consumption",
            ]
            doctor_trust_flags["has_acute_risk"] = True
            next_steps.append(
                "Peripheral smear and repeat count may assist evaluation"
            )
        elif Plate > 450:
            patterns.append(
                {"pattern": "thrombocytosis", "reason": f"Platelets {Plate}"}
            )
            routes.append("Thrombocytosis evaluation")
            differentials += [
                "Reactive thrombocytosis",
                "Myeloproliferative disorder",
            ]
            next_steps.append("Repeat count and inflammatory markers may be useful")

    # =============================
    # RENAL
    # =============================
    if Creat is not None and Creat > 120:
        patterns.append(
            {"pattern": "renal impairment", "reason": f"Creatinine {Creat}"}
        )
        routes.append("Renal function evaluation")
        differentials += [
            "Acute kidney injury",
            "Chronic kidney disease",
            "Dehydration",
        ]
        doctor_trust_flags["has_acute_risk"] = True
        next_steps.append(
            "Trend creatinine and assess volume status and medications"
        )

    # =============================
    # ELECTROLYTES
    # =============================
    if K is not None and (K < 3.2 or K > 6.0):
        patterns.append(
            {"pattern": "potassium disturbance", "reason": f"Potassium {K}"}
        )
        routes.append("Electrolyte disturbance")
        doctor_trust_flags["has_acute_risk"] = True
        next_steps.append(
            "Clinical correlation and ECG consideration may be appropriate"
        )

    if Na is not None and (Na < 125 or Na > 155):
        patterns.append({"pattern": "sodium disturbance", "reason": f"Sodium {Na}"})
        routes.append("Electrolyte disturbance")
        doctor_trust_flags["has_acute_risk"] = True
        next_steps.append(
            "Gradual correction with careful monitoring is advised"
        )

    # =============================
    # DOMINANT SEVERITY
    # =============================
    overall_severity = max(severity_scores) if severity_scores else 1

    # Doctor trust override (long-term risk cannot be 'normal')
    if doctor_trust_flags.get("has_long_term_risk") and overall_severity == 1:
        overall_severity = 2

    # Deduplicate differentials
    differentials = list(dict.fromkeys(differentials))

    return {
        "patterns": patterns,
        "routes": routes,
        "differentials": differentials,
        "next_steps": next_steps,
        "per_analyte": per_key,
        "overall_severity": overall_severity,
        "severity_text": SEVERITY_MAP[overall_severity],
    }
# ==============================
# Trend Analysis
# ==============================

def trend_analysis(
    current: Dict[str, Dict[str, Any]],
    previous: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Simple numeric trend comparison.
    Never infers missing values.
    """
    if not previous:
        return {"trend": "no_previous"}

    diffs: Dict[str, Any] = {}

    for k, v in current.items():
        prev_val = None
        try:
            prev_val = previous.get("canonical", {}).get(k, {}).get("value")
        except Exception:
            prev_val = None

        cur_val = v.get("value")

        if prev_val is None or cur_val is None:
            continue

        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {
                "previous": prev_val,
                "current": cur_val,
                "delta": delta,
                "pct_change": pct,
            }
        except Exception:
            continue

    return {"trend": diffs}


# ==============================
# process_report
# ==============================

def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main processing pipeline.
    Order is SAFETY-CRITICAL and must not be changed.
    """

    report_id = job.get("id") or job.get("report_id")
    patient_id = job.get("patient_id")
    patient_age = job.get("age")
    patient_sex = job.get("sex")
    l_text = job.get("l_text") or ""

    print(f"Processing report {report_id}")

    try:
        # ----------------------------------
        # 1. PDF ingestion
        # ----------------------------------
        pdf_bytes = download_pdf_from_supabase(job)

        # ----------------------------------
        # 2. Digital vs scanned detection
        # ----------------------------------
        scanned = is_scanned_pdf(pdf_bytes)

        # ----------------------------------
        # 3. Text extraction / OCR
        # ----------------------------------
        parsed: Dict[str, Dict[str, Any]] = {}
        merged_text = ""

        if scanned:
            ocr_pages = do_ocr_on_pdf(pdf_bytes)
            combined_items: List[Dict[str, Any]] = []

            for page in ocr_pages:
                if isinstance(page, dict) and page.get("cbc"):
                    combined_items.extend(page.get("cbc"))

            if combined_items:
                parsed = parse_values_from_ocr_json({"cbc": combined_items})
                merged_text = json.dumps({"cbc": combined_items})
            else:
                text = extract_text_from_pdf(pdf_bytes)
                parsed = parse_values_from_text(text)
                merged_text = text
        else:
            text = extract_text_from_pdf(pdf_bytes)
            merged_text = f"{l_text}\n\n{text}".strip() if l_text else text
            parsed = parse_values_from_text(merged_text)

        # ----------------------------------
        # 4. Canonical mapping (PURE)
        # ----------------------------------
        canonical = canonical_map(parsed)

        # ----------------------------------
        # 5. Doctor trust flags (INIT FIRST)
        # ----------------------------------
        doctor_trust_flags = {
            "has_acute_risk": False,
            "has_long_term_risk": False,
        }

        # ----------------------------------
        # 6. Lipid normalization (naming)
        # ----------------------------------
        lipid_aliases = {
            "LDL Cholesterol": "LDL",
            "LDL-C": "LDL",
            "Triglyceride": "Triglycerides",
            "Non HDL": "Non-HDL",
            "Non-HDL Cholesterol": "Non-HDL",
            "HDL Cholesterol": "HDL",
        }

        for src, dest in lipid_aliases.items():
            if src in canonical and dest not in canonical:
                canonical[dest] = canonical[src]

        # ----------------------------------
        # 7. Long-term risk guardrails
        # ----------------------------------
        ldl = canonical.get("LDL", {}).get("value")
        triglycerides = canonical.get("Triglycerides", {}).get("value")
        non_hdl = canonical.get("Non-HDL", {}).get("value")

        if ldl is not None and ldl >= 3.0:
            doctor_trust_flags["has_long_term_risk"] = True

        if triglycerides is not None and triglycerides >= 1.7:
            doctor_trust_flags["has_long_term_risk"] = True

        if non_hdl is not None and non_hdl >= 3.4:
            doctor_trust_flags["has_long_term_risk"] = True

        # ----------------------------------
        # 8. eGFR calculation (CKD-EPI 2021)
        # ----------------------------------
        creat = canonical.get("Creatinine", {}).get("value")
        if creat is not None and patient_age:
            egfr = calculate_egfr(creat, patient_age, patient_sex)
            if egfr is not None:
                canonical["eGFR"] = {
                    "value": egfr,
                    "units": "mL/min/1.73m²",
                    "raw": "calculated CKD-EPI 2021",
                }

        # ----------------------------------
        # 9. Fetch previous ai_results
        # ----------------------------------
        previous = None
        try:
            if supabase and patient_id:
                prev_q = (
                    supabase
                    .table(SUPABASE_TABLE)
                    .select("ai_results,created_at")
                    .eq("patient_id", patient_id)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                prev_rows = prev_q.data if hasattr(prev_q, "data") else []
                if prev_rows:
                    previous = prev_rows[0].get("ai_results")
        except Exception:
            previous = None

        # ----------------------------------
        # 10. Trend analysis
        # ----------------------------------
        trends = trend_analysis(canonical, previous)

        # ----------------------------------
        # 11. Route engine
        # ----------------------------------
        route_info = route_engine_all(
            canonical,
            {"age": patient_age, "sex": patient_sex},
            previous,
            doctor_trust_flags,
        )

        # ----------------------------------
        # 12. Final AI result object
        # ----------------------------------
        ai_results = {
            "processed_at": now_iso(),
            "scanned": scanned,
            "canonical": canonical,
            "routes": route_info,
            "trends": trends,
            "doctor_trust_flags": doctor_trust_flags,
        }

        # ----------------------------------
        # 13. Save to Supabase
        # ----------------------------------
        if supabase and report_id:
            supabase.table(SUPABASE_TABLE).update(
                {
                    "ai_status": "completed",
                    "ai_results": ai_results,
                    "ai_error": None,
                }
            ).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_results}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        try:
            if supabase and report_id:
                supabase.table(SUPABASE_TABLE).update(
                    {
                        "ai_status": "failed",
                        "ai_error": err,
                    }
                ).eq("id", report_id).execute()
        except Exception:
            pass

        return {"error": err}
# ==============================
# Poll Loop
# ==============================

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))


def poll_loop() -> None:
    """
    Poll Supabase for pending reports and process them.
    Designed to run indefinitely.
    """
    if not supabase:
        print("❌ Supabase client not configured — poll loop disabled.")
        return

    print("🚀 AMI Health Worker started — polling for pending reports")

    while True:
        try:
            res = (
                supabase
                .table(SUPABASE_TABLE)
                .select("*")
                .eq("ai_status", "pending")
                .limit(5)
                .execute()
            )

            rows = res.data if hasattr(res, "data") else []

            if not rows:
                time.sleep(POLL_INTERVAL)
                continue

            for job in rows:
                report_id = job.get("id")
                try:
                    # Mark as processing
                    supabase.table(SUPABASE_TABLE).update(
                        {"ai_status": "processing"}
                    ).eq("id", report_id).execute()
                except Exception:
                    pass

                process_report(job)

        except Exception as e:
            print("⚠️ Polling error:", e)
            traceback.print_exc()
            time.sleep(5)


# ==============================
# CLI Test Harness
# ==============================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AMI Health Worker")
    parser.add_argument(
        "--test-pdf",
        help="Path to local PDF file to test processing",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Process pending jobs once and exit",
    )

    args = parser.parse_args()

    # --------------------------
    # Local PDF test mode
    # --------------------------
    if args.test_pdf:
        print("🧪 Running local PDF test mode")

        with open(args.test_pdf, "rb") as f:
            test_pdf_bytes = f.read()

        # Override downloader for local test
        def _local_download(_job: Dict[str, Any]) -> bytes:
            return test_pdf_bytes

        globals()["download_pdf_from_supabase"] = _local_download

        dummy_job = {
            "id": "local-test",
            "file_path": "local",
            "patient_id": "local",
            "age": None,
            "sex": None,
            "l_text": "",
        }

        result = process_report(dummy_job)
        print("RESULT:")
        print(json.dumps(result, indent=2))

    # --------------------------
    # Single-run mode
    # --------------------------
    elif args.once:
        print("▶️ Running single poll iteration")

        try:
            res = (
                supabase
                .table(SUPABASE_TABLE)
                .select("*")
                .eq("ai_status", "pending")
                .limit(5)
                .execute()
            )
            rows = res.data if hasattr(res, "data") else []

            for job in rows:
                try:
                    supabase.table(SUPABASE_TABLE).update(
                        {"ai_status": "processing"}
                    ).eq("id", job.get("id")).execute()
                except Exception:
                    pass

                process_report(job)

        except Exception as e:
            print("Error during single-run mode:", e)
            traceback.print_exc()

    # --------------------------
    # Continuous worker mode
    # --------------------------
    else:
        poll_loop()
