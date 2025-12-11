#!/usr/bin/env python3
"""
AMI Health Worker V4 - Full worker.py with Balanced OCR + CBC parsing + Clinical Notes Engine V1

Features:
- Polls Supabase for reports with ai_status='pending'
- Downloads PDF from Supabase storage (or pdf_url)
- Detects digital vs scanned PDF
- If scanned: renders pages → balanced aggressive preprocess (max width 1000px, quality 55, grayscale) → OpenAI Vision OCR (gpt-4o)
- If digital: extracts text via pypdf
- Parses CBC + chemistry values (robust regex)
- Canonical mapping into defined keys
- Route Engine V4 (comprehensive lab-based routes)
- NEW: Clinical Notes Engine V1 — extracts symptoms/signs from free text when no lab data present,
  produces routes, differentials, severity, urgency, and ER-friendly next steps
- Trend analysis (if previous results exist)
- Saves ai_results to Supabase and sets ai_status appropriately

Environment variables expected:
- SUPABASE_URL
- SUPABASE_SERVICE_KEY or SUPABASE_KEY
- OPENAI_API_KEY
- SUPABASE_TABLE (optional, default "reports")
- SUPABASE_BUCKET (optional, default "reports")
- POLL_INTERVAL (optional, default 6)
- AMI_OCR_MAX_WIDTH (optional, default 1000)
- AMI_OCR_JPEG_QUALITY (optional, default 55)
"""

import os
import io
import re
import time
import json
import base64
import traceback
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps

# Optional local OCR support
try:
    import pytesseract
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

# Clients
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from supabase import create_client, Client
except Exception:
    create_client = None
    Client = None

# Load environment
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "reports")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "6"))
MAX_WIDTH = int(os.getenv("AMI_OCR_MAX_WIDTH", "1000"))
JPEG_QUALITY = int(os.getenv("AMI_OCR_JPEG_QUALITY", "55"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

# Initialize clients
supabase = None
if create_client:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Connected to Supabase")
    except Exception as e:
        print("Warning: Failed to create Supabase client:", e)
else:
    print("Warning: supabase library not available.")

openai_client = None
if OpenAI and OPENAI_API_KEY:
    try:
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print("Warning: Failed to init OpenAI client:", e)
else:
    if not OPENAI_API_KEY:
        print("Warning: OPENAI_API_KEY not set.")
    else:
        print("Warning: openai package not available.")

# ---- helpers ----
def safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---- PDF text extraction / scanned detection ----
def download_pdf_from_supabase(record: Dict[str, Any]) -> bytes:
    if record.get("pdf_url"):
        import requests
        r = requests.get(record["pdf_url"])
        r.raise_for_status()
        return r.content
    if supabase and record.get("file_path"):
        res = supabase.storage.from_(SUPABASE_BUCKET).download(record["file_path"])
        if hasattr(res, "data"):
            return res.data
        return res
    raise ValueError("No pdf_url or file_path in report record")

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n\n".join(pages).strip()
    except Exception as e:
        # pypdf may fail on some files
        print("pypdf extract_text_from_pdf error:", e)
        return ""

def is_scanned_pdf(pdf_bytes: bytes, threshold_chars: int = 80) -> bool:
    text = extract_text_from_pdf(pdf_bytes)
    if not text or len(text.strip()) < threshold_chars:
        return True
    return False

# ---- OCR preprocessing & calls ----
def preprocess_image_for_ocr(img: Image.Image, max_width: int = MAX_WIDTH, quality: int = JPEG_QUALITY) -> bytes:
    try:
        img = ImageOps.exif_transpose(img)
        w, h = img.size
        if w > max_width:
            new_h = int((max_width / float(w)) * h)
            img = img.resize((max_width, new_h), Image.LANCZOS)
        img = img.convert("L")  # grayscale to reduce size
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        return buf.getvalue()
    except Exception as e:
        print("preprocess_image_for_ocr error:", e)
        buf = io.BytesIO()
        try:
            img.convert("RGB").save(buf, format="JPEG", quality=quality)
            return buf.getvalue()
        except Exception:
            return b""

def ocr_image_with_openai(img_bytes: bytes) -> Dict[str, Any]:
    """Send image to OpenAI Vision (gpt-4o) and request strict JSON for CBC values.
       Returns dict like {"cbc":[{...}, ...]} or {"cbc":[]}."""
    if not openai_client:
        print("OpenAI client not configured - cannot run Vision OCR")
        return {"cbc": []}
    try:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        system = (
            "You are an OCR assistant specialized for laboratory reports. Extract ALL CBC and common chemistry analytes "
            "(Hb, RBC, HCT, MCV, MCH, MCHC, RDW, WBC, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils, Platelets, "
            "CRP, Creatinine, Sodium, Potassium, Chloride, Urea, ALT, AST, CK). "
            "Return STRICT JSON ONLY with this exact structure:\n"
            '{"cbc":[{"analyte":"Hb","value":11.6,"units":"g/dL","reference_low":12.4,"reference_high":16.7}, ... ]}\n'
            "If you cannot find analytes, return {\"cbc\":[]}."
        )
        # Vision call using chat.completions with image payload (format may vary by SDK)
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":system},
                {"role":"user","content":[
                    {"type":"text","text":"Extract lab values from this image and return the strict JSON described."},
                    {"type":"image_url","image_url":{"url":f"data:image/jpeg;base64,{b64}"}}
                ]}
            ],
            response_format={"type":"json_object"},
            temperature=0.0,
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except Exception:
                m = re.search(r'\{.*\}', raw, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except:
                        pass
        return {"cbc": []}
    except Exception as e:
        print("OpenAI Vision OCR error:", e)
        return {"cbc": []}

def do_ocr_on_pdf(pdf_bytes: bytes) -> List[Dict[str, Any]]:
    outputs = []
    try:
        pages = convert_from_bytes(pdf_bytes, dpi=200)
    except Exception as e:
        print("pdf2image convert_from_bytes error:", e)
        return []
    for i, page in enumerate(pages):
        try:
            compressed = preprocess_image_for_ocr(page)
            ocr_json = ocr_image_with_openai(compressed)
            outputs.append(ocr_json)
        except Exception as e:
            print(f"OCR exception page {i}:", e)
            outputs.append({"cbc": []})
    return outputs

# ---- Parsing CBC from text or OCR JSON ----
SYNONYMS = {
    "hb":"Hb","haemoglobin":"Hb","hemoglobin":"Hb",
    "rbc":"RBC","erythrocyte":"RBC",
    "hct":"HCT","haematocrit":"HCT",
    "mcv":"MCV","mean corpuscular volume":"MCV",
    "mch":"MCH","mchc":"MCHC","rdw":"RDW",
    "wbc":"WBC","white cell count":"WBC","leukocyte":"WBC","leucocyte":"WBC",
    "neutrophils":"Neutrophils","neutrophil":"Neutrophils",
    "lymphocytes":"Lymphocytes","monocytes":"Monocytes","eosinophils":"Eosinophils","basophils":"Basophils",
    "platelets":"Platelets","thrombocytes":"Platelets",
    "crp":"CRP","c-reactive protein":"CRP",
    "creatinine":"Creatinine","sodium":"Sodium","na":"Sodium",
    "potassium":"Potassium","k":"Potassium",
    "chloride":"Chloride","cl":"Chloride",
    "urea":"Urea","alt":"ALT","ast":"AST","ck":"CK"
}

def normalize_label(label: str) -> Optional[str]:
    if not label: return None
    l = re.sub(r'[^a-z0-9 ]','',label.lower()).strip()
    if l in SYNONYMS: return SYNONYMS[l]
    for k,v in SYNONYMS.items():
        if k in l: return v
    return None

def parse_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not text: return out
    lines = [ln.strip() for ln in re.split(r'\r|\n', text) if ln.strip()]
    # common patterns: label ... number (units) (ref ...)
    for line in lines:
        # numeric patterns with optional units and optional ref
        m = re.findall(r'([A-Za-z\-\s]{2,30})[:\s]{1,6}(-?\d+\.\d+|-?\d+)(?:\s*([a-zA-Z/%\-\(\)\^0-9]+))?(?:.*ref[:\s]*\(?([0-9\.\-–to\s,]+)\)?)?', line)
        if m:
            for g in m:
                label_raw = g[0].strip()
                val = safe_float(g[1])
                units = g[2].strip() if g[2] else None
                ref = g[3].strip() if g[3] else None
                key = normalize_label(label_raw)
                if key and val is not None:
                    out.setdefault(key,{})['value'] = val
                    if units: out[key]['units']=units
                    if ref:
                        parts = re.split(r'[-–to,]', ref)
                        if len(parts)>=2:
                            out[key]['reference_low']=safe_float(parts[0])
                            out[key]['reference_high']=safe_float(parts[1])
                    out[key]['raw'] = line
        # percents
        p = re.findall(r'([A-Za-z\-\s]{2,30})[:\s]{1,6}(\d{1,3}\.?\d*)\s*%', line)
        if p:
            for g in p:
                k = normalize_label(g[0])
                v = safe_float(g[1])
                if k and v is not None:
                    out.setdefault(k,{})['value']=v
                    out[k]['units']='%'
                    out[k]['raw']=line
    # fallback simple finds
    fb = re.findall(r'\b(hb|haemoglobin|wbc|platelets|crp|creatinine|mcv|mch)\b[^\d]{0,12}(-?\d+\.\d+|-?\d+)', text, re.I)
    for f in fb:
        k = normalize_label(f[0])
        v = safe_float(f[1])
        if k and v is not None:
            out.setdefault(k,{})['value']=v
            out[k]['raw']=f[0]
    return out

def parse_values_from_ocr_json(ocr_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    if not isinstance(ocr_json, dict):
        return out
    items = ocr_json.get("cbc") or []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict): continue
        raw_label = it.get("analyte") or it.get("label") or it.get("name") or ""
        key = normalize_label(raw_label) or raw_label.strip()
        val = safe_float(it.get("value"))
        if val is not None:
            out.setdefault(key,{})['value']=val
        if it.get("units"):
            out.setdefault(key,{})['units']=it.get("units")
        rl = it.get("reference_low") or it.get("ref_low")
        rh = it.get("reference_high") or it.get("ref_high")
        if rl is not None or rh is not None:
            out.setdefault(key,{})['reference_low']=safe_float(rl)
            out.setdefault(key,{})['reference_high']=safe_float(rh)
        out.setdefault(key,{})['raw']=it
    return out

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for k,v in parsed.items():
        out[k] = {
            "value": safe_float(v.get("value")),
            "units": v.get("units"),
            "raw": v.get("raw"),
            "reference_low": v.get("reference_low"),
            "reference_high": v.get("reference_high")
        }
    # compute NLR if possible (Neutrophils & Lymphocytes)
    try:
        n = out.get("Neutrophils",{}).get("value")
        l = out.get("Lymphocytes",{}).get("value")
        if n is not None and l is not None and l != 0:
            out["NLR"] = {"value": round(n / l, 2), "units": None, "raw":"computed NLR"}
    except Exception:
        pass
    return out

# ---- Route Engine (lab-based) ----
COLOR_MAP = {
    5: {"label":"critical","color":"#b91c1c","tw":"bg-red-600","urgency":"high"},
    4: {"label":"severe","color":"#f97316","tw":"bg-orange-500","urgency":"high"},
    3: {"label":"moderate","color":"#f59e0b","tw":"bg-yellow-400","urgency":"medium"},
    2: {"label":"borderline","color":"#facc15","tw":"bg-yellow-300","urgency":"low"},
    1: {"label":"normal","color":"#10b981","tw":"bg-green-500","urgency":"low"}
}

def age_group(age: Optional[float]) -> str:
    if age is None: return "adult"
    try:
        a = float(age)
    except:
        return "adult"
    if a < (1/12): return "neonate"
    if a < 1: return "infant"
    if a < 13: return "child"
    if a < 18: return "teen"
    if a < 65: return "adult"
    return "elderly"

def score_severity_key(key: str, val: Optional[float], ag: str, sex: str) -> int:
    if val is None: return 1
    k = key.lower()
    try:
        if k == "hb":
            low = 12.0 if sex and str(sex).lower()=="female" else 13.0
            if ag in ("neonate","infant"): low = 14.0
            if val < low - 4: return 5
            if val < low - 2: return 4
            if val < low: return 3
            return 1
        if k == "wbc":
            if val > 25: return 5
            if val > 15: return 4
            if val > 11: return 3
            return 1
        if k == "crp":
            if val > 200: return 5
            if val > 100: return 4
            if val > 50: return 3
            if val > 10: return 2
            return 1
        if k == "platelets":
            if val < 20: return 5
            if val < 50: return 4
            if val < 100: return 3
            return 1
        if k == "creatinine":
            if val > 354: return 5
            if val > 200: return 4
            if val > 120: return 3
            return 1
        if k == "nlr":
            if val > 10: return 4
            if val > 5: return 3
            return 1
        if k == "ck":
            if val > 10000: return 5
            if val > 5000: return 4
            if val > 2000: return 3
            return 1
        if k in ("potassium","k"):
            if val < 3.2 or val > 6.0: return 5
            if val < 3.5 or val > 5.5: return 3
            return 1
        if k in ("sodium","na"):
            if val < 120 or val > 160: return 5
            if val < 125 or val > 155: return 3
            return 1
    except:
        return 1
    return 1

def route_engine_all(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ag = age_group(patient_meta.get("age"))
    sex = patient_meta.get("sex") or "unknown"
    patterns = []
    routes = []
    next_steps = []
    ddx = []
    per_key = {}
    severity_scores = []

    for k,v in canonical.items():
        val = v.get("value")
        score = score_severity_key(k, val, ag, sex)
        cmap = COLOR_MAP.get(score, COLOR_MAP[1])
        per_key[k] = {
            "value": val,
            "units": v.get("units"),
            "severity": score,
            "urgency": cmap["urgency"],
            "color": cmap["color"],
            "tw_class": cmap["tw"],
            "raw": v.get("raw"),
            "reference_low": v.get("reference_low"),
            "reference_high": v.get("reference_high")
        }
        severity_scores.append(score)

    Hb = canonical.get("Hb",{}).get("value")
    MCV = canonical.get("MCV",{}).get("value")
    WBC = canonical.get("WBC",{}).get("value")
    Neut = canonical.get("Neutrophils",{}).get("value")
    Lymph = canonical.get("Lymphocytes",{}).get("value")
    NLR = canonical.get("NLR",{}).get("value")
    CRP = canonical.get("CRP",{}).get("value")
    Plate = canonical.get("Platelets",{}).get("value")
    Creat = canonical.get("Creatinine",{}).get("value")
    CK = canonical.get("CK",{}).get("value")
    K = canonical.get("Potassium",{}).get("value")
    Na = canonical.get("Sodium",{}).get("value")
    ALT = canonical.get("ALT",{}).get("value")
    AST = canonical.get("AST",{}).get("value")
    RDW = canonical.get("RDW",{}).get("value")

    # Anaemia
    if Hb is not None:
        s_hb = score_severity_key("Hb", Hb, ag, sex)
        if s_hb > 1:
            patterns.append({"pattern":"anemia","reason":f"Hb {Hb}"})
            if MCV is not None and MCV < 80:
                patterns.append({"pattern":"microcytic anemia","reason":f"MCV {MCV}"})
                routes.append("Iron deficiency route")
                ddx += ["Iron deficiency anemia","Thalassaemia trait","Chronic blood loss"]
                if ag=="teen" and sex.lower()=="female":
                    next_steps.append("Evaluate menstrual blood loss; order ferritin + reticulocyte count")
                else:
                    next_steps.append("Order ferritin, reticulin/reticulocyte count; stool occult blood if adult")
            elif MCV is not None and MCV > 100:
                patterns.append({"pattern":"macrocytic anemia","reason":f"MCV {MCV}"})
                routes.append("Macrocytic route")
                ddx += ["B12 deficiency","Folate deficiency","Alcohol related"]
                next_steps.append("Order B12, folate, reticulocyte; review meds")
            else:
                patterns.append({"pattern":"normocytic anemia","reason":"MCV normal/missing"})
                routes.append("Normocytic anemia route")
                ddx += ["Acute blood loss","Haemolysis","Anaemia of chronic disease"]
                next_steps.append("Order reticulocyte, LDH, peripheral smear")

    # Leukocytosis/infection
    if WBC is not None and WBC > 11:
        patterns.append({"pattern":"leukocytosis","reason":f"WBC {WBC}"})
        if Neut is not None and Neut > 70:
            patterns.append({"pattern":"neutrophilic predominance","reason":f"Neutrophils {Neut}%"})
            routes.append("Bacterial infection route")
            ddx += ["Bacterial infection","Sepsis"]
            next_steps.append("Assess clinically; blood cultures if febrile; consider empiric antibiotics if unstable")
        elif Lymph is not None and Lymph > 50:
            patterns.append({"pattern":"lymphocytosis","reason":f"Lymphocytes {Lymph}%"})
            routes.append("Viral infection route")
            ddx += ["Viral infection","Pertussis"]

    # High CRP / NLR
    if CRP is not None:
        if CRP > 50:
            patterns.append({"pattern":"significant inflammation","reason":f"CRP {CRP}"})
            routes.append("Significant inflammatory response")
            ddx += ["Severe bacterial infection","Systemic inflammatory disease"]
            next_steps.append("Consider urgent review; blood cultures if febrile")
        elif CRP > 10:
            patterns.append({"pattern":"elevated CRP","reason":f"CRP {CRP}"})
            next_steps.append("Clinical correlation for infection/inflammation")

    if NLR is not None and NLR > 5:
        patterns.append({"pattern":"high NLR","reason":f"NLR {NLR}"})
        routes.append("High NLR route")
        next_steps.append("Evaluate for sepsis/serious bacterial infection")

    # Platelets
    if Plate is not None:
        if Plate < 150:
            patterns.append({"pattern":"thrombocytopenia","reason":f"Platelets {Plate}"})
            routes.append("Thrombocytopenia route")
            ddx += ["ITP","DIC","Bone marrow suppression"]
            next_steps.append("Check peripheral smear; repeat platelet count; evaluate bleeding")
        elif Plate > 450:
            patterns.append({"pattern":"thrombocytosis","reason":f"Platelets {Plate}"})
            routes.append("Thrombocytosis route")
            next_steps.append("Consider reactive cause; repeat count and assess inflammation")

    # Creatinine / AKI
    if Creat is not None:
        if Creat > 120:
            patterns.append({"pattern":"renal impairment","reason":f"Creatinine {Creat}"})
            routes.append("AKI route")
            next_steps.append("Repeat creatinine; check electrolytes; assess urine output")

    # Rhabdomyolysis
    if CK is not None and CK > 2000:
        patterns.append({"pattern":"rhabdomyolysis physiology","reason":f"CK {CK}"})
        routes.append("Rhabdomyolysis route")
        next_steps.append("Aggressive IV fluids; monitor potassium and creatinine")

    # Electrolyte criticals
    if K is not None:
        if K < 3.2 or K > 6.0:
            patterns.append({"pattern":"critical potassium","reason":f"K {K}"})
            routes.append("Potassium emergency")
            next_steps.append("Immediate ECG; correct potassium per protocol")
    if Na is not None:
        if Na < 125 or Na > 155:
            patterns.append({"pattern":"significant sodium disturbance","reason":f"Na {Na}"})
            next_steps.append("Assess neurologic status; correct carefully")

    # Liver enzymes
    if (ALT is not None and ALT > 3*40) or (AST is not None and AST > 3*40):
        patterns.append({"pattern":"liver enzyme elevation","reason":f"ALT {ALT} AST {AST}"})
        routes.append("Hepatic injury route")
        next_steps.append("Check hepatitis serology; review meds & alcohol")

    # Pancytopenia/onco flags
    if (Hb is not None and Hb < 10 and Plate is not None and Plate < 150) or ((Hb and Hb < 10) and (WBC and WBC < 4)):
        patterns.append({"pattern":"possible marrow pathology","reason":"Cytopenias present"})
        routes.append("Haematology/Oncology referral route")
        next_steps.append("Urgent haematology referral; consider bone marrow biopsy")

    ddx = list(dict.fromkeys([d for d in ddx if d]))
    overall_sev = max(severity_scores) if severity_scores else 1
    color = COLOR_MAP.get(overall_sev, COLOR_MAP[1])
    urgency = color["urgency"]

    summary_parts = []
    if patterns: summary_parts.append("Patterns: " + "; ".join([p["pattern"] for p in patterns]))
    if routes: summary_parts.append("Primary routes: " + "; ".join(routes))
    if ddx: summary_parts.append("Top differentials: " + ", ".join(ddx[:6]))
    if next_steps: summary_parts.append("Immediate suggestions: " + " | ".join(next_steps[:6]))

    age_note = ""
    if ag == "elderly": age_note = "Elderly patient — broaden differential to include chronic disease and malignancy."

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "per_key": per_key,
        "overall_severity": overall_sev,
        "urgency": urgency,
        "color": color["color"],
        "tw_class": color["tw"],
        "age_group": ag,
        "age_note": age_note,
        "summary": "\n".join(summary_parts) if summary_parts else "No significant abnormalities detected."
    }

# ---- Clinical Notes Engine V1 (NEW) ----
# Extracts symptoms and signs from free text and predicts routes/differentials/next steps.

SYMPTOM_KEYWORDS = {
    "fever": ["fever","febrile","temperature","pyrexia","°c","°f"],
    "cough": ["cough","coughing","productive cough","dry cough","wheeze","wheezing"],
    "dyspnea": ["shortness of breath","dyspnoea","dyspnea","breathless","difficulty breathing","respiratory distress"],
    "diarrhoea": ["diarr","diarrhoea","diarrhea","loose stool"],
    "vomiting": ["vomit","vomiting","emesis","retch"],
    "abd_pain": ["abdominal pain","tummy pain","stomach pain","abdo pain"],
    "rash": ["rash","urticaria","erythematous"],
    "dehydration": ["dehydrated","dry mucosa","reduced urine","oliguria","not drinking","drowsy"],
    "seizure": ["seizure","convulsion","fits"],
    "bleeding": ["bleed","bleeding","haemorrhage","hemorrhage"],
    "collapse": ["syncope","collapse","unconscious","loss of consciousness"],
    "infection_site_pulmonary": ["pneumonia","chest infection","lung","respiratory"],
    "trauma": ["trauma","injury","fracture","wound","laceration"]
}

VITALS_REGEX = {
    "temp": r'\b(?:temp(?:erature)?|fever)[:\s]*([0-9]{2}\.?[0-9]?)\s*(?:°?c|c|°?f|f)?\b',
    "hr": r'\b(?:hr|pulse|bpm)[:\s]*([0-9]{2,3})\b',
    "rr": r'\b(?:rr|respiratory rate)[:\s]*([0-9]{1,3})\b',
    "spo2": r'\b(?:sp[o0]2)[:\s]*([0-9]{2})\b'
}

def extract_symptoms_from_text(text: str) -> Dict[str, Any]:
    text_l = text.lower() if text else ""
    found = {"symptoms":[], "vitals":{}}
    # find keywords
    for k,kw_list in SYMPTOM_KEYWORDS.items():
        for kw in kw_list:
            if kw in text_l:
                found["symptoms"].append(k)
                break
    # extract vitals
    for k,rx in VITALS_REGEX.items():
        m = re.search(rx, text_l)
        if m:
            try:
                val = float(m.group(1))
                found["vitals"][k]=val
            except:
                found["vitals"][k]=m.group(1)
    # dedupe symptoms
    found["symptoms"] = list(dict.fromkeys(found["symptoms"]))
    return found

def clinical_notes_engine(text: str, patient_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses keyword and small-rule heuristics to interpret clinical notes.
    Returns structure similar to route_engine_all.
    """
    ag = age_group(patient_meta.get("age"))
    sex = patient_meta.get("sex") or "unknown"
    extracted = extract_symptoms_from_text(text)
    symptoms = extracted.get("symptoms", [])
    vitals = extracted.get("vitals", {})

    patterns = []
    routes = []
    next_steps = []
    ddx = []
    severity_score = 1

    # Fever logic
    temp = vitals.get("temp")
    if temp:
        try:
            t = float(temp)
            if t >= 39:
                patterns.append({"pattern":"high fever","reason":f"Temp {t}°C"})
                routes.append("High fever / sepsis risk")
                ddx += ["Severe bacterial infection","Sepsis"]
                next_steps.append("Urgent clinical review; check CRP/CBC; consider blood cultures if febrile")
                severity_score = max(severity_score,4)
            elif t >= 38:
                patterns.append({"pattern":"fever","reason":f"Temp {t}°C"})
                routes.append("Infection/inflammation route")
                ddx += ["Infection","Viral illness"]
                next_steps.append("Clinical review; consider symptomatic treatment; repeat vitals")
                severity_score = max(severity_score,3)
        except:
            pass
    else:
        # keyword-based fever
        if "fever" in symptoms:
            patterns.append({"pattern":"reported fever","reason":"fever mentioned"})
            next_steps.append("Check temperature and look for infection focus")
            severity_score = max(severity_score,2)

    # Respiratory
    if "dyspnea" in symptoms or "cough" in symptoms:
        patterns.append({"pattern":"respiratory symptoms","reason":", ".join([s for s in symptoms if s in ('dyspnea','cough')])})
        routes.append("Respiratory route")
        ddx += ["Bronchiolitis","Pneumonia","Asthma exacerbation","Viral URTI"]
        next_steps.append("Assess respiratory rate and oxygen saturation; consider chest exam and CXR if indicated")
        severity_score = max(severity_score,3)

    # Gastroenteritis / dehydration
    if "diarrhoea" in symptoms or "vomiting" in symptoms:
        patterns.append({"pattern":"gastrointestinal symptoms","reason":", ".join([s for s in symptoms if s in ('diarrhoea','vomiting')])})
        routes.append("Gastroenteritis / dehydration route")
        ddx += ["Viral gastroenteritis","Bacterial enteritis","Food poisoning"]
        next_steps.append("Assess hydration, encourage ORS; consider stool MCS if bloody or persistent; check electrolytes if severe")
        severity_score = max(severity_score,3)
    if "dehydration" in symptoms or "dehydration" in text.lower() or vitals.get("hr",0) and vitals.get("hr",0) > 120:
        next_steps.append("Treat dehydration as needed; consider IV fluids if unable to tolerate ORS")
        severity_score = max(severity_score,4)

    # Neurological red flags
    if "seizure" in symptoms or "collapse" in symptoms:
        patterns.append({"pattern":"neurological red flag","reason":", ".join([s for s in symptoms if s in ('seizure','collapse')])})
        routes.append("Neurology / emergency route")
        ddx += ["Seizure disorder","Syncope","Severe systemic illness"]
        next_steps.append("Urgent clinical assessment; consider ED transfer")
        severity_score = max(severity_score,5)

    # Bleeding / trauma
    if "bleeding" in symptoms or "trauma" in symptoms:
        patterns.append({"pattern":"bleeding/trauma","reason":", ".join([s for s in symptoms if s in ('bleeding','trauma')])})
        routes.append("Trauma / bleeding route")
        next_steps.append("Assess haemodynamic status; immediate haemorrhage control if bleeding")
        severity_score = max(severity_score,4)

    # Rash
    if "rash" in symptoms:
        patterns.append({"pattern":"rash","reason":"rash mentioned"})
        routes.append("Dermatology / allergic route")
        ddx += ["Allergic reaction","Viral exanthem","Drug reaction"]
        next_steps.append("Assess for anaphylaxis signs; treat symptomatically; consider antihistamine/steroid if indicated")
        severity_score = max(severity_score,3)

    # If no symptoms found
    if not patterns:
        patterns.append({"pattern":"no clear symptoms extracted","reason":"No keywords matched or text too short/handwritten"})
        next_steps.append("Document clinical findings and consider ordering relevant tests if clinically indicated (CBC/CRP/ECG/CXR)")

    ddx = list(dict.fromkeys([d for d in ddx if d]))

    color = COLOR_MAP.get(severity_score, COLOR_MAP[1])
    urgency = color["urgency"]

    summary_parts = []
    if patterns: summary_parts.append("Patterns: " + "; ".join([p["pattern"] for p in patterns]))
    if routes: summary_parts.append("Primary routes: " + "; ".join(routes))
    if ddx: summary_parts.append("Top differentials: " + ", ".join(ddx[:6]))
    if next_steps: summary_parts.append("Immediate suggestions: " + " | ".join(next_steps[:6]))

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differential": ddx,
        "severity_score": severity_score,
        "urgency": urgency,
        "color": color["color"],
        "tw_class": color["tw"],
        "age_group": age_group(patient_meta.get("age")),
        "age_note": ("Elderly patient — broaden differential." if age_group(patient_meta.get("age"))=="elderly" else ""),
        "summary": "\n".join(summary_parts) if summary_parts else "No significant findings extracted from notes."
    }

# ---- Trend analysis ----
def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {"trend":"no_previous"}
    diffs = {}
    for k,v in current.items():
        prev_val = previous.get(k,{}).get("value") if previous else None
        cur_val = v.get("value")
        if prev_val is None or cur_val is None: continue
        try:
            delta = cur_val - prev_val
            pct = (delta / prev_val) * 100 if prev_val != 0 else None
            diffs[k] = {"previous":prev_val,"current":cur_val,"delta":delta,"pct_change":pct}
        except:
            pass
    return {"trend":diffs}

# ---- Interpreter wrapper (optional) ----
def call_ai_on_report(text: str) -> Dict[str, Any]:
    """Light wrapper to call gpt-4o-mini for a clean human-style JSON if desired. If OpenAI not configured, returns minimal JSON."""
    if not openai_client:
        # simple fallback: return a minimal JSON shell
        return {"patient":{"name":None,"age":None,"sex":"Unknown"}, "cbc":[], "summary":{"impression":"No AI model available","suggested_follow_up":""}}
    try:
        system_prompt = (
            "You are an assistive clinical tool analysing CBC, chemistry results or structured JSON from OCR. "
            "Return STRICT JSON with fields: patient, cbc (list), summary (impression + suggested_follow_up). "
            "Use concise ER-friendly wording. Never provide a formal diagnosis."
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":system_prompt}, {"role":"user","content":text}],
            response_format={"type":"json_object"},
            temperature=0.1
        )
        raw = resp.choices[0].message.content
        if isinstance(raw, dict): return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except:
                m = re.search(r'\{.*\}', raw, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except:
                        pass
        # fallback
        return {"patient":{"name":None,"age":None,"sex":"Unknown"}, "cbc":[], "summary":{"impression":"No structured interpretation available","suggested_follow_up":""}}
    except Exception as e:
        print("call_ai_on_report error:", e)
        return {"error": str(e)}

# ---- Supabase save ----
def save_ai_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        print("Supabase not configured - skipping save.")
        return
    try:
        payload = {"ai_status":"completed","ai_results":ai_results,"ai_error":None}
        supabase.table(SUPABASE_TABLE).update(payload).eq("id",report_id).execute()
        print(f"Saved ai_results for {report_id}")
    except Exception as e:
        print("Failed to save ai_results:", e)
        try:
            supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":str(e)}).eq("id",report_id).execute()
        except:
            pass

# ---- Main processing ----
def process_report(job: Dict[str, Any]) -> Dict[str, Any]:
    report_id = job.get("id") or job.get("report_id")
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""
    patient_age = job.get("age")
    patient_sex = job.get("sex")
    print(f"Processing report {report_id} (path={file_path})")
    try:
        if not file_path and not job.get("pdf_url"):
            err = f"Missing file_path or pdf_url for report {report_id}"
            print("Error:", err)
            if supabase:
                supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":err}).eq("id",report_id).execute()
            return {"error":err}

        pdf_bytes = download_pdf_from_supabase(job)
        scanned = is_scanned_pdf(pdf_bytes)
        parsed = {}
        merged_text_for_ai = ""

        if scanned:
            print(f"Report {report_id} detected as SCANNED — running Balanced OCR")
            ocr_pages = do_ocr_on_pdf(pdf_bytes)
            combined_items = []
            for p in ocr_pages:
                if isinstance(p, dict) and p.get("cbc") and isinstance(p.get("cbc"), list):
                    combined_items.extend(p.get("cbc"))
            if combined_items:
                parsed = parse_values_from_ocr_json({"cbc": combined_items})
                merged_text_for_ai = json.dumps({"cbc": combined_items})
            else:
                # OCR didn't find CBC analytes: try to extract any text via pypdf and also run plain OCR-to-text if needed
                text_from_pdf = extract_text_from_pdf(pdf_bytes)
                merged_text_for_ai = (l_text + "\n\n" + text_from_pdf).strip() if l_text else text_from_pdf
                parsed = parse_values_from_text(merged_text_for_ai)
        else:
            print(f"Report {report_id} appears DIGITAL — extracting text")
            text_from_pdf = extract_text_from_pdf(pdf_bytes)
            merged_text_for_ai = (l_text + "\n\n" + text_from_pdf).strip() if l_text else text_from_pdf
            parsed = parse_values_from_text(merged_text_for_ai)

        canonical = canonical_map(parsed)

        # fetch previous results for trend analysis
        previous = None
        try:
            if supabase and job.get("patient_id"):
                prev_q = supabase.table(SUPABASE_TABLE).select("ai_results,created_at").eq("patient_id", job.get("patient_id")).order("created_at", desc=True).limit(1).execute()
                prev_rows = prev_q.data if hasattr(prev_q, "data") else prev_q
                if prev_rows:
                    previous = prev_rows[0].get("ai_results")
        except Exception:
            previous = None

        trends = trend_analysis(canonical, previous)
        # If canonical has analytes → use lab route engine
        if canonical:
            route_info = route_engine_all(canonical, {"age":patient_age,"sex":patient_sex}, previous)
            # interpreter: pass structured canonical JSON for natural-language summary
            ai_input = json.dumps({"canonical":canonical,"routes":route_info}, default=str)
            interpretation = call_ai_on_report(ai_input)
            ai_results = {
                "processed_at": now_iso(),
                "scanned": scanned,
                "canonical": canonical,
                "routes": route_info,
                "trends": trends,
                "ai_interpretation": interpretation
            }
        else:
            # No lab analytes — run Clinical Notes Engine to extract value from notes
            notes_text = merged_text_for_ai or ""
            # if scanned and OCR pages present, aggregate any OCR raw text we can: we prefer text from pypdf (already merged)
            clinical_routes = clinical_notes_engine(notes_text, {"age":patient_age,"sex":patient_sex})
            # we can still call interpretation model with the clinical summary
            ai_input = json.dumps({"clinical_summary":clinical_routes,"notes_excerpt":notes_text[:4000]}, default=str)
            interpretation = call_ai_on_report(ai_input)
            ai_results = {
                "processed_at": now_iso(),
                "scanned": scanned,
                "canonical": canonical,  # empty
                "clinical_routes": clinical_routes,
                "trends": trends,
                "ai_interpretation": interpretation
            }

        save_ai_results_to_supabase(report_id, ai_results)
        print(f"✅ Report {report_id} processed successfully")
        return {"success":True,"data":ai_results}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()
        try:
            if supabase:
                supabase.table(SUPABASE_TABLE).update({"ai_status":"failed","ai_error":err}).eq("id",report_id).execute()
        except:
            pass
        return {"error":err}

# ---- Poll loop ----
def poll_loop():
    if not supabase:
        print("Supabase client not configured - poll disabled.")
        return
    print("AMI Worker V4 polling for pending reports...")
    while True:
        try:
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status","pending").limit(5).execute()
            rows = res.data if hasattr(res, "data") else res
            if rows:
                for r in rows:
                    rid = r.get("id")
                    try:
                        supabase.table(SUPABASE_TABLE).update({"ai_status":"processing"}).eq("id",rid).execute()
                    except:
                        pass
                    process_report(r)
            else:
                time.sleep(POLL_INTERVAL)
        except Exception as e:
            print("Polling error:", e)
            traceback.print_exc()
            time.sleep(5)

# ---- CLI test harness ----
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-pdf", help="Path to local PDF to test")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    if args.test_pdf:
        with open(args.test_pdf, "rb") as f:
            pdfb = f.read()
        def _dl(r): return pdfb
        globals()['download_pdf_from_supabase'] = _dl
        dummy = {"id":"local-test","file_path":"local","patient_id":"local","age":17,"sex":"female","l_text":""}
        print("Processing local PDF test...")
        out = process_report(dummy)
        print("RESULT:", json.dumps(out, indent=2))
    else:
        if args.once:
            res = supabase.table(SUPABASE_TABLE).select("*").eq("ai_status","pending").limit(5).execute()
            rows = res.data if hasattr(res, "data") else res
            if rows:
                for r in rows:
                    try:
                        supabase.table(SUPABASE_TABLE).update({"ai_status":"processing"}).eq("id", r.get("id")).execute()
                    except:
                        pass
                    process_report(r)
            else:
                print("No pending reports.")
        else:
            poll_loop()
