#!/usr/bin/env python3
"""
AMI Worker V4
- Handles scanned (image) PDFs via pdf2image -> OpenAI Vision (Responses API, input_image)
- Handles digital PDFs via pypdf text extraction
- Uses a robust route engine: Patterns -> Routes -> Suggested Next Steps
- Adds urgency flags, severity grading, differential suggestions, and simple trend comparison
- Writes results back to Supabase 'reports' table as ai_results with structured JSON
"""

print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

# ------------------------
# Standard imports
# ------------------------
import os
import time
import json
import io
import traceback
import datetime
import re
from typing import Optional, List, Dict, Any

# ------------------------
# Third-party imports with defensive prints
# ------------------------
try:
    from supabase import create_client, Client
    print(">>> supabase client imported")
except Exception as e:
    print("❌ Could not import supabase:", e)
    raise

try:
    from openai import OpenAI
    print(">>> openai client imported")
except Exception as e:
    print("❌ Could not import openai:", e)
    raise

try:
    from pypdf import PdfReader
    print(">>> pypdf imported")
except Exception as e:
    print("❌ Could not import pypdf:", e)
    raise

try:
    from pdf2image import convert_from_bytes
    print(">>> pdf2image imported")
except Exception as e:
    print("⚠️ pdf2image not available or poppler not installed:", e)
    # We'll raise later if we need to OCR scanned PDFs
    convert_from_bytes = None

# ------------------------
# Environment + clients
# ------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY missing")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)  # may provide responses API
BUCKET = "reports"

# ------------------------
# Utility helpers
# ------------------------
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except Exception:
        pass

def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def clean_number(val) -> Optional[float]:
    """Normalise and extract the first numeric value from strings like '88.0%', '11,6 g/dL', '<0.2' -> float"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if s == "":
        return None
    # Replace comma decimal
    s = s.replace(",", ".")
    # Remove common trailing characters like % or *
    # Use regex to get first float-looking token
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

# ------------------------
# PDF text extraction
# ------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable PDFs. Returns concatenated text of pages."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        combined = "\n\n".join(pages).strip()
        return combined
    except Exception as e:
        safe_print("PDF parse error:", e)
        return ""

def is_scanned_pdf(pdf_text: str) -> bool:
    """If very little text was extracted, assume it's a scanned PDF."""
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False

# ------------------------
# OpenAI Vision OCR (Responses API, input_image) - robust wrapper
# ------------------------
def ocr_image_via_responses(image_bytes: bytes) -> Dict[str, Any]:
    """
    Perform OCR / table extraction on a single image using the OpenAI Responses API.
    Important: This uses 'input_image' style in 'input' list (binary image bytes), preventing token explosion.
    Returns a dict like {"cbc": [ {analyte, value, units, reference_low, reference_high} ] }
    If failure -> returns {"cbc": []}
    """
    system_prompt = (
        "You are an OCR assistant specialised in laboratory reports. "
        "Extract ALL analytes you can see: CBC (Hb, RBC, HCT, MCV, MCH, MCHC), differential (neutrophils/lymphocytes/monocytes/eosinophils/basophils), platelets, ESR, CRP, "
        "electrolytes (Na/K/Ca/Cl), urea, creatinine, eGFR, ALT/AST/ALP/GGT, bilirubin, CK, CK-MB and other chemistry values.\n\n"
        "Return STRICT JSON only with this structure:\n"
        "{ \"cbc\": [ { \"analyte\": \"\", \"value\": \"\", \"units\": \"\", \"reference_low\": \"\", \"reference_high\": \"\" } ] }\n"
        "Do not add any extra commentary."
    )

    # Prefer Responses API if available
    try:
        # client.responses.create should support binary `input_image` objects per 2025 API
        resp = client.responses.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Please extract analytes from the attached image."},
                {
                    # input_image block — binary image bytes and mime_type
                    "input_image": {
                        "data": image_bytes,
                        "mime_type": "image/png"
                    }
                }
            ],
            # keep temperature low for deterministic extraction
            temperature=0.0
        )

        # resp.output -> list; content[0].json should be the parsed object
        out = None
        try:
            out = resp.output[0].content[0].json
        except Exception:
            # Some clients may expose different shapes — try safer extraction
            # resp.output can be a list of dicts
            safe_print("OCR: unexpected response shape, raw resp:", getattr(resp, "output", None))
            # fallback: attempt to parse text field if present
            try:
                text_content = resp.output_text or ""
                out = json.loads(text_content)
            except Exception:
                out = None

        if isinstance(out, dict):
            # ensure at least an empty cbc list
            if "cbc" not in out or not isinstance(out["cbc"], list):
                out["cbc"] = out.get("cbc", [])
            return out
        else:
            safe_print("OCR: response did not yield dict JSON; returning empty")
            return {"cbc": []}

    except Exception as e:
        safe_print("OCR call failed (Responses API):", type(e).__name__, e)
        # Try fallback to chat.completions (older client) if available
    # Fallback path: older openai clients used chat.completions with structured messages.
    try:
        # Build a plain chat-completion call that includes an image_url-like object (some clients support this)
        # NOTE: This fallback may fail depending on client version — keep guarded.
        resp2 = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,<omitted>"} } ]}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        # If this works (rare), extract json
        raw = resp2.choices[0].message.content
        if isinstance(raw, dict):
            return raw
        # sometimes it's string
        try:
            return json.loads(raw)
        except Exception:
            return {"cbc": []}
    except Exception as e2:
        safe_print("OCR fallback failed:", type(e2).__name__, e2)
        return {"cbc": []}

# ------------------------
# Text interpretation via AI (main interpretation model)
# ------------------------
def interpret_text_via_responses(text: str) -> Dict[str, Any]:
    """
    Send extracted text or JSON table to the interpretation model.
    Returns a dict with keys: patient, cbc (list), summary (impression + suggested_follow_up)
    """
    if not text:
        return {}

    system_prompt = (
        "You are AMI — an assistive clinical tool for interpreting blood results. Do NOT give a formal diagnosis or prescribe. "
        "Return STRICT JSON with: patient, cbc (list of analytes with value/units/reference_low/reference_high/flag), summary (impression and suggested_follow_up).\n"
        "If values are missing, omit them. Keep output machine-friendly and concise."
    )

    try:
        resp = client.responses.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.05
        )
        try:
            out = resp.output[0].content[0].json
        except Exception:
            safe_print("Interpret: unexpected response shape; trying output_text")
            out = None
            try:
                out = json.loads(resp.output_text)
            except Exception:
                out = None

        if isinstance(out, dict):
            return out
        else:
            safe_print("Interpret: non-dict response, returning empty")
            return {}
    except Exception as e:
        safe_print("Interpretation call failed (Responses API):", type(e).__name__, e)
        # Fallback to chat.completions if available
    try:
        resp2 = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0.05
        )
        raw = resp2.choices[0].message.content
        if isinstance(raw, dict):
            return raw
        try:
            return json.loads(raw)
        except Exception:
            return {}
    except Exception as e2:
        safe_print("Interpret fallback failed:", e2)
        return {}

# ------------------------
# Build canonical cbc dict for routes
# ------------------------
def build_cbc_value_dict(ai_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert ai_json['cbc'] list -> dictionary keyed by canonical names
    """
    mapping = {}
    rows = ai_json.get("cbc") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("analyte") or r.get("test") or "").strip().lower()
        value = r.get("value")
        # canonical mapping heuristics:
        if any(k in name for k in ["hemoglobin", "haemoglobin", "hb"]):
            mapping["Hb"] = {"raw": r, "value": clean_number(value)}
        elif name.startswith("mcv"):
            mapping["MCV"] = {"raw": r, "value": clean_number(value)}
        elif name.startswith("mch"):
            mapping["MCH"] = {"raw": r, "value": clean_number(value)}
        elif "rbc" in name or "red cell" in name:
            mapping["RBC"] = {"raw": r, "value": clean_number(value)}
        elif "white" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            mapping["WBC"] = {"raw": r, "value": clean_number(value)}
        elif "neutrophil" in name or name.startswith("neut"):
            mapping["Neutrophils"] = {"raw": r, "value": clean_number(value)}
        elif "lymph" in name:
            mapping["Lymphocytes"] = {"raw": r, "value": clean_number(value)}
        elif "platelet" in name or "plt" in name:
            mapping["Platelets"] = {"raw": r, "value": clean_number(value)}
        elif "creatinine" in name:
            mapping["Creatinine"] = {"raw": r, "value": clean_number(value)}
        elif "urea" in name:
            mapping["Urea"] = {"raw": r, "value": clean_number(value)}
        elif name in ("alt", "alanine aminotransferase", "sgpt"):
            mapping["ALT"] = {"raw": r, "value": clean_number(value)}
        elif name in ("ast", "aspartate aminotransferase", "sgot"):
            mapping["AST"] = {"raw": r, "value": clean_number(value)}
        elif "alp" in name:
            mapping["ALP"] = {"raw": r, "value": clean_number(value)}
        elif "ggt" in name or "gamma" in name:
            mapping["GGT"] = {"raw": r, "value": clean_number(value)}
        elif "bilirubin" in name:
            mapping["Bilirubin"] = {"raw": r, "value": clean_number(value)}
        elif name in ("ck", "creatine kinase"):
            mapping["CK"] = {"raw": r, "value": clean_number(value)}
        elif "ck-mb" in name or "ck mb" in name:
            mapping["CK-MB"] = {"raw": r, "value": clean_number(value)}
        elif "sodium" in name or name == "na":
            mapping["Sodium"] = {"raw": r, "value": clean_number(value)}
        elif "potassium" in name or name == "k":
            mapping["Potassium"] = {"raw": r, "value": clean_number(value)}
        elif "calcium" in name or name.startswith("ca"):
            mapping["Calcium"] = {"raw": r, "value": clean_number(value)}
        elif "crp" in name:
            mapping["CRP"] = {"raw": r, "value": clean_number(value)}
        else:
            # also include any other analyte under 'other' by its raw name
            if name:
                mapping[name] = {"raw": r, "value": clean_number(value)}
    return mapping

# ------------------------
# Route Engine V4: Patterns -> Routes -> Next Steps
# ------------------------
def route_engine_v4(cbc_values: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Given canonical cbc_values, produce:
     - patterns: short list of detected patterns
     - routes: what to do / areas to investigate
     - next_steps: specific practical next tests/actions (general)
     - urgency: 'low'|'moderate'|'high'
     - severity_score: 0..10
     - differentials: short tree-like list of likely differentials
    """
    patterns = []
    routes = []
    next_steps = []
    differentials = []
    urgency = "low"
    severity = 0  # 0..10

    # helper to get numeric
    def v(key):
        val = cbc_values.get(key)
        return None if val is None else val.get("value")

    Hb = v("Hb")
    MCV = v("MCV")
    MCH = v("MCH")
    WBC = v("WBC")
    Neut = v("Neutrophils") or v("Neut")
    Lymph = v("Lymphocytes")
    Plt = v("Platelets")
    Cr = v("Creatinine")
    Urea = v("Urea")
    ALT = v("ALT")
    AST = v("AST")
    ALP = v("ALP")
    Bili = v("Bilirubin")
    CK = v("CK")
    Na = v("Sodium")
    K = v("Potassium")
    CRP = v("CRP")

    # quick helper to bump severity/urgency
    def bump(sev_delta=1, set_urgency=None):
        nonlocal severity, urgency
        severity = min(10, severity + sev_delta)
        if set_urgency:
            if set_urgency == "high":
                urgency = "high"
            elif set_urgency == "moderate" and urgency != "high":
                urgency = "moderate"

    # ---------- Anaemia logic ----------
    if Hb is not None:
        # using adult male cutoff ~13 g/dL (adjust in future by sex/age)
        if Hb < 8:
            patterns.append("Severe anaemia")
            routes.append("Immediate anaemia route — urgent clinical review recommended")
            next_steps.append("Urgent full clinical review; transfusion threshold depends on clinical context")
            bump(3, "high")
            differentials.append({"Anaemia": ["Acute blood loss", "Hemolysis", "Severe iron deficiency", "Bone marrow failure"]})
        elif Hb < 11:
            patterns.append("Moderate anaemia")
            routes.append("Investigate cause of anaemia (MCV-based)")
            bump(2, "moderate")
            # MCV patterns
            if MCV is not None:
                if MCV < 80:
                    patterns.append("Microcytic pattern")
                    routes.append("Likely iron-deficiency / chronic blood loss route")
                    next_steps.append("Ferritin, iron studies (TIBC/Transferrin), reticulocyte count")
                    differentials.append({"Microcytic anaemia": ["Iron deficiency", "Thalassaemia trait", "Chronic disease (late)"]})
                elif MCV >= 100:
                    patterns.append("Macrocytic pattern")
                    routes.append("Consider B12/folate deficiency, liver disease, drugs")
                    next_steps.append("B12, folate, LFTs, review medication list (e.g. methotrexate, anticonvulsants)")
                    differentials.append({"Macrocytic anaemia": ["B12/folate deficiency", "Alcohol/liver disease", "Myelodysplasia"]})
                else:
                    patterns.append("Normocytic anaemia")
                    routes.append("Consider chronic disease, renal impairment, early iron deficiency")
                    next_steps.append("Consider renal function, CRP, reticulocyte count")
                    differentials.append({"Normocytic anaemia": ["Chronic disease", "Renal disease", "Acute blood loss", "Hemolysis"]})
            else:
                next_steps.append("Obtain MCV to subtype anaemia")
        else:
            # normal Hb
            pass

    # ---------- White cell patterns ----------
    if WBC is not None:
        if WBC >= 20:
            patterns.append("Marked leukocytosis")
            routes.append("Severe inflammatory/infective / haematologic route")
            next_steps.append("Clinical review, blood cultures if septic picture; consider urgent haematology review")
            bump(3, "high")
            differentials.append({"Marked leukocytosis": ["Severe bacterial infection", "Leukaemoid reaction", "Primary haematologic neoplasm"]})
        elif WBC > 12:
            patterns.append("Leukocytosis")
            if Neut is not None and Neut > 70:
                patterns.append("Neutrophilia pattern")
                routes.append("Bacterial infection / acute inflammation route")
                next_steps.append("Consider cultures, antibiotics guided by clinical context")
                differentials.append({"Neutrophilia": ["Bacterial infection", "Steroids", "Stress response"]})
            elif Lymph is not None and Lymph > 45:
                patterns.append("Lymphocytosis pattern")
                routes.append("Viral or recovery-phase pattern")
                next_steps.append("Consider viral testing depending on symptoms")
                differentials.append({"Lymphocytosis": ["Viral infection", "Pertussis", "Chronic lymphocytic leukaemia (older)"]})
            bump(1, "moderate")
        elif WBC < 4:
            patterns.append("Leukopenia")
            routes.append("Consider viral suppression, bone marrow suppression or drug effect")
            next_steps.append("Check medication list, repeat WBC, consider viral screen")
            bump(2, "moderate")
            differentials.append({"Leukopenia": ["Viral infection", "Drug-induced", "Bone marrow failure"]})

    # ---------- Platelets ----------
    if Plt is not None:
        if Plt < 50:
            patterns.append("Severe thrombocytopenia")
            routes.append("High bleeding risk — urgent review")
            next_steps.append("Clinical bleeding assessment, urgent haematology review; repeat platelet count")
            bump(3, "high")
            differentials.append({"Severe thrombocytopenia": ["Immune thrombocytopenia", "DIC", "Bone marrow failure", "Drug-induced"]})
        elif Plt < 150:
            patterns.append("Thrombocytopenia")
            routes.append("Investigate for immune, consumptive or marrow causes")
            next_steps.append("Review medications, assess for splenomegaly, consider peripheral smear")
            bump(1, "moderate")
        elif Plt > 450:
            patterns.append("Thrombocytosis")
            routes.append("Reactive thrombocytosis vs primary myeloproliferative disorder")
            next_steps.append("Assess for infection/inflammation/iron deficiency; consider repeat and platelet function if needed")
            differentials.append({"Thrombocytosis": ["Reactive (infection/inflammation/iron deficiency)", "Primary MPN"]})

    # ---------- Renal ----------
    if Cr is not None:
        # crude adult creatinine threshold (uM); consider sex/age in later improvements
        if Cr > 150:
            patterns.append("Elevated creatinine")
            routes.append("Renal impairment route")
            next_steps.append("Repeat U&E; check urine output, consider ultrasound and nephrology if rising")
            bump(2, "moderate")
            differentials.append({"High creatinine": ["AKI (pre-renal, intrinsic)", "CKD"]})
        elif Cr > 120:
            patterns.append("Mildly elevated creatinine")
            next_steps.append("Correlate with prior result / hydration status")

    # ---------- Liver ----------
    # treat ALT/AST absolute numbers—thresholds simplified; later replace with lab references
    if ALT is not None or AST is not None or Bili is not None:
        Lvals = [x for x in (ALT, AST, Bili) if x is not None]
        if any(x is not None and x > 3 * 40 for x in (ALT, AST) if x is not None):
            patterns.append("Marked transaminitis")
            routes.append("Hepatocellular injury route")
            next_steps.append("Assess for viral hepatitis, toxins, drugs and consider urgent hepatology if symptomatic")
            bump(3, "high")
            differentials.append({"Transaminitis": ["Acute viral hepatitis", "Drug/toxin", "Ischemic hepatitis"]})
        elif any(x is not None and x > 1.5 * 40 for x in (ALT, AST) if x is not None):
            patterns.append("Mild-moderate transaminitis")
            routes.append("Consider viral/medication/alcohol-related causes")
            next_steps.append("Review medication and alcohol history, consider viral serology")

        if Bili is not None and Bili > 20:
            patterns.append("Hyperbilirubinaemia")
            routes.append("Jaundice evaluation (hemolytic vs hepatic vs obstructive)")
            next_steps.append("Fractionated bilirubin, ultrasound if obstructive features")
            bump(2, "moderate")
            differentials.append({"Hyperbilirubinaemia": ["Hemolysis", "Liver disease", "Obstruction"]})

    # ---------- Cardiac / Muscle (CK) ----------
    if CK is not None and CK > 1000:
        patterns.append("Very high CK")
        routes.append("Rhabdomyolysis physiology suspected")
        next_steps.append("Check creatinine, urine myoglobin, aggressive fluids, consider urgent review")
        bump(3, "high")
        differentials.append({"High CK": ["Rhabdomyolysis (trauma, severe exertion, toxin)"]})

    # ---------- Electrolytes ----------
    if K is not None:
        if K < 3.3:
            patterns.append("Hypokalaemia")
            routes.append("Correct potassium and search for cause")
            next_steps.append("Replace potassium; check ECG if symptomatic")
            bump(2, "moderate")
        elif K > 5.5:
            patterns.append("Hyperkalaemia")
            routes.append("High arrhythmia risk — urgent management based on ECG/symptoms")
            next_steps.append("Urgent ECG; treat hyperkalaemia per local protocol")
            bump(3, "high")

    if Na is not None:
        if Na < 130:
            patterns.append("Moderate-severe hyponatraemia")
            routes.append("Investigate volume status and endocrine causes")
            next_steps.append("Assess volume status, review diuretics, consider endocrinology if chronic")
            bump(2, "moderate")

    # ---------- Inflammation ----------
    if CRP is not None:
        if CRP > 100:
            patterns.append("Very high CRP — strong inflammation")
            routes.append("Suggest source-focused search and urgent evaluation")
            next_steps.append("Correlate with WBC and clinical signs; consider imaging if source unclear")
            bump(2, "moderate")
        elif CRP > 10:
            patterns.append("Raised CRP")
            routes.append("Active inflammation/infection likely")
            next_steps.append("Clinical correlation & targeted investigations")
            bump(1, "moderate")

    # ---------- Build severity label ----------
    if severity >= 7 or urgency == "high":
        severity_label = "critical"
    elif severity >= 4 or urgency == "moderate":
        severity_label = "moderate"
    else:
        severity_label = "mild"

    # If nothing flagged, return reassuring message
    if not patterns:
        patterns.append("No major acute abnormalities detected.")
        next_steps.append("Correlate clinically and compare with prior results if available.")

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "differentials": differentials,
        "urgency": urgency,
        "severity_score": severity,
        "severity_label": severity_label
    }

# ------------------------
# Trend comparison (simple)
# ------------------------
def fetch_prior_reports(report_row: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """
    Try to fetch prior completed reports for same patient (by patient_id or name).
    This is best-effort: checks for patient_id field or name in report row.
    """
    try:
        # If we have patient_id in DB row, use it
        pid = report_row.get("patient_id")
        name = report_row.get("name")
        q = supabase.table("reports").select("*").eq("ai_status", "completed").order("created_at", {"ascending": False}).limit(limit)
        if pid:
            q = q.eq("patient_id", pid)
        elif name:
            q = q.ilike("name", f"%{name}%")
        res = q.execute()
        data = getattr(res, "data", None) or []
        # Exclude current report if present
        data = [r for r in data if r.get("id") != report_row.get("id")]
        return data[:limit]
    except Exception as e:
        safe_print("fetch_prior_reports error:", e)
        return []

def compute_trend_summary(current_cbc_map: Dict[str, Dict[str, Any]], priors: List[Dict[str, Any]]) -> List[str]:
    """
    Compare a few key analytes to previous reports (if any) and summarize trends.
    """
    out = []
    if not priors:
        return ["No prior data available for trend comparison."]
    # build prior numeric snapshots (take most recent with ai_results.cbc)
    def extract_map(report):
        ai = report.get("ai_results") or {}
        if isinstance(ai, str):
            try:
                ai = json.loads(ai)
            except Exception:
                return {}
        rows = ai.get("cbc") or []
        d = {}
        for r in rows:
            name = (r.get("analyte") or "").strip().lower()
            val = clean_number(r.get("value"))
            if name:
                d[name] = val
        return d

    prior_maps = [extract_map(r) for r in priors]

    # check a short list: Hb, WBC, Platelets, Creatinine, CRP
    keys = [("haemoglobin", "Hb"), ("hemoglobin", "Hb"), ("white cell", "WBC"), ("wbc", "WBC"), ("platelet", "Platelets"), ("creatinine", "Creatinine"), ("crp", "CRP")]
    checked = set()
    for raw_key, canonical in keys:
        if canonical in checked:
            continue
        checked.add(canonical)
        # find current value
        cur_entry = current_cbc_map.get(canonical)
        cur_val = None
        if cur_entry:
            cur_val = cur_entry.get("value")
        else:
            # try fuzzy match in current_cbc_map
            for k in current_cbc_map.keys():
                if raw_key in k.lower():
                    cur_val = current_cbc_map[k].get("value")
                    break
        if cur_val is None:
            continue
        # find most recent prior value for same analyte name (loose)
        prior_vals = []
        for pm in prior_maps:
            # try same canonical key names
            if canonical.lower() in pm:
                prior_vals.append(pm.get(canonical.lower()))
            else:
                # try fuzzy keys
                for pk, pv in pm.items():
                    if raw_key in pk:
                        prior_vals.append(pv)
                        break
        if not prior_vals:
            continue
        # compare with most recent
        prev = prior_vals[0]
        if prev is None:
            continue
        try:
            if abs(cur_val - prev) / max(abs(prev), 0.0001) > 0.15:
                out.append(f"{canonical}: changed from {prev} → {cur_val} (significant change)")
            else:
                out.append(f"{canonical}: stable compared to prior ({prev} → {cur_val})")
        except Exception:
            continue

    if not out:
        return ["No meaningful trends detected vs recent results."]
    return out

# ------------------------
# Core report processing
# ------------------------
def process_report(report_row: Dict[str, Any]) -> Dict[str, Any]:
    report_id = report_row.get("id")
    file_path = report_row.get("file_path")
    l_text = report_row.get("l_text") or ""
    safe_print(f"Processing report {report_id} (file_path={file_path}) at {now_iso()}")

    try:
        if not file_path:
            err = "Missing file_path"
            supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
            return {"error": err}

        # download pdf bytes from supabase storage
        pdf_obj = supabase.storage.from_(BUCKET).download(file_path)
        pdf_bytes = getattr(pdf_obj, "data", pdf_obj)

        # extract text and detect scanned
        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        safe_print(f"Report {report_id}: scanned={scanned}, text_length={len(text)}")

        # aggregate CBC rows from OCR or use extracted text for interpretation
        table_rows = []

        if scanned:
            if convert_from_bytes is None:
                raise RuntimeError("pdf2image/poppler not available for scanned PDF OCR.")

            # convert to images (pages)
            pages = convert_from_bytes(pdf_bytes, dpi=200)
            safe_print(f"Converted scanned PDF to {len(pages)} images")
            page_index = 0
            for img in pages:
                page_index += 1
                safe_print(f"OCR page {page_index} ...")
                # serialize image to PNG bytes
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                # perform OCR
                ocr_out = ocr_image_via_responses(img_bytes)
                if ocr_out and isinstance(ocr_out, dict):
                    rows = ocr_out.get("cbc", []) or []
                    if rows:
                        safe_print(f" → OCR extracted {len(rows)} rows on page {page_index}")
                        # append as-is
                        table_rows.extend(rows)
                    else:
                        safe_print(f" → No CBC rows on page {page_index}")
                else:
                    safe_print(f" → OCR returned no usable data on page {page_index}")

            if not table_rows:
                raise ValueError("No usable CBC data extracted from scanned PDF.")

            # merged_text to feed interpretation is JSON table string
            merged_text = json.dumps({"cbc": table_rows}, ensure_ascii=False)
        else:
            # use textual PDF content OR provided l_text
            merged_text = (text or l_text or "").strip()
            if not merged_text:
                raise ValueError("No usable text content extracted from digital PDF.")

        # Interpret the merged_text into structured JSON
        safe_print("Calling interpretation model...")
        ai_interpretation = interpret_text_via_responses(merged_text)
        if not ai_interpretation or not isinstance(ai_interpretation, dict):
            raise ValueError("Interpretation model returned no data")

        # build canonical values for route engine
        canonical = build_cbc_value_dict(ai_interpretation)

        # route engine
        safe_print("Running route engine v4...")
        routes_out = route_engine_v4(canonical)

        # trend comparison (best-effort)
        prior_reports = fetch_prior_reports(report_row, limit=3)
        trend_summary = compute_trend_summary(canonical, prior_reports)

        # assemble final ai_results object
        ai_results = {
            "meta": {
                "generated_at": now_iso(),
                "model": "ami-v4",
                "version": "v4.0"
            },
            "patient": ai_interpretation.get("patient") or {"name": report_row.get("name"), "age": report_row.get("age"), "sex": report_row.get("sex", "Unknown")},
            "cbc_raw": ai_interpretation.get("cbc") or (table_rows if table_rows else []),
            "summary": ai_interpretation.get("summary") or {},
            "canonical": canonical,
            "route_engine": routes_out,
            "trend_summary": trend_summary
        }

        # Save to supabase
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_results,
            "ai_error": None
        }).eq("id", report_id).execute()

        safe_print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_results}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        safe_print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()
        try:
            supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
        except Exception as e2:
            safe_print("Failed to update supabase error:", e2)
        return {"error": err}

# ------------------------
# Worker loop
# ------------------------
def main():
    safe_print("Entering main loop — watching for pending reports...")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
            # supabase client returns an object which exposes .data in many versions
            jobs = getattr(res, "data", None) or (res.get("data") if hasattr(res, "get") else None) or []
            if not jobs:
                safe_print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            safe_print("🔎 Found job:", job.get("id"))
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", job.get("id")).execute()
            process_report(job)

        except KeyboardInterrupt:
            safe_print("Worker interrupted by user — exiting.")
            break
        except Exception as e:
            safe_print("LOOP ERROR:", type(e).__name__, e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
