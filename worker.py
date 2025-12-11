# worker.py  — AMI Worker v4 (Patterns -> Route -> Next Steps)
# Paste/replace your existing worker.py with this file.

print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

import os
import time
import json
import io
import traceback
import base64
import re
from datetime import datetime

# 3rd party imports
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image  # pillow is in requirements

# -------------------------
# ENV + CLIENTS
# -------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set — vision/AI calls will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

# -------------------------
# Utilities
# -------------------------
def safe_get_api_data(res):
    """
    supabase client .execute() can return object shapes depending on sdk version.
    Prefer res.data, fallback to dict-like get.
    """
    try:
        data = getattr(res, "data", None)
        if data is not None:
            return data
    except Exception:
        pass
    try:
        return res.get("data")  # if res is a dict-like
    except Exception:
        return None

def clean_number(val):
    """
    Convert a variety of lab number formats into float (or None).
    Accepts strings like '88.0%', '11,6 g/dL', '<5.0', '≥ 2.0', '4.2*'
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Remove common non-numeric tokens
    s = s.replace(",", ".")
    s = s.replace("%", "")
    s = s.replace("*", "")
    s = s.replace("≥", "")
    s = s.replace(">", "")
    s = s.replace("<", "-")  # keep marker to check later
    # extract first number-looking token
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group())
    except:
        return None

def severity_label_from_score(score):
    if score >= 8:
        return "Critical"
    if score >= 5:
        return "High"
    if score >= 3:
        return "Moderate"
    return "Low"

# -------------------------
# PDF text extraction
# -------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        texts = []
        for page in reader.pages:
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            texts.append(txt)
        return "\n\n".join(texts).strip()
    except Exception as e:
        print("PDF parse error:", e)
        return ""

def is_scanned_pdf(pdf_text: str) -> bool:
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 40:
        return True
    return False

# -------------------------
# OpenAI Vision OCR helper (best-effort)
# - Uses chat completions with image embedded as data URI in a user message.
# - Some OpenAI client versions / model settings may not accept images in this style.
# - If your environment supports the newer Responses API, replace this function accordingly.
# -------------------------
def send_image_ocr_to_openai(image_bytes: bytes) -> dict:
    """
    Send the image to OpenAI & ask it to extract lab analyte table rows.
    Returns a dict like { "cbc": [ { "analyte": "...", "value": "...", "units": "...", "reference_low":"", "reference_high":"" }, ... ] }
    If the call fails or response cannot be parsed, returns {"cbc": []}.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        user_text = (
            "OCR the image below and extract any laboratory analytes (CBC, diff, platelets, "
            "electrolytes, urea, creatinine, liver enzymes, CK, CK-MB, CRP, ESR, etc.).\n\n"
            "Return STRICT JSON with the structure:\n"
            "{ 'cbc': [ { 'analyte': '', 'value': '', 'units': '', 'reference_low': '', 'reference_high': '' } ] }\n"
            "Return only JSON with no additional text.\n\n"
            "Image (data URI) below:\n"
            f"![lab]({f'data:image/png;base64,{b64}'})"
        )

        # Use chat completion — this worked for you earlier. Keep temperature low.
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise OCR and data extraction assistant."},
                {"role": "user", "content": user_text}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # Try a few possible response shapes
        raw = None
        try:
            # new style: resp.choices[0].message.content (string or object)
            content = resp.choices[0].message.content
            if isinstance(content, dict):
                raw = content
            else:
                # content may be a JSON string
                raw = json.loads(content)
        except Exception:
            try:
                # some SDKs expose parsed
                raw = getattr(resp.choices[0].message, "parsed", None)
            except Exception:
                raw = None

        if isinstance(raw, dict) and "cbc" in raw:
            return raw
        # fallback: if resp.output exists (Responses API style), attempt to use it
        try:
            # Attempt Responses-style extraction (if available)
            out = getattr(resp, "output", None)
            if out and isinstance(out, list):
                for item in out:
                    # find .content text or json
                    if isinstance(item, dict):
                        # try item['content'][0]['json'] style
                        for c in item.get("content", []):
                            if isinstance(c, dict) and "json" in c:
                                return c["json"]
        except Exception:
            pass

        # If nothing parsed, attempt to parse text fallback
        try:
            txt = resp.choices[0].message.content
            if isinstance(txt, str):
                parsed = json.loads(txt)
                if isinstance(parsed, dict) and "cbc" in parsed:
                    return parsed
        except Exception:
            pass

        print("⚠️ OCR: Unable to parse OpenAI response; returning empty cbc.")
        return {"cbc": []}

    except Exception as e:
        # Do not crash on OCR; just report and return empty cbc
        print("OCR call failed:", type(e).__name__, e)
        return {"cbc": []}

# -------------------------
# AI interpretation (text)
# -------------------------
def call_ai_on_report(text: str) -> dict:
    """
    Call OpenAI to structure the lab results into JSON (patient, cbc list, summary).
    Returns a dict. On failure, raises or returns basic structure.
    """
    try:
        system_prompt = (
            "You are a clinical lab assistant. Receive lab data (either JSON {cbc:[..]} or free text). "
            "Return STRICT JSON object with keys: patient, cbc, summary.\n\n"
            "cbc should be a list of {analyte, value, units, reference_low, reference_high, flag}.\n"
            "flag should be one of 'low', 'normal', 'high', or 'unknown'.\n"
            "Patient object should have name, age, sex if known (or nulls).\n"
            "Summary should include short 'impression' and 'suggested_follow_up' fields.\n"
            "If you cannot find a numeric value for an analyte, do not invent it — skip it."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # Parse many possible shapes
        try:
            content = resp.choices[0].message.content
            if isinstance(content, dict):
                return content
            else:
                return json.loads(content)
        except Exception:
            try:
                parsed = getattr(resp.choices[0].message, "parsed", None)
                if parsed:
                    return parsed
            except Exception:
                pass

        # last fallback: try to get text and parse
        try:
            rawtext = resp.choices[0].message.content
            if isinstance(rawtext, str):
                return json.loads(rawtext)
        except Exception:
            pass

        raise ValueError("Could not parse AI interpretation response")

    except Exception as e:
        print("AI interpretation error:", type(e).__name__, e)
        # Return a minimal structure so the worker can continue
        return {"patient": {"name": None, "age": None, "sex": "Unknown"}, "cbc": [], "summary": {"impression": "", "suggested_follow_up": ""}}

# -------------------------
# Build canonical cbc dict
# -------------------------
def build_cbc_value_dict(ai_json: dict) -> dict:
    """
    Turn ai_json['cbc'] list into canonical dict keyed by common analyte names.
    """
    mapping = {}
    rows = ai_json.get("cbc") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("analyte") or r.get("test") or "").lower()
        val = r.get("value")
        units = r.get("units")
        if not name:
            continue

        # Map flexible names to canonical keys
        if "haemoglobin" in name or "hemoglobin" in name or name in ("hb", "hgb"):
            mapping["Hb"] = {"raw": r, "value": clean_number(val), "units": units}
        elif name.startswith("mcv"):
            mapping["MCV"] = {"raw": r, "value": clean_number(val), "units": units}
        elif name.startswith("mch") and "mchc" not in name:
            mapping["MCH"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "mchc" in name:
            mapping["MCHC"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "rdw" in name:
            mapping["RDW"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "white cell" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            mapping["WBC"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "neutrophil" in name:
            mapping["Neutrophils"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "lymph" in name:
            mapping["Lymphocytes"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "monocyte" in name:
            mapping["Monocytes"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "platelet" in name or name.startswith("plt"):
            mapping["Platelets"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "creatinine" in name:
            mapping["Creatinine"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "urea" in name:
            mapping["Urea"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "sodium" in name or name == "na":
            mapping["Sodium"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "potassium" in name or name == "k":
            mapping["Potassium"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "crp" in name:
            mapping["CRP"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "alt" in name or "alanine aminotransferase" in name:
            mapping["ALT"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "ast" in name or "aspartate aminotransferase" in name:
            mapping["AST"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "alkaline phosphatase" in name or name.startswith("alp"):
            mapping["ALP"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "bilirubin" in name:
            mapping["Bilirubin"] = {"raw": r, "value": clean_number(val), "units": units}
        elif "ck-mb" in name or "ck mb" in name:
            mapping["CKMB"] = {"raw": r, "value": clean_number(val), "units": units}
        elif name == "ck" or "creatine kinase" in name:
            mapping["CK"] = {"raw": r, "value": clean_number(val), "units": units}
        else:
            # keep other analytes under their raw name
            key = r.get("analyte", name)
            mapping[key] = {"raw": r, "value": clean_number(val), "units": units}
    return mapping

# -------------------------
# Route Engine v4 — Patterns → Route → Next Steps
# - SA-friendly thresholds (defaults)
# - Age-aware: basic child/adult split
# - Urgency and severity scoring
# - Differential branches
# - Trend comparison (if prior results found)
# -------------------------
# Default reference ranges (very conservative defaults; tune by site)
DEFAULTS = {
    "Hb": {"male": (13.0, 17.5), "female": (12.0, 15.5), "child": (11.5, 15.5)},
    "MCV": (80.0, 100.0),
    "MCH": (27.0, 34.0),
    "WBC": (4.0, 11.0),
    "Platelets": (150, 450),
    "Creatinine": (44, 110),  # umol/L typical adult range, adjust by lab
    "CRP": (0, 5),
    "Sodium": (136, 145),
    "Potassium": (3.5, 5.1),
}

def grade_severity_and_urgency(score):
    return severity_label_from_score(score), ("urgent" if score >= 7 else "semi-urgent" if score >= 4 else "routine")

def compare_trends(current, previous):
    """
    Very simple trend comparator:
    current & previous are canonical dicts (from build_cbc_value_dict).
    Returns a short list of trend statements.
    """
    trends = []
    if not previous:
        return trends
    for key in ["Hb", "WBC", "Neutrophils", "Platelets", "CRP", "Creatinine"]:
        cur = current.get(key, {}).get("value")
        prev = previous.get(key, {}).get("value")
        if cur is None or prev is None:
            continue
        delta = cur - prev
        if abs(delta) / (abs(prev) + 1e-6) > 0.15:  # >15% change
            direction = "↑" if delta > 0 else "↓"
            trends.append(f"{key} {direction} by {abs(delta):.2f} (prev {prev}, now {cur})")
    return trends

def generate_clinical_routes(cdict: dict, patient_age=None, patient_sex=None, prior_cdict=None):
    """
    Returns routes list and meta with severity/urgency/differentials/trends.
    """
    routes = []
    score = 0  # internal severity score

    # helper getter
    def v(k):
        return cdict.get(k, {}).get("value") if cdict.get(k) else None

    Hb = v("Hb")
    MCV = v("MCV")
    MCH = v("MCH")
    WBC = v("WBC")
    Neut = v("Neutrophils")
    Lymph = v("Lymphocytes")
    Platelets = v("Platelets")
    Creatinine = v("Creatinine")
    CRP = v("CRP")
    Sodium = v("Sodium")
    Potassium = v("Potassium")

    # Age group
    age_group = "adult"
    try:
        if patient_age is not None and patient_age < 18:
            age_group = "child"
    except:
        pass

    # ---------- Anaemia routes ----------
    if Hb is not None:
        # pick reference based on sex & age
        if patient_sex and patient_sex.lower().startswith("m"):
            low_hb = DEFAULTS["Hb"]["male"][0]
        elif patient_sex and patient_sex.lower().startswith("f"):
            low_hb = DEFAULTS["Hb"]["female"][0]
        else:
            low_hb = DEFAULTS["Hb"]["female"][0]
        if age_group == "child":
            low_hb = DEFAULTS["Hb"]["child"][0]

        if Hb < low_hb:
            score += 3
            routes.append({"pattern": "Anaemia (Hb low)", "detail": f"Hb {Hb} g/dL — below expected for age/sex (threshold {low_hb})"})
            # microcytic?
            if MCV is not None:
                if MCV < DEFAULTS["MCV"][0]:
                    score += 1
                    routes.append({"route": "Microcytic anaemia pattern", "next_steps": ["Ferritin (iron stores)", "Iron studies (Fe/TIBC)", "Reticulocyte count", "Consider GI blood loss if applicable"]})
                    routes.append({"differentials": ["Iron deficiency", "Thalassaemia trait (check MCV vs RBC)", "Anaemia of chronic disease (less likely in microcytic)"]})
                elif MCV > DEFAULTS["MCV"][1]:
                    score += 1
                    routes.append({"route": "Macrocytic anaemia pattern", "next_steps": ["B12 and folate levels", "LFTs", "Medication review (e.g., methotrexate)"]})
                    routes.append({"differentials": ["B12/folate deficiency", "Alcohol, liver disease", "Myelodysplasia (in older patients)"]})
                else:
                    routes.append({"route": "Normocytic anaemia pattern", "next_steps": ["Consider chronic disease, early iron deficiency, renal disease — check CRP, Creatinine, ferritin"]})
            else:
                routes.append({"route": "Anaemia pattern unknown - need red cell indices (MCV/MCH) to subtype", "next_steps": ["Order MCV/MCH if absent; Ferritin if microcytosis suspected"]})

    # ---------- WBC / Infection routes ----------
    if WBC is not None:
        if WBC > DEFAULTS["WBC"][1]:
            score += 2
            routes.append({"pattern": "Leukocytosis", "detail": f"WBC {WBC}"})
            if Neut is not None and Neut > 70:
                score += 2
                routes.append({"route": "Neutrophil-predominant leukocytosis → bacterial infection favored", "next_steps": ["Correlate clinically for source", "Consider blood cultures if systemic", "Start/adjust antibiotics as per clinical judgement"]})
            if Lymph is not None and Lymph > 45:
                routes.append({"route": "Lymphocytosis → viral / viral-reactive pattern", "next_steps": ["Consider viral tests", "Observe vs specific therapy"]})
        elif WBC < DEFAULTS["WBC"][0]:
            score += 2
            routes.append({"pattern": "Leukopenia", "detail": f"WBC {WBC}", "next_steps": ["Consider viral suppression, bone marrow suppression, cytotoxic drugs, or sepsis with margination"]})

    # ---------- Platelets ----------
    if Platelets is not None:
        if Platelets < DEFAULTS["Platelets"][0]:
            score += 2
            routes.append({"pattern": "Thrombocytopenia", "detail": f"Platelets {Platelets}", "next_steps": ["Assess bleeding/clotting risk", "Repeat platelet count", "Consider causes: immune, sepsis, marrow failure, DIC"]})
        elif Platelets > DEFAULTS["Platelets"][1]:
            routes.append({"pattern": "Thrombocytosis", "detail": f"Platelets {Platelets}", "next_steps": ["Reactive thrombocytosis (infection/inflammation/iron deficiency) vs primary; correlate with CRP, iron studies"]})

    # ---------- Renal ----------
    if Creatinine is not None:
        if Creatinine > DEFAULTS["Creatinine"][1]:
            score += 3
            routes.append({"pattern": "Renal impairment", "detail": f"Creatinine {Creatinine} umol/L", "next_steps": ["Repeat U&E urgently", "Assess urine output/hydration, review nephrotoxic meds", "Calculate eGFR if available"]})
        else:
            routes.append({"note": "Creatinine within typical limits (interpret with eGFR/age/sex)"})

    # ---------- Electrolytes ----------
    if Potassium is not None:
        if Potassium < DEFAULTS["Potassium"][0]:
            score += 4
            routes.append({"pattern": "Hypokalaemia", "detail": f"K {Potassium}", "urgency": "urgent", "next_steps": ["Assess ECG if symptomatic", "Replete potassium carefully"]})
        elif Potassium > DEFAULTS["Potassium"][1]:
            score += 5
            routes.append({"pattern": "Hyperkalaemia", "detail": f"K {Potassium}", "urgency": "urgent", "next_steps": ["ECG", "Immediate hyperkalaemia pathway (calcium, insulin+dextrose, consider ion-exchange, dialysis) if severe"]})

    if Sodium is not None:
        if Sodium < DEFAULTS["Sodium"][0]:
            routes.append({"pattern": "Hyponatraemia", "detail": f"Na {Sodium}", "next_steps": ["Assess fluid status, consider SIADH, diuretics, adrenal causes"]})
        elif Sodium > DEFAULTS["Sodium"][1]:
            routes.append({"pattern": "Hypernatraemia", "detail": f"Na {Sodium}", "next_steps": ["Assess dehydration/water loss; manage carefully"]})

    # ---------- CRP / Inflammation ----------
    if CRP is not None:
        if CRP > 100:
            score += 4
            routes.append({"pattern": "High CRP", "detail": f"CRP {CRP} mg/L — strong inflammation/infection signal", "next_steps": ["Consider sepsis workup if clinical signs present", "Source control & empirical antibiotics as per protocol"]})
        elif CRP > 10:
            score += 2
            routes.append({"pattern": "Elevated CRP", "detail": f"CRP {CRP}", "next_steps": ["Correlate with WBC and clinical findings; consider infection/inflammation"]})
        else:
            routes.append({"note": "CRP not substantially elevated"})

    # ---------- CK ----------
    # (simple)
    CK = cdict.get("CK", {}).get("value")
    if CK and CK > 1000:
        score += 3
        routes.append({"pattern": "Very high CK → possible rhabdomyolysis", "next_steps": ["Check renal function urgently", "Aggressive fluids, monitor urine/myoglobin"]})

    # ---------- Differential diagnoses (simple trees) ----------
    differentials = []
    # Microcytic anemia tree
    if Hb and Hb < (DEFAULTS["Hb"]["female"][0] if not patient_sex or not patient_sex.lower().startswith("m") else DEFAULTS["Hb"]["male"][0]) and MCV and MCV < DEFAULTS["MCV"][0]:
        differentials.append({"problem": "Microcytic anaemia", "likely_causes": ["Iron deficiency (most common)", "Thalassaemia trait", "Chronic blood loss (GI, menstruation)"], "tests_to_distinguish": ["Ferritin", "Iron studies (Fe/TIBC)", "Peripheral smear", "Hb electrophoresis if microcytosis with normal ferritin"]})

    # Leukocytosis tree
    if WBC and WBC > DEFAULTS["WBC"][1]:
        if Neut and Neut > 70:
            differentials.append({"problem": "Neutrophil-predominant leukocytosis", "likely_causes": ["Bacterial infection", "Stress response, steroids"], "distinguish_by": ["CRP, cultures, clinical signs"]})
        elif Lymph and Lymph > 45:
            differentials.append({"problem": "Lymphocytosis", "likely_causes": ["Viral infection", "Pertussis", "Chronic lymphocytic states"], "distinguish_by": ["Viral PCRs, clinical course"]})

    # ---------- Trend comparison ----------
    trends = compare_trends(cdict, prior_cdict)

    # ---------- Severity & urgency ----------
    severity_label, urgency = grade_severity_and_urgency(score)

    # if high CRP + high WBC + neutrophilia escalate
    if (CRP and CRP > 80) and (WBC and WBC > 12) and (Neut and Neut > 70):
        severity_label = "Critical"
        urgency = "urgent"
        score = max(score, 9)

    meta = {
        "severity_score": score,
        "severity_label": severity_label,
        "urgency": urgency,
        "differentials": differentials,
        "trends": trends,
    }

    return routes, meta

# -------------------------
# Worker core: process a single report job
# -------------------------
def process_report(job: dict) -> dict:
    report_id = job.get("id")
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""
    patient_age = None
    patient_sex = None

    print(f"🔎 Processing report {report_id} (file_path={file_path})")

    if not file_path:
        err = f"Missing file_path for report {report_id}"
        print("⚠️", err)
        supabase.table("reports").update({"ai_status": "failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Download file bytes
    pdf_download = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_download, "data"):
        pdf_bytes = pdf_download.data
    else:
        # Some SDK versions return dict-like
        try:
            pdf_bytes = pdf_download.get("data")
        except Exception:
            pdf_bytes = pdf_download

    # Extract text
    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)
    print(f"Report {report_id}: scanned={scanned}, text_length={len(text)}")

    combined_rows = []

    if scanned:
        print(f"📄 Report {report_id}: SCANNED PDF detected → running OCR pipeline")
        try:
            images = convert_from_bytes(pdf_bytes, dpi=200)
        except Exception as e:
            print("pdf2image convert_from_bytes error:", e)
            images = []

        for i, pil_img in enumerate(images, start=1):
            try:
                print(f"OCR page {i} ...")
                # Convert to PNG bytes
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                # Attempt OpenAI OCR
                ocr_result = send_image_ocr_to_openai(img_bytes)
                rows = ocr_result.get("cbc", []) if isinstance(ocr_result, dict) else []
                if rows:
                    print(f" → OCR extracted {len(rows)} rows on page {i}")
                    combined_rows.extend(rows)
                else:
                    print(f" → No CBC rows on page {i}")
            except Exception as e:
                print("OCR page exception:", e)
                traceback.print_exc()
    else:
        # Not scanned — attempt to extract structured lines from text
        print("📝 Digital PDF — using text parsing")
        # If the PDF text is in a "CBC Results" block, pass the whole text to AI interpreter
        combined_rows = text

    if (isinstance(combined_rows, list) and len(combined_rows) == 0) or combined_rows is None:
        err = "No usable CBC data extracted"
        print("❌", err)
        supabase.table("reports").update({"ai_status":"failed", "ai_error": err}).eq("id", report_id).execute()
        return {"error": err}

    # Build merged_text for AI interpreter
    if isinstance(combined_rows, list):
        merged_text = json.dumps({"cbc": combined_rows}, ensure_ascii=False)
    else:
        merged_text = combined_rows

    # AI interpretation
    ai_json = call_ai_on_report(merged_text)

    # Extract patient age/sex if AI provided
    try:
        p = ai_json.get("patient", {})
        if isinstance(p, dict):
            patient_age = p.get("age")
            patient_sex = p.get("sex")
    except Exception:
        pass

    # Build canonical cdict
    cdict = build_cbc_value_dict(ai_json)

    # Try to load prior report for trend comparison (best-effort)
    prior_cdict = None
    try:
        # Try to find previous completed report for same patient (if patient name provided)
        pat_name = (ai_json.get("patient") or {}).get("name")
        if pat_name:
            res_prev = supabase.table("reports").select("ai_results, created_at").eq("ai_status", "completed").ilike("ai_results->>patient->>name", f"%{pat_name}%").order("created_at", {"ascending": False}).limit(1).execute()
            prev_data = safe_get_api_data(res_prev)
            if prev_data and len(prev_data) > 0:
                prev_ai = prev_data[0].get("ai_results") or prev_data[0].get("ai_results", {})
                prior_cdict = build_cbc_value_dict(prev_ai)
    except Exception as e:
        print("Trend lookup failed:", e)

    # Generate routes
    try:
        routes, meta = generate_clinical_routes(cdict, patient_age=patient_age, patient_sex=patient_sex, prior_cdict=prior_cdict)
        ai_json["routes"] = routes
        ai_json["routes_meta"] = meta
    except Exception as e:
        print("Route engine error:", e)
        traceback.print_exc()

    # Save results back to DB
    try:
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_json,
            "ai_error": None,
            "processed_at": datetime.utcnow().isoformat()
        }).eq("id", report_id).execute()
        print(f"✅ Report {report_id} processed — saved results.")
    except Exception as e:
        print("Failed saving results back to supabase:", e)
        traceback.print_exc()

    return ai_json

# -------------------------
# Main loop
# -------------------------
def main():
    print("Entering main loop...")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
            jobs = safe_get_api_data(res) or []

            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            job_id = job.get("id")
            print("🔎 Found job:", job_id)
            # mark processing
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", job_id).execute()

            try:
                process_report(job)
            except Exception as e:
                print("Error while processing job:", e)
                traceback.print_exc()
                # mark failed
                supabase.table("reports").update({"ai_status":"failed", "ai_error": str(e)}).eq("id", job_id).execute()

        except Exception as e:
            print("LOOP ERROR:", type(e).__name__, e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
