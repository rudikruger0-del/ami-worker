print(">>> AMI Worker starting, loading imports...")

# -----------------------
# BASE PYTHON IMPORTS
# -----------------------
import os
import time
import json
import io
import traceback
import base64
import re

print(">>> Base imports loaded")

# -----------------------
# THIRD PARTY IMPORTS
# -----------------------
try:
    from supabase import create_client, Client
    print(">>> Supabase imported")
except Exception as e:
    print("❌ Failed importing Supabase:", e)
    raise e

try:
    from openai import OpenAI
    print(">>> OpenAI imported")
except Exception as e:
    print("❌ Failed importing OpenAI:", e)
    raise e

try:
    from pypdf import PdfReader
    print(">>> pypdf imported")
except Exception as e:
    print("❌ Failed importing pypdf:", e)
    raise e

try:
    from pdf2image import convert_from_bytes
    print(">>> pdf2image imported")
except Exception as e:
    print("❌ Failed importing pdf2image:", e)
    raise e


# ======================================================
#   ENV + CLIENTS
# ======================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(">>> Environment variables loaded")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# ======================================================
#   PDF HELPERS
# ======================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable text PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("PDF parse error:", e)
        return ""


def is_scanned_pdf(pdf_text: str) -> bool:
    """If there’s almost no text, assume the PDF is scanned."""
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False


# ======================================================
#   SMALL UTILITIES FOR ROUTE ENGINE
# ======================================================

def to_float(value, default=None):
    """
    Safely convert lab values like '88.0%', '11.6 g/dL', '4.2*' → float.
    Returns default if conversion fails.
    """
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return default

    # Replace comma decimal, keep first number we see
    text = value.replace(",", ".")
    match = re.findall(r"-?\d+\.?\d*", text)
    if not match:
        return default
    try:
        return float(match[0])
    except Exception:
        return default


def index_tests_from_ai(ai_json: dict) -> dict:
    """
    Build a lookup dict: name_lower -> row
    Accepts multiple possible keys to be robust to different AI formats.
    """
    rows = []

    for key in ["cbc", "cbc_values", "chemistry", "chemistry_values", "labs", "lab_values"]:
        val = ai_json.get(key)
        if isinstance(val, list):
            rows.extend(val)
        elif isinstance(val, dict):
            for name, sub in val.items():
                if isinstance(sub, dict):
                    row = sub.copy()
                    row.setdefault("analyte", name)
                    rows.append(row)

    index = {}
    for row in rows:
        name = (
            row.get("analyte")
            or row.get("test")
            or row.get("name")
            or row.get("analyte_name")
        )
        if not name:
            continue
        index[name.strip().lower()] = row

    return index


def get_row(index: dict, *names):
    for n in names:
        if not n:
            continue
        row = index.get(n.strip().lower())
        if row:
            return row
    return {}


def get_val(index: dict, *names, default=None):
    row = get_row(index, *names)
    return to_float(row.get("value"), default)


def get_flag(index: dict, *names):
    row = get_row(index, *names)
    return (row.get("flag") or "").lower()


# ======================================================
#   VISION OCR: EXTRACT CBC TABLE FROM SCANNED IMAGE
# ======================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Sends a scanned image to OpenAI Vision to extract CBC + chemistry values.
    Returns dict like { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans. "
        "Extract ALL CBC and chemistry analytes you can see, including: full blood count, "
        "differential, platelets, ESR (if present), electrolytes, urea, creatinine, eGFR, "
        "bicarbonate / CO2, liver enzymes, CK, CK-MB, CRP and other inflammatory markers. "
        "Return STRICT JSON with this structure:\n"
        "{ 'cbc': [ { 'analyte': '', 'value': '', 'units': '', "
        "'reference_low': '', 'reference_high': '' } ] }\n"
        "Do not summarise, do not interpret – just the table."
    )

    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        },
                    }
                ],
            },
        ],
    )

    content = resp.choices[0].message.content
    if not content:
        raise ValueError("OCR model returned empty content")

    return json.loads(content)


# ======================================================
#   INTERPRETATION MODEL (TEXT ONLY)
# ======================================================

def call_ai_on_report(text: str) -> dict:
    """
    Main interpretation model. We still ask it for narrative + basic structure.
    The route engine will add a 'routes' field afterwards.
    """
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing full blood count (CBC) and chemistry results.\n"
        "You do NOT diagnose or prescribe. You only describe patterns and possible routes.\n\n"
        "Return STRICT JSON with at least these fields:\n"
        "{\n"
        '  "patient": { "age": null, "sex": null },\n'
        '  "cbc": [ { "analyte": "", "value": "", "units": "", '
        '"reference_low": "", "reference_high": "", "flag": "low|normal|high" } ],\n'
        '  "summary": {\n'
        '      "impression": "",\n'
        '      "suggested_follow_up": ""\n'
        "  }\n"
        "}\n"
        "If you cannot find some values, just omit them – never invent numbers."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
    )

    content = resp.choices[0].message.content
    if not content:
        raise ValueError("Interpretation model returned empty content")

    return json.loads(content)


# ======================================================
#   AMI CLINICAL ROUTE ENGINE
# ======================================================

def generate_clinical_routes_from_ai(ai_json: dict) -> list[str]:
    """
    High-level wrapper: build test index and generate routes.
    """
    index = index_tests_from_ai(ai_json)
    if not index:
        return []

    routes: list[str] = []

    # ---------- CORE CBC VALUES ----------
    Hb = get_val(index, "haemoglobin", "hemoglobin", "hb")
    Hb_flag = get_flag(index, "haemoglobin", "hemoglobin", "hb")

    MCV = get_val(index, "mcv")
    MCV_flag = get_flag(index, "mcv")

    MCH = get_val(index, "mch")
    MCH_flag = get_flag(index, "mch")

    RDW = get_val(index, "rdw")
    RDW_flag = get_flag(index, "rdw")

    RBC = get_val(index, "erythrocyte count", "red cell count", "rbc")
    RBC_flag = get_flag(index, "erythrocyte count", "red cell count", "rbc")

    WBC = get_val(index, "leucocyte count", "white cell count", "wbc")
    WBC_flag = get_flag(index, "leucocyte count", "white cell count", "wbc")

    Neut = get_val(index, "neutrophils", "neutrophils %", "neutrophils count")
    Neut_flag = get_flag(index, "neutrophils", "neutrophils %", "neutrophils count")

    Lymph = get_val(index, "lymphocytes", "lymphocytes %", "lymphocyte count")
    Lymph_flag = get_flag(index, "lymphocytes", "lymphocytes %", "lymphocyte count")

    Plt = get_val(index, "platelets", "platelet count")
    Plt_flag = get_flag(index, "platelets", "platelet count")

    CRP = get_val(index, "crp", "c-reactive protein", "crp (c-reactive protein)")
    CRP_flag = get_flag(index, "crp", "c-reactive protein", "crp (c-reactive protein)")

    Bicarb = get_val(index, "st-co2", "bicarbonate", "total co2")
    Na = get_val(index, "s-sodium", "sodium")
    K = get_val(index, "s-potassium", "potassium")
    Urea = get_val(index, "s-urea", "urea")
    Creat = get_val(index, "s-creatinine", "creatinine")

    ALT = get_val(index, "alt", "alanine aminotransferase")
    AST = get_val(index, "ast", "aspartate aminotransferase")
    ALP = get_val(index, "alp", "alkaline phosphatase")
    GGT = get_val(index, "ggt", "gamma glutamyl transferase")
    Bili = get_val(index, "bilirubin", "total bilirubin")

    # ---------- 1. ANAEMIA ROUTES ----------
    if Hb_flag == "low":
        # microcytic
        if MCV_flag == "low" or (MCV and MCV < 80):
            if MCH_flag == "low" or RDW_flag == "high":
                routes.append(
                    "Low haemoglobin with microcytosis and hypochromia "
                    "(low MCV/MCH and/or high RDW) – route suggests iron deficiency "
                    "pattern. Consider ferritin, transferrin saturation and "
                    "reticulocyte count; screen for chronic blood loss "
                    "(GI, gynaecological) and dietary iron intake."
                )
            else:
                routes.append(
                    "Low haemoglobin with low MCV but normal MCH/RDW – consider "
                    "iron deficiency, thalassaemia trait or anaemia of chronic disease. "
                    "Route: request ferritin, CRP and, if indicated, haemoglobinopathy screen."
                )

        # normocytic
        elif (MCV_flag == "normal" or (MCV and 80 <= MCV <= 100)) and RBC_flag != "low":
            routes.append(
                "Normocytic anaemia – route suggests checking for chronic disease, "
                "renal impairment, endocrine causes or early blood loss. "
                "Consider urea/creatinine/eGFR, CRP/ESR and reticulocyte count."
            )

        # macrocytic
        elif MCV_flag == "high" or (MCV and MCV > 100):
            routes.append(
                "Low haemoglobin with macrocytosis (high MCV) – route suggests "
                "B12/folate deficiency, alcohol excess, liver disease or "
                "bone marrow disorder. Consider serum B12, folate, liver "
                "function tests and review medications (e.g. chemotherapy, "
                "anticonvulsants)."
            )

        # safety
        if Hb is not None and Hb < 8:
            routes.append(
                "Very low haemoglobin – consider urgent same-day clinical review, "
                "repeat FBC and assessment for haemodynamic instability or rapid blood loss."
            )

    # ---------- 2. ERYTHROCYTOSIS ----------
    if Hb_flag == "high" or RBC_flag == "high":
        routes.append(
            "Raised haemoglobin and/or red cell count – route suggests assessing "
            "for dehydration, smoking, chronic hypoxia (lung/cardiac disease) or "
            "primary polycythaemia. Check history, oxygen saturation and consider "
            "repeat FBC when well hydrated."
        )

    # ---------- 3. INFECTION / INFLAMMATION ROUTES ----------
    if (WBC_flag == "high" or Neut_flag == "high") and (CRP_flag == "high" or (CRP and CRP > 10)):
        routes.append(
            "Neutrophilia with elevated CRP – route suggests acute infection or "
            "inflammatory process. Correlate with symptoms (fever, focal pain, "
            "respiratory or urinary symptoms) and consider source-directed work-up."
        )

    if WBC_flag == "high" and Lymph_flag == "high" and (Neut_flag != "high"):
        routes.append(
            "Leucocytosis with relative lymphocytosis – route suggests viral infection, "
            "post-viral state or possible lymphoproliferative disease if persistent. "
            "Consider repeat FBC in 4–6 weeks and clinical correlation."
        )

    if WBC_flag == "low" or Neut_flag == "low":
        routes.append(
            "Low white cell / neutrophil count – route suggests reviewing recent "
            "viral illness, medications (especially chemotherapy, immunosuppressants) "
            "and nutritional status. If febrile or clinically unwell, "
            "consider urgent assessment for neutropenic sepsis."
        )

    # ---------- 4. PLATELET / BLEEDING ROUTES ----------
    if Plt_flag == "low":
        routes.append(
            "Thrombocytopenia – route suggests reviewing medications (e.g. heparin, "
            "antibiotics), alcohol intake, viral infections and autoimmune history. "
            "Consider repeat FBC, blood film and review for bleeding or bruising."
        )

    if Plt_flag == "high":
        routes.append(
            "Thrombocytosis – can be reactive to infection, inflammation or iron "
            "deficiency, but may rarely reflect a myeloproliferative process. "
            "Correlate with CRP, iron status and clinical context; consider repeat "
            "count once acute illness has settled."
        )

    # ---------- 5. PANCYTOPENIA / MARROW RED FLAGS ----------
    low_lineages = 0
    if Hb_flag == "low":
        low_lineages += 1
    if WBC_flag == "low":
        low_lineages += 1
    if Plt_flag == "low":
        low_lineages += 1

    if low_lineages >= 2:
        routes.append(
            "More than one cell line is reduced (e.g. anaemia with leukopenia and/or "
            "thrombocytopenia) – route suggests possible bone marrow suppression, "
            "infiltration or severe systemic illness. Consider urgent haematology "
            "discussion, blood film and repeat FBC."
        )

    # ---------- 6. RENAL / ELECTROLYTE ROUTES ----------
    if (Urea and Urea > 8) or (Creat and Creat > 100):
        routes.append(
            "Raised urea and/or creatinine – route suggests assessing renal function "
            "trend, hydration status, blood pressure and medication list "
            "(ACE-inhibitors, NSAIDs, diuretics). Consider eGFR and urinalysis."
        )

    if Na is not None and (Na < 135 or Na > 145):
        routes.append(
            "Abnormal sodium – route suggests checking fluid balance, medications "
            "(diuretics, antidepressants), adrenal/thyroid status and serum/urine osmolality "
            "if significantly deranged or symptomatic."
        )

    if K is not None and (K < 3.3 or K > 5.3):
        routes.append(
            "Abnormal potassium – route suggests urgent ECG if significantly high or low, "
            "review of medications (ACE-inhibitors, ARBs, diuretics) and assessment for "
            "renal impairment or acid–base disturbance."
        )

    if Bicarb is not None and Bicarb < 21:
        routes.append(
            "Low bicarbonate / total CO₂ – route suggests metabolic acidosis. "
            "Correlate with lactate, ketones, renal function and clinical state "
            "to distinguish sepsis, renal failure, ketoacidosis or toxin ingestion."
        )

    # ---------- 7. LIVER PATTERN ROUTES ----------
    if (ALT and ALT > 50) or (AST and AST > 50) or (ALP and ALP > 130) or (GGT and GGT > 60) or (Bili and Bili > 20):
        routes.append(
            "Abnormal liver biochemistry – route suggests pattern analysis "
            "(hepatocellular vs cholestatic) and correlation with alcohol use, "
            "medications, viral hepatitis risk and metabolic risk factors. "
            "Consider hepatitis screen, ultrasound and repeat testing if persistent."
        )

    # ---------- 8. INTEGRATED ANAEMIA + INFLAMMATION ROUTE ----------
    if Hb_flag == "low" and (CRP_flag == "high" or (CRP and CRP > 10)):
        routes.append(
            "Anaemia with raised CRP – route suggests anaemia of inflammation / chronic disease "
            "or iron deficiency in the context of chronic inflammatory conditions. "
            "Consider ferritin, transferrin saturation, CRP trend and evaluation for "
            "chronic infection, autoimmune disease or malignancy."
        )

    # ---------- 9. SAFETY CATCH-ALL ----------
    if not routes:
        routes.append(
            "No major route-level abnormalities detected from the available results. "
            "Interpretation should still be correlated with the patient’s symptoms, "
            "history and prior results."
        )

    # De-duplicate while preserving order
    seen = set()
    unique_routes = []
    for r in routes:
        if r not in seen:
            seen.add(r)
            unique_routes.append(r)

    return unique_routes


# ======================================================
#   CORE JOB PROCESSING
# ======================================================

def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path or str(file_path).strip() == "":
            err = f"Missing file_path for report {report_id}"
            print("⚠️", err)
            supabase.table("reports").update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
            return {"error": err}

        # Download original PDF
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        print(f"Report {report_id}: scanned={scanned}, text_length={len(text)}")

        # ------- SCANNED PDF → OCR TABLE -------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → OCR pipeline")
            images = convert_from_bytes(pdf_bytes)
            combined_rows = []

            for img in images:
                img_bytes_io = io.BytesIO()
                img.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                try:
                    ocr_result = extract_cbc_from_image(img_bytes)
                    if isinstance(ocr_result, dict) and "cbc" in ocr_result:
                        combined_rows.extend(ocr_result["cbc"])
                except Exception as e:
                    print("Vision OCR error:", e)

            if not combined_rows:
                raise ValueError("Vision OCR could not extract CBC values")

            # Use OCR JSON as text input for interpreter
            merged_text = json.dumps({"cbc": combined_rows}, ensure_ascii=False)

        # ------- DIGITAL PDF → DIRECT TEXT -------
        else:
            print(f"📝 Report {report_id}: Digital PDF detected → text interpreter")
            merged_text = text or l_text

        if not merged_text.strip():
            raise ValueError("No usable content extracted for AI processing")

        # ------- MAIN INTERPRETATION -------
        ai_json = call_ai_on_report(merged_text)

        # ------- ROUTE ENGINE -------
        try:
            routes = generate_clinical_routes_from_ai(ai_json)
            ai_json["routes"] = routes
        except Exception as e:
            print("Route engine error:", e)
            traceback.print_exc()

        supabase.table("reports").update(
            {
                "ai_status": "completed",
                "ai_results": ai_json,
                "ai_error": None,
            }
        ).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_json}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        supabase.table("reports").update(
            {"ai_status": "failed", "ai_error": err}
        ).eq("id", report_id).execute()

        return {"error": err}


# ======================================================
#   WORKER LOOP
# ======================================================

def main():
    print(">>> Entering main worker loop…")
    print("AMI Worker with Vision OCR + Route Engine started – watching for jobs.")

    while True:
        try:
            res = (
                supabase.table("reports")
                .select("*")
                .eq("ai_status", "pending")
                .limit(1)
                .execute()
            )

            jobs = res.data or []

            if jobs:
                job = jobs[0]
                job_id = job["id"]
                print(f"🔎 Found job {job_id}")

                supabase.table("reports").update(
                    {"ai_status": "processing"}
                ).eq("id", job_id).execute()

                process_report(job)
            else:
                print("No jobs...")

            time.sleep(1)

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
