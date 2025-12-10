print(">>> AMI Worker starting, loading imports...")

# -----------------------
# BASE PYTHON IMPORTS
# -----------------------
try:
    import os
    import time
    import json
    import io
    import traceback
    import base64
    import re
    print(">>> Base imports loaded")
except Exception as e:
    print("❌ Failed loading base imports:", e)
    raise e

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
#   SMALL UTILITIES
# ======================================================

def clean_number(val):
    """
    Safely convert lab values like '88.0%', '11,6 g/dL', '4.2*' → float.
    Returns None if conversion fails.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val)
    # Replace comma decimal + remove percent/star/units
    s = s.replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


# ======================================================
#   VISION OCR: EXTRACT CBC TABLE FROM SCANNED IMAGE
#   (FIXED FOR NEW OPENAI API)
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

    resp = client.responses.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            },
        ],
    )

    # For response_format=json_object, this is already a Python dict
    try:
        raw = resp.output[0].content[0].json
    except Exception as e:
        print("❌ Unexpected OCR response structure:", e, getattr(resp, "output", None))
        raise

    if not isinstance(raw, dict):
        raise ValueError("OCR result is not a dict JSON object")

    return raw


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
        '  \"patient\": { \"name\": null, \"age\": null, \"sex\": \"Unknown\" },\n'
        '  \"cbc\": [ { \"analyte\": \"\", \"value\": \"\", \"units\": \"\", '
        '\"reference_low\": \"\", \"reference_high\": \"\", \"flag\": \"low|normal|high|unknown\" } ],\n'
        '  \"summary\": {\n'
        '      \"impression\": \"\",\n'
        '      \"suggested_follow_up\": \"\"\n'
        "  }\n"
        "}\n"
        "If you cannot find some values, just omit them – never invent numbers."
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        input=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            },
        ],
    )

    try:
        ai_json = resp.output[0].content[0].json
    except Exception as e:
        print("❌ Unexpected interpretation response structure:", e, getattr(resp, "output", None))
        raise

    if not isinstance(ai_json, dict):
        raise ValueError("Interpretation result is not a dict JSON object")

    return ai_json


# ======================================================
#   BUILD CANONICAL CBC VALUE DICT FOR ROUTES
# ======================================================

def build_cbc_value_dict(ai_json: dict) -> dict:
    """
    Turn ai_json["cbc"] (list) into a dict like:
    {
      "Hb": {"value": "...", ...},
      "WBC": {...},
      "MCV": {...},
      ...
    }
    so the route engine can work on consistent names.
    """
    cbc_values = {}

    rows = ai_json.get("cbc") or []
    if not isinstance(rows, list):
        return cbc_values

    for row in rows:
        if not isinstance(row, dict):
            continue
        name = (row.get("analyte") or row.get("test") or "").lower()
        if not name:
            continue

        def put(key):
            if key not in cbc_values:
                cbc_values[key] = row

        # Red cells
        if "haemoglobin" in name or "hemoglobin" in name or name == "hb":
            put("Hb")
        elif name.startswith("mcv"):
            put("MCV")
        elif name.startswith("mch"):
            put("MCH")
        elif "red cell" in name or "rbc" in name:
            put("RBC")

        # White cells
        elif "white cell" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            put("WBC")
        elif "neutrophil" in name:
            put("Neutrophils")
        elif "lymphocyte" in name:
            put("Lymphocytes")

        # Platelets
        elif "platelet" in name or "plt" in name:
            put("Platelets")

        # Renal
        elif "creatinine" in name:
            put("Creatinine")
        elif name.startswith("urea"):
            put("Urea")

        # LFT
        elif name == "alt" or "alanine aminotransferase" in name:
            put("ALT")
        elif name == "ast" or "aspartate aminotransferase" in name:
            put("AST")
        elif name == "alp" or "alkaline phosphatase" in name:
            put("ALP")
        elif "ggt" in name or "gamma glutamyl" in name:
            put("GGT")
        elif "bilirubin" in name:
            put("Bilirubin")

        # Muscle
        elif name == "ck" or "creatine kinase" in name:
            put("CK")
        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")

        # Electrolytes
        elif "sodium" in name or name == "na":
            put("Sodium")
        elif "potassium" in name or name == "k":
            put("Potassium")
        elif "calcium" in name or name.startswith("ca"):
            put("Calcium")

        # Inflammation
        elif name.startswith("crp") or "c-reactive" in name:
            put("CRP")

    return cbc_values


# ======================================================
#   CBC + CHEMISTRY ROUTE ENGINE
# ======================================================

def generate_clinical_routes(cbc_values: dict):
    """
    Takes parsed CBC + chemistry values → generates diagnostic routes (roetes).
    """

    routes = []

    # Helper getter
    def val(name):
        return clean_number(cbc_values.get(name, {}).get("value"))

    # CBC values
    Hb  = val("Hb")
    MCV = val("MCV")
    MCH = val("MCH")
    WBC = val("WBC")
    Plt = val("Platelets")
    Neut = val("Neutrophils")
    Lymph = val("Lymphocytes")

    # Chemistry
    Cr   = val("Creatinine")
    Urea = val("Urea")
    ALT  = val("ALT")
    AST  = val("AST")
    ALP  = val("ALP")
    GGT  = val("GGT")
    Bili = val("Bilirubin")
    CK   = val("CK")
    CKMB = val("CK-MB")

    Na = val("Sodium")
    K  = val("Potassium")
    Ca = val("Calcium")
    CRP = val("CRP")

    # -----------------------------
    # ANAEMIA ENGINE
    # -----------------------------
    if Hb is not None and Hb < 13:
        routes.append("🔻 Hb low → anaemia route triggered.")

        if MCV is not None:
            if MCV < 80:
                routes.append("➡️ Low MCV → microcytic anaemia pattern.")
                routes.append("   • Suggest ferritin, iron studies and reticulocyte count.")
            elif 80 <= MCV <= 100:
                routes.append("➡️ Normal MCV → normocytic anaemia pattern.")
                routes.append("   • Consider chronic inflammation, renal disease, early iron deficiency.")
            else:
                routes.append("➡️ High MCV → macrocytic anaemia pattern.")
                routes.append("   • Suggest B12, folate, liver function tests and medication review.")

    # -----------------------------
    # WBC ROUTES
    # -----------------------------
    if WBC is not None:
        if WBC > 12:
            routes.append("🔺 WBC raised → inflammatory / infective route.")
            if Neut and Neut > 70:
                routes.append("   • Neutrophils high → bacterial pattern more likely.")
            if Lymph and Lymph > 45:
                routes.append("   • Lymphocytes high → viral pattern or recovery phase.")
        elif WBC < 4:
            routes.append("🔻 WBC low → consider viral suppression, bone marrow issues or medication effect.")

    # -----------------------------
    # PLATELETS
    # -----------------------------
    if Plt is not None:
        if Plt < 150:
            routes.append("🔻 Platelets low → bleeding risk assessment route.")
        elif Plt > 450:
            routes.append("🔺 Platelets high → reactive thrombocytosis (infection/inflammation/iron deficiency) vs primary process.")

    # -----------------------------
    # KIDNEY FUNCTION ROUTES
    # -----------------------------
    if Cr is not None:
        if Cr > 120:
            routes.append("🔺 Creatinine high → renal function route. Suggest repeat U&E, review medications and hydration status.")
        else:
            routes.append("✔ Creatinine within expected range (interpret with eGFR and clinical context).")

    # -----------------------------
    # LIVER FUNCTION ROUTES
    # -----------------------------
    liver_markers = [ALT, AST, ALP, GGT, Bili]
    if any(x is not None and x > 1.5 for x in liver_markers):
        routes.append("🔺 Liver enzymes / bilirubin abnormal → liver pattern route.")
        if ALT and ALT > 3 * 40:
            routes.append("   • ALT markedly raised → hepatocellular injury pattern.")
        if ALP and ALP > 120:
            routes.append("   • ALP raised → cholestatic/obstructive pattern.")
        if Bili and Bili > 20:
            routes.append("   • Bilirubin raised → jaundice evaluation (haemolytic, hepatic or obstructive).")

    # -----------------------------
    # MUSCLE INJURY ROUTES
    # -----------------------------
    if CK and CK > 300:
        routes.append("🔺 CK elevated → muscle injury route.")
        if CK > 2000:
            routes.append("   • Very high CK → consider rhabdomyolysis physiology; check renal function and hydration.")

    # -----------------------------
    # ELECTROLYTES
    # -----------------------------
    if K is not None:
        if K < 3.3:
            routes.append("⚠️ Potassium low → arrhythmia risk. Urgent clinical review advised if symptomatic.")
        elif K > 5.5:
            routes.append("⚠️ Potassium high → significant arrhythmia risk. Consider urgent ECG & review.")

    if Na is not None:
        if Na < 133:
            routes.append("🔻 Sodium low → hyponatraemia route. Assess fluid status, medications and endocrine causes.")
        elif Na > 146:
            routes.append("🔺 Sodium high → hypernatraemia route. Often dehydration or water loss; assess mental status and fluids.")

    # -----------------------------
    # INFLAMMATION
    # -----------------------------
    if CRP is not None:
        if CRP > 10:
            routes.append("🔺 CRP raised → active inflammation/infection route. Correlate with clinical picture and WBC pattern.")
        else:
            routes.append("✔ CRP not raised → significant systemic inflammation less likely (within limits of test).")

    # END
    if not routes:
        routes.append("No major route-level abnormalities triggered from available results. Correlate with symptoms and prior results.")

    return routes


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

        # Download PDF
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
            cbc_values = build_cbc_value_dict(ai_json)
            routes = generate_clinical_routes(cbc_values)
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
