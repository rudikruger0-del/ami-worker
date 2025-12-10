print(">>> Worker starting, loading imports...")

# -----------------------
# BASE IMPORTS
# -----------------------
import os, time, json, io, traceback, base64
print(">>> Base imports loaded")

# -----------------------
# SUPABASE
# -----------------------
from supabase import create_client, Client
print(">>> Supabase imported")

# -----------------------
# OPENAI
# -----------------------
from openai import OpenAI
print(">>> OpenAI imported")

# -----------------------
# PDF PARSER
# -----------------------
from pypdf import PdfReader
print(">>> pypdf imported")

# -----------------------
# PDF → IMAGE
# -----------------------
from pdf2image import convert_from_bytes
print(">>> pdf2image imported")


# ===================================================================
#                 ENVIRONMENT & CLIENT INITIALIZATION
# ===================================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(">>> Environment variables loaded")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE variables missing")

if not OPENAI_API_KEY:
    print("⚠️  Missing OpenAI API key!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# ===================================================================
#                       PDF TEXT EXTRACTION
# ===================================================================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("❌ PDF parse error:", e)
        return ""


def is_scanned_pdf(text: str) -> bool:
    """Detect scanned PDFs by extremely low text output."""
    if not text:
        return True
    if len(text.strip()) < 30:
        return True
    return False


# ===================================================================
#                     OCR — CBC VALUE EXTRACTION
# ===================================================================
def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """OpenAI Vision OCR — extract CBC + chemistry from scanned PDFs."""

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant for medical laboratory scans.\n"
        "Extract ALL CBC and chemistry analytes, including:\n"
        "- WBC, neutrophils, lymphocytes, monocytes, eosinophils, basophils\n"
        "- RBC, Hb, Hct, MCV, MCH, MCHC, RDW, PLT\n"
        "- Urea, creatinine, eGFR, electrolytes, ALP, ALT, AST, GGT\n"
        "- CK, CK-MB, CRP\n\n"
        "Return STRICT JSON ONLY:\n"
        "{ 'cbc': [ { 'analyte':'', 'value':'', 'units':'', "
        "'reference_low':'', 'reference_high':'' } ] }"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                { "role": "system", "content": system_prompt },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": { "url": f"data:image/png;base64,{b64}" }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )

        raw = response.choices[0].message.content
        return json.loads(raw)

    except Exception as e:
        print("OCR error:", e)
        raise e



# ===================================================================
#                  CLINICAL ROUTE ENGINE (NEW FEATURE)
# ===================================================================
def generate_clinical_routes(cbc: dict) -> list:
    """
    Produces clinical 'routes' based on CBC values
    EXACTLY as the doctor requested.
    """

    routes = []
    # Helper get()
    def g(key):
        return float(cbc.get(key, {}).get("value", -999))

    Hb  = g("Hb")
    MCV = g("MCV")
    MCH = g("MCH")
    MCHC = g("MCHC")

    # -----------------------------
    # 🔻 ANAEMIA ROUTES
    # -----------------------------
    if Hb != -999 and Hb < 12:
        if MCV < 80:
            routes.append("Low Hb + Low MCV → Microcytic pattern → Suggest iron studies (Ferritin) + Reticulocyte count.")
        elif MCV > 100:
            routes.append("Low Hb + High MCV → Macrocytic pattern → Suggest B12/Folate evaluation.")
        else:
            routes.append("Low Hb + Normal MCV → Possible early iron deficiency or anemia of chronic disease.")

    # -----------------------------
    # 🔺 POLYCYTHAEMIA
    # -----------------------------
    if Hb > 16:
        routes.append("High Hb → Consider dehydration vs polycythaemia → Suggest repeat CBC + hydration evaluation.")

    # -----------------------------
    # 🔥 INFECTION / INFLAMMATION
    # -----------------------------
    WBC = g("WBC")
    Neut = g("Neutrophils")
    Lymph = g("Lymphocytes")

    if WBC > 11:
        routes.append("High WBC → Evaluate differential: Neutrophilia suggests bacterial infection; lymphocytosis suggests viral pattern.")
    if Neut > 7.5:
        routes.append("Neutrophilia → Consider bacterial infection or stress response.")
    if Lymph > 4:
        routes.append("Lymphocytosis → Suggest viral infection workup.")

    # -----------------------------
    # 💥 PLATELET ROUTES
    # -----------------------------
    Plt = g("Platelets")

    if Plt < 150:
        routes.append("Low platelets → Consider viral suppression, autoimmune causes, or bone marrow issues.")
    if Plt > 450:
        routes.append("High platelets → Consider inflammation, iron deficiency, or reactive thrombocytosis.")

    return routes



# ===================================================================
#                       AI INTERPRETATION STEP
# ===================================================================
def call_ai_on_report(text: str) -> dict:
    MAX = 14000
    if len(text) > MAX:
        text = text[:MAX]

    system_prompt = (
        "You analyse CBC + chemistry and generate interpretation.\n"
        "DO NOT diagnose. Use safe language.\n"
        "Output STRICT JSON: { patient:{}, cbc:{}, summary:[], routes:[] }"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            { "content": system_prompt, "role": "system" },
            { "content": text, "role": "user" }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    return json.loads(response.choices[0].message.content)



# ===================================================================
#                        PROCESS A SINGLE REPORT
# ===================================================================
def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job["file_path"]

    try:
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)

        # -------------------------------------
        # SCANNED → OCR
        # -------------------------------------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED → OCR")
            pages = convert_from_bytes(pdf_bytes)

            combined = []

            for pg in pages:
                buf = io.BytesIO()
                pg.save(buf, format="PNG")
                try:
                    result = extract_cbc_from_image(buf.getvalue())
                    if "cbc" in result:
                        combined.extend(result["cbc"])
                except Exception as e:
                    print("OCR error:", e)

            if not combined:
                raise ValueError("Vision OCR could not extract CBC values")

            cbc_json = { "cbc": combined }
            merged = json.dumps(cbc_json)

        # -------------------------------------
        # DIGITAL PDF → DIRECT TEXT
        # -------------------------------------
        else:
            print(f"📝 Report {report_id}: DIGITAL PDF → Text interpreter")
            merged = text

        if not merged.strip():
            raise ValueError("No usable content")

        # AI INTERPRETATION
        ai_json = call_ai_on_report(merged)

        # Add clinical routes
        cbc_values = ai_json.get("cbc", {})
        ai_json["routes"] = generate_clinical_routes(cbc_values)

        # Save result
        supabase.table("reports").update(
            { "ai_status": "completed", "ai_results": ai_json, "ai_error": None }
        ).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed")
        return { "success": True }

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing {report_id}: {err}")
        traceback.print_exc()

        supabase.table("reports").update(
            { "ai_status": "failed", "ai_error": err }
        ).eq("id", report_id).execute()

        return { "error": err }



# ===================================================================
#                           WORKER LOOP
# ===================================================================
def main():
    print(">>> Worker started. Waiting for jobs...")

    while True:
        try:
            res = supabase.table("reports") \
                .select("*") \
                .eq("ai_status", "pending") \
                .limit(1) \
                .execute()

            jobs = res.data or []
            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            print(f"🔎 Found job {job['id']}")

            supabase.table("reports").update(
                { "ai_status": "processing" }
            ).eq("id", job["id"]).execute()

            process_report(job)

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
