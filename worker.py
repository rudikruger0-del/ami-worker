print(">>> Worker starting, loading imports...")

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
    print(">>> Base imports loaded")
except Exception as e:
    print("❌ Failed loading base imports:", e)
    raise e


# -----------------------
# SUPABASE IMPORT
# -----------------------
try:
    from supabase import create_client, Client
    print(">>> Supabase imported")
except Exception as e:
    print("❌ Failed importing Supabase:", e)
    raise e


# -----------------------
# OPENAI IMPORT
# -----------------------
try:
    from openai import OpenAI
    print(">>> OpenAI imported")
except Exception as e:
    print("❌ Failed importing OpenAI:", e)
    raise e


# -----------------------
# PYPDF IMPORT
# -----------------------
try:
    from pypdf import PdfReader
    print(">>> pypdf imported")
except Exception as e:
    print("❌ Failed importing pypdf:", e)
    raise e


# -----------------------
# PDF2IMAGE IMPORT
# -----------------------
try:
    from pdf2image import convert_from_bytes
    print(">>> pdf2image imported")
except Exception as e:
    print("❌ Failed importing pdf2image:", e)
    raise e



# ======================================================
#   START REAL WORKER CODE (unchanged below)
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


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
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
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False


# ---------------------------------------------------------
# IMAGE OCR → Extract CBC
# ---------------------------------------------------------
def extract_cbc_from_image(image_bytes: bytes) -> dict:
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans. "
        "Extract ALL CBC-related analytes INCLUDING chemistry, liver enzymes, CK, CK-MB, creatinine, electrolytes. "
        "Return STRICT JSON: { 'cbc': [ { 'analyte': '', 'value': '', 'units': '', "
        "'reference_low': '', 'reference_high': '' } ] }"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ---------------------------------------------------------
# INTERPRETATION MODEL
# ---------------------------------------------------------
def call_ai_on_report(text: str) -> dict:
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing CBC and chemistry results. "
        "Do NOT diagnose. Only describe lab patterns.\n\n"
        "Output STRICT JSON with fields: patient, cbc, summary."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ---------------------------------------------------------
# REPORT PROCESSOR
# ---------------------------------------------------------
def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path:
            err = f"Missing file_path for report {report_id}"
            supabase.table("reports").update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
            return {"error": err}

        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)

        cbc_json = None

        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → Using OCR")

            images = convert_from_bytes(pdf_bytes)
            combined_cbc = []

            for img in images:
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                img_bytes = img_bytes.getvalue()

                try:
                    result = extract_cbc_from_image(img_bytes)
                    if "cbc" in result:
                        combined_cbc.extend(result["cbc"])
                except Exception as e:
                    print("Vision OCR error:", e)

            if not combined_cbc:
                raise ValueError("Vision OCR failed to extract CBC values")

            cbc_json = {"cbc": combined_cbc}
            merged_text = json.dumps(cbc_json)

        else:
            print(f"📝 Report {report_id}: Digital PDF detected → Using text parser")
            merged_text = text or l_text

        if not merged_text.strip():
            raise ValueError("No usable content extracted")

        ai_json = call_ai_on_report(merged_text)

        supabase.table("reports").update(
            {"ai_status": "completed", "ai_results": ai_json, "ai_error": None}
        ).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed successfully")
        return {"success": True}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        supabase.table("reports").update(
            {"ai_status": "failed", "ai_error": err}
        ).eq("id", report_id).execute()

        return {"error": err}


# ---------------------------------------------------------
# WORKER LOOP
# ---------------------------------------------------------
def main():
    print(">>> Entering main worker loop...")
    print("AMI Worker with Vision OCR started… watching for jobs.")

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
