import os
import json
import time
import io

from dotenv import load_dotenv
from pypdf import PdfReader
from supabase import create_client, Client
from openai import OpenAI

# ---------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or None
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL:
    raise Exception("Missing SUPABASE_URL")
if not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY")

print("Supabase URL Loaded:", bool(SUPABASE_URL))
print("Supabase Key Loaded:", bool(SUPABASE_KEY))
print("OpenAI Key Loaded :", bool(OPENAI_API_KEY))

# ---------------------------------------------------
# CLIENTS
# ---------------------------------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------
# STRICT JSON SCHEMA (inside ai_results.json)
# ---------------------------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": {"type": "string"}},
        "trend_summary": {"type": "array", "items": {"type": "string"}},
        "flagged_results": {"type": "array", "items": {"type": "string"}},
        "interpretation": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "when_to_seek_urgent_care": {
            "type": "array",
            "items": {"type": "string"},
        },
        "cbc_values": {"type": "object"},
        "chemistry_values": {"type": "object"},
        "disclaimer": {"type": "string"},
    },
    "required": [
        "summary",
        "trend_summary",
        "flagged_results",
        "interpretation",
        "risk_level",
        "recommendations",
        "when_to_seek_urgent_care",
        "cbc_values",
        "chemistry_values",
        "disclaimer",
    ],
}


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------
def extract_pdf_text_from_bytes(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF given as bytes using pypdf.
    We never write the patient PDF to disk – keeps POPIA safer.
    """
    try:
        buffer = io.BytesIO(pdf_bytes)
        reader = PdfReader(buffer)
        parts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            parts.append(txt)
        text = "\n".join(parts).strip()

        # Safety: hard cap on length so we don't burn tokens
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars]

        return text
    except Exception as e:
        print("PDF extraction error:", repr(e))
        return ""


def run_ai_interpretation(extracted_text: str) -> dict:
    """
    Call OpenAI Responses API with structured output.
    Returns a Python dict that matches JSON_SCHEMA.
    """
    prompt = f"""
You are AMI — Artificial Medical Intelligence.

You are interpreting a blood / lab report for a medical doctor.
Be concise but clinically rich and practical. Focus on patterns, flags,
and actions that a GP or specialist can actually use.

Rules:
- Use plain clinical language, not patient-facing.
- Highlight abnormal/flagged values and patterns (e.g. anemia, infection, renal risk).
- Always include a clear overall risk level: "low", "moderate", or "high".
- Always fill every field in the JSON schema, even if some arrays are empty.
- Do NOT include any free text outside the JSON.

Lab Report Text
----------------
{extracted_text}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=1800,
        response_format={
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        },
    )

    # Response API: first output, first content block
    raw = response.output[0].content[0].text
    return json.loads(raw)


def download_pdf_bytes_from_storage(file_path: str) -> bytes:
    """
    Download the PDF from Supabase Storage (private bucket).
    Uses service role key, so it ignores RLS but is never public.
    """
    # Your bucket name is "reports"
    bucket_name = "reports"

    # For POPIA safety, we do NOT construct any public URL here.
    # We directly fetch the raw bytes via storage API.
    pdf_bytes = supabase.storage.from_(bucket_name).download(file_path)
    return pdf_bytes


# ---------------------------------------------------
# CORE JOB PROCESSING
# ---------------------------------------------------
def process_next_job():
    # Find oldest queued report
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not job.data:
        print("No queued jobs.")
        return

    report = job.data[0]
    report_id = report["id"]
    file_path = report.get("file_path")

    print("\n----------------------------------------")
    print("Processing report ID:", report_id)
    print("File path:", file_path)
    print("----------------------------------------\n")

    if not file_path:
        print("ERROR: report has no file_path.")
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_error": "No file_path set on report row",
            }
        ).eq("id", report_id).execute()
        return

    # 1) Download from private Storage
    try:
        pdf_bytes = download_pdf_bytes_from_storage(file_path)
    except Exception as e:
        print("Storage download error:", repr(e))
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                # store generic error; does not contain PHI
                "ai_error": f"Storage download error: {str(e)}",
            }
        ).eq("id", report_id).execute()
        return

    # 2) Extract text
    extracted_text = extract_pdf_text_from_bytes(pdf_bytes)

    if not extracted_text or len(extracted_text) < 20:
        print("Unreadable or empty PDF text.")
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "extracted_text": extracted_text,
                "ai_error": "PDF extraction produced insufficient readable text",
            }
        ).eq("id", report_id).execute()
        return

    # 3) Run AI interpretation
    try:
        result_json = run_ai_interpretation(extracted_text)
    except Exception as e:
        print("AI error:", repr(e))
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "extracted_text": extracted_text,
                "ai_error": f"AI error: {str(e)}",
            }
        ).eq("id", report_id).execute()
        return

    # 4) Save AI results
    try:
        # To keep your existing frontend working:
        # wrap the JSON under "json" key.
        ai_results_wrapped = {"json": result_json}

        cbc_json = result_json.get("cbc_values", {}) or {}
        trend_json = result_json.get("trend_summary", []) or []

        supabase.table("reports").update(
            {
                "ai_status": "completed",
                "extracted_text": extracted_text,
                "ai_results": ai_results_wrapped,
                "cbc_json": cbc_json,
                "trend_json": trend_json,
                "ai_error": None,
            }
        ).eq("id", report_id).execute()

        print("Report completed:", report_id)
    except Exception as e:
        print("DB update error:", repr(e))
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_error": f"DB update error after AI: {str(e)}",
            }
        ).eq("id", report_id).execute()


# ---------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------
def main():
    print("AMI Worker started (POPIA-safe). Listening for jobs...")
    while True:
        try:
            process_next_job()
        except Exception as e:
            # Generic log only – no PHI
            print("Worker runtime error:", repr(e))
        time.sleep(4)


if __name__ == "__main__":
    main()
