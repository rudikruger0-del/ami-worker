import os
import json
import time
import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

# ------------------------------
# ENV VARIABLES (AUTO-DETECT)
# ------------------------------
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
print("OpenAI Key Loaded:", bool(OPENAI_API_KEY))

# ------------------------------
# CLIENTS
# ------------------------------
supabase: Client = create_client(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY
)

client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# STRICT JSON SCHEMA
# ------------------------------
JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "array", "items": {"type": "string"}},
        "trend_summary": {"type": "array", "items": {"type": "string"}},
        "flagged_results": {"type": "array", "items": {"type": "string"}},
        "interpretation": {"type": "array", "items": {"type": "string"}},
        "risk_level": {"type": "string"},
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "when_to_seek_urgent_care": {"type": "array", "items": {"type": "string"}},
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
        "disclaimer"
    ]
}

# ------------------------------
# PDF TEXT EXTRACTION
# ------------------------------
def extract_pdf_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


# ------------------------------
# AI INTERPRETATION
# ------------------------------
def run_ai_interpretation(extracted_text: str) -> dict:
    prompt = f"""
You are AMI — Artificial Medical Intelligence.
Provide strong medical interpretation with trends, risks, flagged values.
Return ONLY valid JSON.

Lab Report:
-----------
{extracted_text}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2000,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA}
    )

    return json.loads(response.output[0].content[0].text)


# ------------------------------
# DOWNLOAD PDF
# ------------------------------
def download_pdf(url: str, local_path: str):
    r = requests.get(url)
    r.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(r.content)


# ------------------------------
# PROCESS JOB
# ------------------------------
def process_next_job():
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .limit(1)
        .execute()
    )

    if not job.data:
        print("No queued jobs.")
        return

    report = job.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print("\n-----------------------------------")
    print("Processing report:", report_id)
    print("-----------------------------------\n")

    pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/{file_path}"
    local_file = "/tmp/report.pdf"

    # Download PDF
    try:
        download_pdf(pdf_url, local_file)
    except Exception as e:
        print("Download error:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"Download error: {e}"
        }).eq("id", report_id).execute()
        return

    # Extract text
    extracted_text = extract_pdf_text(local_file)

    if len(extracted_text) < 20:
        print("Unreadable PDF")
        supabase.table("reports").update({
            "ai_status": "failed",
            "extracted_text": extracted_text,
            "ai_error": "PDF extraction produced no readable text"
        }).eq("id", report_id).execute()
        return

    # AI interpretation
    try:
        result_json = run_ai_interpretation(extracted_text)
    except Exception as e:
        print("AI error:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"AI error: {e}"
        }).eq("id", report_id).execute()
        return

    # Save results
    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": extracted_text,
        "ai_results": result_json,
        "cbc_json": result_json.get("cbc_values", {}),
        "trend_json": result_json.get("trend_summary", [])
    }).eq("id", report_id).execute()

    print("Report completed:", report_id)


# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    print("AMI Worker started. Ready for reports...")
    while True:
        try:
            process_next_job()
        except Exception as e:
            print("Worker error:", e)
        time.sleep(4)


if __name__ == "__main__":
    main()
