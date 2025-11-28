import os
import json
import time
import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

# -----------------------------------
# ENV
# -----------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL:
    raise Exception("Missing SUPABASE_URL")
if not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_SERVICE_ROLE_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------------
# JSON Schema
# -----------------------------------
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
        "disclaimer",
    ],
}

# -----------------------------------
# PDF extraction
# -----------------------------------
def extract_pdf_text(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


# -----------------------------------
# AI INTERPRETATION (FIXED)
# -----------------------------------
def run_ai_interpretation(text: str) -> dict:
    prompt = f"""
You are AMI — Artificial Medical Intelligence.
Produce a structured interpretation and return ONLY valid JSON.

Lab Report:
{text}
"""

    # FIXED — THIS IS THE CORRECT ENDPOINT
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA},
        max_tokens=1500,
    )

    data = response.choices[0].message["content"]
    return json.loads(data)


# -----------------------------------
# SIGNED URL DOWNLOAD
# -----------------------------------
def download_pdf_signed(path: str) -> str:
    signed = supabase.storage.from_("reports").create_signed_url(
        path, expires_in=3600
    )

    url = signed.get("signedURL")
    if not url:
        raise Exception("No signed URL returned")

    print("Signed URL:", url)

    local_path = "/tmp/report.pdf"
    r = requests.get(url)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(r.content)

    return local_path


# -----------------------------------
# PROCESS JOB
# -----------------------------------
def process_next_job():
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .limit(1)
        .execute()
    )

    if not job.data:
        print("No jobs...")
        return

    report = job.data[0]
    report_id = report["id"]
    file_path = report.get("file_path")

    if not file_path:
        err = f"Missing file_path for report {report_id}"
        print(err)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": err
        }).eq("id", report_id).execute()
        return

    print(f"\nProcessing {report_id}...")

    try:
        local_pdf = download_pdf_signed(file_path)
    except Exception as e:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"Signed URL download error: {e}"
        }).eq("id", report_id).execute()
        return

    text = extract_pdf_text(local_pdf)
    if len(text) < 10:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": "PDF extraction failed",
            "extracted_text": text
        }).eq("id", report_id).execute()
        return

    try:
        ai = run_ai_interpretation(text)
    except Exception as e:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"AI error: {e}"
        }).eq("id", report_id).execute()
        return

    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": text,
        "ai_results": ai,
        "cbc_json": ai.get("cbc_values", {}),
        "trend_json": ai.get("trend_summary", [])
    }).eq("id", report_id).execute()

    print("Completed:", report_id)


# -----------------------------------
# MAIN LOOP
# -----------------------------------
def main():
    print("AMI Worker started… watching for jobs.")
    while True:
        try:
            process_next_job()
        except Exception as e:
            print("Worker error:", e)
        time.sleep(4)


if __name__ == "__main__":
    main()
