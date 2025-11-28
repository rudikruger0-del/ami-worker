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
# ENV
# ------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Supabase URL Loaded:", bool(SUPABASE_URL))
print("Supabase Key Loaded:", bool(SUPABASE_KEY))
print("OpenAI Key Loaded:", bool(OPENAI_API_KEY))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
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
# PDF EXTRACT
# ------------------------------
def extract_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            text += (p.extract_text() or "") + "\n"
        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


# ------------------------------
# AI
# ------------------------------
def run_ai(text: str) -> dict:
    prompt = f"""
You are AMI — Artificial Medical Intelligence.
Provide detailed CBC interpretation with trends, risks, flagged values.

Return ONLY valid JSON matching the schema.

Lab Report Text:
{text}
"""

    r = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA}
    )

    return json.loads(r.output[0].content[0].text)


# ------------------------------
# MAIN PROCESSOR
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

    print("\n==============================")
    print("Processing:", report_id)
    print("==============================\n")

    # CORRECT PUBLIC URL
    pdf_url = f"{SUPABASE_URL}/storage/v1/object/public/reports/{file_path}"
    local_pdf = "/tmp/report.pdf"

    # Download PDF
    try:
        r = requests.get(pdf_url)
        r.raise_for_status()
        with open(local_pdf, "wb") as f:
            f.write(r.content)
    except Exception as e:
        print("Download error:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"Download error: {e}"
        }).eq("id", report_id).execute()
        return

    # Extract PDF
    extracted = extract_pdf_text(local_pdf)

    if len(extracted) < 10:
        supabase.table("reports").update({
            "ai_status": "failed",
            "extracted_text": extracted,
            "ai_error": "Empty or unreadable PDF."
        }).eq("id", report_id).execute()
        return

    # AI step
    try:
        result = run_ai(extracted)
    except Exception as e:
        print("AI error:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"AI error: {e}"
        }).eq("id", report_id).execute()
        return

    # Save to DB
    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": extracted,
        "ai_results": result,
        "cbc_json": result.get("cbc_values", {}),
        "trend_json": result.get("trend_summary", [])
    }).eq("id", report_id).execute()

    print("Report completed:", report_id)


# ------------------------------
# LOOP
# ------------------------------
def main():
    print("AMI Worker running...")
    while True:
        try:
            process_next_job()
        except Exception as e:
            print("Worker crash:", e)
        time.sleep(4)


if __name__ == "__main__":
    main()
