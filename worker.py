import os
import json
import time
from pypdf import PdfReader
from supabase import create_client, Client
from dotenv import load_dotenv
from openai import OpenAI
import requests

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Strict JSON schema the AI must follow
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


def extract_pdf_text(file_path):
    try:
        reader = PdfReader(file_path)
        text = ""

        for page in reader.pages:
            txt = page.extract_text() or ""
            text += txt + "\n"

        return text.strip()
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


def run_ai_interpretation(extracted_text):
    prompt = f"""
You are AMI — Artificial Medical Intelligence.

Analyse this lab report text and produce DOCTOR-LEVEL detail with patterns, trends, flagged values, and clinical interpretation.

Return ONLY the JSON object. No text outside JSON.

Lab Report Text:
----------------
{extracted_text}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=2500,
        response_format={"type": "json_schema", "json_schema": JSON_SCHEMA}
    )

    return json.loads(response.output[0].content[0].text)


def download_pdf(public_url, local_path):
    r = requests.get(public_url)
    with open(local_path, "wb") as f:
        f.write(r.content)


def process_next_job():
    job = (
        supabase.table("ami_tasks")
        .select("*")
        .eq("ai_status", "queued")
        .limit(1)
        .execute()
    )

    if not job.data:
        print("No queued jobs.")
        return

    task = job.data[0]
    task_id = task["id"]
    pdf_path = task["pdf_path"]

    print("Processing task:", task_id)

    # PUBLIC URL from Supabase Storage
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{pdf_path}"
    local_file = "/tmp/report.pdf"

    try:
        download_pdf(public_url, local_file)
    except:
        supabase.table("ami_tasks").update({"ai_status": "failed"}).eq("id", task_id).execute()
        print("Failed: Could not download PDF.")
        return

    # Extract
    extracted_text = extract_pdf_text(local_file)

    if len(extracted_text.strip()) < 20:
        supabase.table("ami_tasks").update({
            "ai_status": "failed",
            "extracted_text": extracted_text
        }).eq("id", task_id).execute()
        print("Failed: No readable text.")
        return

    # AI INTERPRETATION
    try:
        result_json = run_ai_interpretation(extracted_text)
    except Exception as e:
        print("AI error:", e)
        supabase.table("ami_tasks").update({"ai_status": "failed"}).eq("id", task_id).execute()
        return

    # SAVE RESULTS
    supabase.table("ami_tasks").update({
        "ai_status": "completed",
        "extracted_text": extracted_text,
        "ai_results": result_json,
        "cbc_json": result_json.get("cbc_values", {}),
        "trend_json": result_json.get("trend_summary", [])
    }).eq("id", task_id).execute()

    print("Completed:", task_id)


def main():
    print("Worker started...")
    while True:
        process_next_job()
        time.sleep(5)


if __name__ == "__main__":
    main()
