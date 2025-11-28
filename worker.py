import os
import time
import base64
import requests
from supabase import create_client, Client
from openai import OpenAI

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"  # storage bucket name

def get_pending_reports():
    res = supabase.table("reports").select("*").eq("ai_status", "queued").execute()
    return res.data or []

def download_pdf_signed(path):
    """
    Generates a signed URL (NOT public), downloads the file safely, POPIA compliant.
    """
    signed = supabase.storage.from_(BUCKET).create_signed_url(path, 3600)
    if "signedURL" not in signed:
        raise Exception("Could not generate signed URL")

    url = signed["signedURL"]

    resp = requests.get(url)
    if resp.status_code != 200:
        raise Exception(f"Download failed {resp.status_code}")

    return resp.content

def extract_text_using_ai(pdf_bytes):
    """
    Uses new 2025 OpenAI Responses API.
    """
    b64_pdf = base64.b64encode(pdf_bytes).decode()

    response = client.responses.create(
        model="gpt-4.1",
        input="Extract ALL CBC values and summarize any abnormalities.",
        attachments=[
            {
                "type": "application/pdf",
                "data": b64_pdf,
            }
        ],
        response_format={ 
            "type": "json_schema",
            "json_schema": {
                "name": "cbc_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "cbc": { "type": "object" },
                        "interpretation": { "type": "string" }
                    },
                    "required": ["cbc", "interpretation"]
                }
            }
        }
    )

    return response.output_json

def process_report(row):
    report_id = row["id"]
    path = row["file_path"]

    if not path:
        raise Exception(f"Missing file_path for report {report_id}")

    # 1. Download PDF via signed URL
    pdf_bytes = download_pdf_signed(path)

    # 2. Extract CBC + Interpretation
    result_json = extract_text_using_ai(pdf_bytes)

    # 3. Save back to database
    supabase.table("reports").update({
        "ai_status": "done",
        "ai_results": result_json
    }).eq("id", report_id).execute()


def main():
    print("AMI Worker started… watching for jobs.")

    while True:
        try:
            jobs = get_pending_reports()

            if not jobs:
                print("No jobs...")
            else:
                print(f"Processing {len(jobs)} job(s)…")

            for job in jobs:
                try:
                    process_report(job)
                except Exception as e:
                    print("Error processing:", e)
                    supabase.table("reports").update({
                        "ai_status": "failed",
                        "ai_error": str(e)
                    }).eq("id", job["id"]).execute()

        except Exception as e:
            print("Worker crashed:", e)

        time.sleep(6)

if __name__ == "__main__":
    main()
