import os
import time
import json
import traceback
from supabase import create_client, Client
from openai import OpenAI

# ---------------------------------------------------------
# ENVIRONMENT VARIABLES
# ---------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")   # MUST be the Service Key

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("❌ SUPABASE_URL or SUPABASE_SERVICE_KEY is missing!")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

client = OpenAI()   # NEW SDK – this is correct

BUCKET = "reports"

# ---------------------------------------------------------
# AI JSON SCHEMA FOR RESPONSE
# ---------------------------------------------------------
AI_SCHEMA = {
    "name": "blood_test_extraction",
    "schema": {
        "type": "object",
        "properties": {
            "cbc": {
                "type": "object",
                "properties": {
                    "WBC": {"type": "string"},
                    "RBC": {"type": "string"},
                    "Hemoglobin": {"type": "string"},
                    "Hematocrit": {"type": "string"},
                    "Platelets": {"type": "string"},
                }
            },
            "summary": {"type": "string"}
        },
        "required": ["summary"]
    }
}

# ---------------------------------------------------------
# JOB PROCESSING FUNCTION
# ---------------------------------------------------------
def process_report(job):
    try:
        report_id = job["id"]
        l_text = job.get("l_text", "")
        file_path = job.get("file_path")

        if not file_path:
            raise Exception(f"Missing file_path for report {report_id}")

        # ---------------------------------------------------------
        # Generate a secure POPIA-safe signed URL (not public)
        # ---------------------------------------------------------
        signed = supabase.storage.from_(BUCKET).create_signed_url(
            file_path,
            expires_in=3600  # 1 hour
        )

        if "signedURL" not in signed:
            raise Exception(f"Supabase signed URL generation failed: {signed}")

        signed_url = signed["signedURL"]
        print("SIGNED URL:", signed_url)

        # ---------------------------------------------------------
        # Call OpenAI using NEW 2024 SDK (responses API)
        # ---------------------------------------------------------
        ai_response = client.responses.create(
            model="gpt-4o-mini",
            input=f"You are an expert medical AI. Extract CBC results from this file:\n\n{l_text}",
            response_format={
                "type": "json_schema",
                "json_schema": AI_SCHEMA
            }
        )

        # Parse JSON result safely
        ai_json = ai_response.output[0].content[0].json

        # ---------------------------------------------------------
        # Update DB → Completed
        # ---------------------------------------------------------
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_json,
            "ai_error": None
        }).eq("id", report_id).execute()

        print("AI processing completed:", report_id)
        return {"success": True}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

        # Update DB → Failed
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": error_msg
        }).eq("id", job["id"]).execute()

        return {"error": error_msg}

# ---------------------------------------------------------
# WORKER LOOP
# ---------------------------------------------------------
def main():
    print("AMI Worker started… watching for jobs.")

    while True:
        try:
            res = supabase.table("reports").select("*")\
                .eq("ai_status", "pending")\
                .limit(1).execute()

            jobs = res.data

            if jobs:
                job = jobs[0]
                print("Processing job:", job["id"])
                process_report(job)
            else:
                print("No jobs...")

            time.sleep(5)

        except Exception as e:
            print("Worker error:", e)
            time.sleep(5)


if __name__ == "__main__":
    main()
