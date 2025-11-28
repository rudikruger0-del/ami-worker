import os
import time
import json
import traceback
from supabase import create_client, Client
from openai import OpenAI

# --- ENV ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "reports"


def process_report(job):
    try:
        report_id = job["id"]
        l_text = job.get("l_text", "")
        file_path = job.get("file_path")

        if not file_path:
            return {"error": f"Missing file_path for report {report_id}"}

        # ------------------------------
        # 1. Generate signed PDF URL
        # ------------------------------
        signed = supabase.storage.from_(BUCKET).create_signed_url(
            file_path,
            expires_in=3600
        )

        signed_url = signed["signedURL"]
        print("SIGNED URL:", signed_url)

        # ------------------------------
        # 2. Call OpenAI JSON Mode
        # ------------------------------
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Extract blood test values and return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": f"Patient blood report text:\n\n{l_text}"
                }
            ],
            response_format={"type": "json_object"}
        )

        # Extract stringified JSON
        raw_json = response.choices[0].message.content
        ai_json = json.loads(raw_json)

        # ------------------------------
        # 3. Update DB
        # ------------------------------
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_json,
            "ai_error": None
        }).eq("id", report_id).execute()

        return {"success": True}

    except Exception as e:
        error_msg = str(e)
        traceback.print_exc()

        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": error_msg
        }).eq("id", job["id"]).execute()

        return {"error": error_msg}


def main():
    print("AMI Worker started… watching for jobs.")

    while True:
        try:
            res = supabase.table("reports") \
                .select("*") \
                .eq("ai_status", "queued") \
                .limit(1) \
                .execute()

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
