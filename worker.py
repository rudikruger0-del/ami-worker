import os
import time
import json
import traceback
from supabase import create_client, Client
from openai import OpenAI

# ------------------------------
# ENVIRONMENT VARIABLES
# ------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Validate envs early
if not SUPABASE_URL:
    raise Exception("SUPABASE_URL missing")
if not SUPABASE_KEY:
    raise Exception("SUPABASE_SERVICE_KEY missing")
if not OPENAI_API_KEY:
    raise Exception("OPENAI_API_KEY missing")

# ------------------------------
# CLIENTS
# ------------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "reports"


# ------------------------------
# PROCESS ONE REPORT
# ------------------------------

def process_report(job):
    try:
        report_id = job["id"]
        file_path = job.get("file_path")
        l_text = job.get("l_text", "")

        if not file_path:
            return {"error": f"Missing file_path for report {report_id}"}

        # ----- Generate signed URL -----
        signed = supabase.storage.from_(BUCKET).create_signed_url(
            file_path,
            expires_in=3600
        )

        signed_url = signed.get("signedURL")
        print("SIGNED URL:", signed_url)

        # ----- Call OpenAI -----
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract and summarize blood test data. Respond in JSON."},
                {"role": "user", "content": l_text}
            ],
            response_format={"type": "json_object"}
        )

        # The new SDK returns JSON inside message["content"]
        raw_json = response.choices[0].message["content"]

        # Convert JSON string → Python dict
        ai_json = json.loads(raw_json)

        # ----- Update DB -----
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_json,
            "ai_error": None
        }).eq("id", report_id).execute()

        return {"success": True}

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": error_msg
        }).eq("id", job["id"]).execute()

        return {"error": error_msg}


# ------------------------------
# WORKER LOOP
# ------------------------------

def main():
    print("AMI Worker started…")

    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "queued").limit(1).execute()
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
