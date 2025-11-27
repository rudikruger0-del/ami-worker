import time
import os
import json
import base64
from supabase import create_client, Client
from openai import OpenAI

# =========================
# ENV VARIABLES
# =========================
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# =========================
# WORKER SETTINGS
# =========================
POLL_SECONDS = 4
MAX_AI_TOKENS = 2000


# =========================
# PDF → Base64
# =========================
def load_pdf_base64(file_path: str):
    try:
        storage_path = f"reports/{file_path}"

        res = supabase.storage.from_("reports").download(file_path)
        if not res:
            print("❌ Could not download PDF:", file_path)
            return ""

        return base64.b64encode(res).decode()
    except Exception as e:
        print("❌ PDF load error:", e)
        return ""


# =========================
# AI CALL
# =========================
def call_ami_ai(extracted_text, pdf_base64):
    system_message = """
You are AMI — an advanced laboratory interpretation AI.

Output STRICT JSON ONLY.

JSON structure:
{
  "summary": [],
  "trend_summary": [],
  "flagged_results": [],
  "interpretation": [],
  "risk_level": "",
  "recommendations": [],
  "cbc_values": {},
  "disclaimer": "This is not medical advice."
}
"""

    user_message = f"""
Extracted Text:
{extracted_text}

PDF (base64 for fallback extraction):
{pdf_base64}

Return JSON only.
"""

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        max_output_tokens=MAX_AI_TOKENS,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    )

    # Response API returns text inside:
    # response.output_text
    ai_text = response.output_text

    # Try parse JSON safely
    try:
        clean = json.loads(ai_text)
        return clean
    except Exception as e:
        print("❌ Failed to parse AI JSON:", e)
        return {"error": "AI JSON parse failed", "raw": ai_text}


# =========================
# PROCESS ONE REPORT
# =========================
def process_report(report):
    print("🔍 Processing report:", report["id"])

    file_path = report["file_path"]
    extracted_text = report.get("extracted_text", "")

    # Load PDF from Supabase → base64
    pdf_base64 = load_pdf_base64(file_path)

    # Call AI
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    # Save JSONB into Supabase
    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json
    }).eq("id", report["id"]).execute()

    print("✔ AI completed for:", report["id"])


# =========================
# MAIN LOOP
# =========================
def worker_loop():
    print("🚀 AMI Worker Started")

    while True:
        try:
            queued = supabase.table("reports") \
                .select("*") \
                .eq("ai_status", "queued") \
                .limit(1) \
                .execute()

            if not queued.data:
                print("⏳ No queued reports…")
                time.sleep(POLL_SECONDS)
                continue

            report = queued.data[0]

            # Mark as processing
            supabase.table("reports").update({
                "ai_status": "processing"
            }).eq("id", report["id"]).execute()

            process_report(report)

        except Exception as e:
            print("❌ Worker error:", e)
            time.sleep(3)


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    worker_loop()
