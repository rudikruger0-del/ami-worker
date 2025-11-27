import os
import time
import json
import base64
from supabase import create_client, Client
from openai import OpenAI

# -----------------------------------
# ENVIRONMENT
# -----------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# Poll every X seconds
POLL_SECONDS = 5


# ---------------------------------------------------------
# GPT-4o Vision – FULL CLINICAL LAB INTERPRETATION ENGINE
# ---------------------------------------------------------
def run_ai_analysis(pdf_bytes, extracted_text=""):

    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    response = client.responses.create(
        model="gpt-4o-mini",                     # can be upgraded later
        response_format={"type": "json"},
        input=[
            {
                "role": "system",
                "content": """
You are AMI — an advanced clinical laboratory interpretation AI.

You receive a laboratory report (CBC + chemistry + hormones + others). 
You will extract values AND generate a complete medical-style interpretation.

-----------------------------------------------------------
RETURN ONLY THIS EXACT JSON STRUCTURE:
{
  "risk_level": "",
  "narrative_text": "",
  "summary": [],
  "trend_summary": [],
  "flagged_results": [],
  "recommendations": [],
  "urgent_care": [],
  "cbc_values": {},
  "chemistry_values": {},
  "disclaimer": "This AI report is for informational purposes only and is not a diagnosis or treatment plan. Always consult a licensed medical professional."
}
-----------------------------------------------------------

DETAILED RULES:

risk_level:
- "low", "moderate", "high", or "indeterminate".

narrative_text:
- 2–5 short paragraphs.
- Clinical but understandable.
- Explain normal + abnormal findings.
- Never diagnose. Never prescribe medication.

summary:
- 3–6 bullet points summarizing the major findings.

trend_summary:
- If no previous values exist:
  ["No trend comparison is possible based on this report alone."]

flagged_results:
- ONLY abnormalities.
- Each item:
  {
    "test": "",
    "value": "",
    "units": "",
    "flag": "",
    "reference_range": "",
    "comment": ""
  }

recommendations:
- 2–5 general health/safety recommendations.
- No drugs, no dosages.

urgent_care:
- RED-FLAG symptoms that should prompt urgent / emergency evaluation.

cbc_values / chemistry_values:
- Extract all values you can read from the PDF or text.
- EACH item stored as:
  "WBC": { "value": 7.2, "units": "10^9/L", "flag": "normal" }

SAFETY:
- Never hallucinate.
- Only report values you truly see.
- Use cautious language (“may indicate”, “can be associated with”).
- Return VALID JSON ONLY — no markdown, no commentary outside JSON.
"""
            },
            {
                "role": "user",
                "content": f"""
Extracted Text (may be incomplete):
{extracted_text}

Base64 PDF:
{pdf_base64}

Respond ONLY with one valid JSON object.
"""
            }
        ]
    )

    # Parse AI JSON safely
    try:
        ai_json = (
            response.output[0].content[0].get("json")
            or response.output[0].content[0].get("text")
            or {}
        )
    except:
        ai_json = {"error": "AI output unreadable"}

    return ai_json


# ---------------------------------------------------------
# PROCESS ONE REPORT
# ---------------------------------------------------------
def process_report(report):
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Processing report: {report_id}")

    # -----------------------------------
    # DOWNLOAD PDF FROM SUPABASE STORAGE
    # -----------------------------------
    try:
        file_res = supabase.storage.from_("reports").download(file_path)
        if not file_res:
            raise Exception("Empty download")
        pdf_bytes = file_res
    except Exception as e:
        print("❌ Failed to download PDF:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": "PDF missing from storage"}
        }).eq("id", report_id).execute()
        return

    # -----------------------------------
    # RUN AI ANALYSIS
    # -----------------------------------
    ai_json = run_ai_analysis(pdf_bytes)

    # -----------------------------------
    # SAVE TO SUPABASE
    # -----------------------------------
    supabase.table("reports").update({
        "ai_status": "completed" if "error" not in ai_json else "failed",
        "ai_results": ai_json
    }).eq("id", report_id).execute()

    print(f"✅ Completed: {report_id}")


# ---------------------------------------------------------
# MAIN WORKER LOOP
# ---------------------------------------------------------
def worker_loop():
    print("🚀 AMI Worker Started — GPT-4o Vision Enabled")

    while True:
        try:
            # fetch next queued report
            res = (
                supabase.table("reports")
                .select("*")
                .eq("ai_status", "queued")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )

            if not res.data:
                print("⏳ No queued reports…")
                time.sleep(POLL_SECONDS)
                continue

            report = res.data[0]

            # mark as processing
            supabase.table("reports").update({
                "ai_status": "processing"
            }).eq("id", report["id"]).execute()

            # process it
            process_report(report)

        except Exception as e:
            print("🔥 Worker Crash Prevented:", e)

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    worker_loop()
