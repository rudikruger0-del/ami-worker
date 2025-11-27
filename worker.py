import os
import time
import json
import base64
from supabase import create_client
from openai import OpenAI

# -----------------------------
# ENV VARS
# -----------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
OPENAI_KEY = os.environ["OPENAI_API_KEY"]

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_KEY)

# -----------------------------
# AI FUNCTION (yours)
# -----------------------------
def call_ami_ai(extracted_text, pdf_base64):

    # 1️⃣ Clean + safely limit extracted text
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 20000:
            extracted_text = extracted_text[:20000]  # avoid context overflow

    # 2️⃣ Only send PDF fallback when text is missing or too short
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 100:
        pdf_chunk = pdf_base64[:60000]  # PDF fallback capped at 60k chars

    system_message = """
You are AMI — an advanced laboratory interpretation AI.
You analyse blood tests (CBC, chemistry, markers of infection/inflammation).

CRITICAL RULES:
1. Base your interpretation ONLY on values found in the provided text/PDF.
2. NEVER invent values, NEVER guess diagnoses.
3. If the labs are incomplete, clearly state that.
4. Output STRICT JSON ONLY.

Expected JSON structure:
{
  "summary": [],
  "trend_summary": [],
  "flagged_results": [],
  "interpretation": [],
  "risk_level": "",
  "recommendations": [],
  "cbc_values": {},
  "chemistry_values": {},
  "disclaimer": "This is not medical advice."
}
"""

    user_message = f"""
Extracted Lab Text (clean):
{extracted_text}

PDF Fallback (limited):
{pdf_chunk}

Extract ALL CBC and chemistry values you can find.
ONLY return the JSON object described in the system prompt.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",
            max_output_tokens=1800,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        ai_text = response.output_text.strip()
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"): ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# -----------------------------
# LOAD PDF FROM SUPABASE
# -----------------------------
def load_pdf_base64(path):
    try:
        file_bytes = supabase.storage.from_("reports").download(path)
        return base64.b64encode(file_bytes).decode()
    except Exception as e:
        print("❌ PDF download error:", e)
        return ""


# -----------------------------
# PROCESS A SINGLE REPORT
# -----------------------------
def process_report(row):
    report_id = row["id"]
    print(f"🔍 Processing report {report_id}")

    pdf_path = row.get("file_path")
    extracted_text = row.get("extracted_text") or ""

    pdf_base64 = ""
    if pdf_path:
        pdf_base64 = load_pdf_base64(pdf_path)

    ai_json = call_ami_ai(extracted_text, pdf_base64)

    supabase.table("reports").update({
        "ai_status": "completed" if "error" not in ai_json else "failed",
        "ai_results": ai_json
    }).eq("id", report_id).execute()

    print(f"✔ Done: {report_id}")


# -----------------------------
# MAIN WORKER LOOP
# -----------------------------
def worker_loop():
    print("🚀 Worker running…")

    while True:
        try:
            res = supabase.table("reports").select("*") \
                .eq("ai_status", "queued") \
                .order("created_at") \
                .limit(1).execute()

            rows = res.data
            if not rows:
                print("⏳ No queued reports…")
                time.sleep(4)
                continue

            report = rows[0]

            # mark as processing
            supabase.table("reports").update({
                "ai_status": "processing"
            }).eq("id", report["id"]).execute()

            # run AI
            process_report(report)

        except Exception as e:
            print("💥 worker loop error:", e)
            time.sleep(3)


if __name__ == "__main__":
    worker_loop()
