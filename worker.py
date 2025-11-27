import time
import base64
import json
import os
from supabase import create_client
from openai import OpenAI

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def call_ami_ai(extracted_text, pdf_base64):
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 20000:
            extracted_text = extracted_text[:20000]

    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 100:
        pdf_chunk = pdf_base64[:60000]

    system_message = """
You are AMI — an advanced laboratory interpretation AI.
Output STRICT JSON ONLY.
"""

    user_message = f"""
Extracted Text:
{extracted_text}

PDF (limited):
{pdf_chunk}
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
        print("AI Error:", e)
        return {"error": str(e)}


def process_next_report():
    print("🔍 Checking for queued reports...")

    result = supabase.table("reports").select("*").eq("ai_status", "queued").limit(1).execute()

    if not result.data:
        print("⏳ No queued reports...")
        return

    report = result.data[0]
    report_id = report["id"]

    print(f"⚡ Processing report: {report_id}")

    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

    # Read PDF
    with open("/app/reports/" + report["file_path"], "rb") as f:
        pdf_bytes = f.read()
        pdf_base64 = base64.b64encode(pdf_bytes).decode()

    extracted_text = report.get("extracted_text", "")

    ai_results = call_ami_ai(extracted_text, pdf_base64)

    # Save results
    supabase.table("reports").update({
        "ai_results": ai_results,
        "ai_status": "completed"
    }).eq("id", report_id).execute()

    print(f"✅ Done: {report_id}")


if __name__ == "__main__":
    print("🚀 Worker started...")

    while True:
        process_next_report()
        time.sleep(5)
