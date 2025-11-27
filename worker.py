import time
import json
import base64
import os
import fitz  # PyMuPDF
from supabase import create_client, Client
from openai import OpenAI

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("🔧 AMI Worker Ready (GPT-4o-mini Vision)")


# ---------------------------------------------------------
# TOKEN-SAFE AI CALL
# ---------------------------------------------------------
def call_ami_ai(pdf_b64, extracted_text):
    """
    Token-optimized Vision→JSON pipeline.
    """
    try:
        # ALWAYS send only the FIRST 10–80 lines of text to reduce tokens
        safe_text = "\n".join(extracted_text.splitlines()[:80]) if extracted_text else ""

        prompt = f"""
You are AMI — Artificial Medical Intelligence.
Interpret a laboratory report with clinical accuracy.

Rules:
- NEVER invent numbers.
- Only use values visible in the image or in the extracted text.
- If a section has no data, return an empty list/object.
- Output STRICT JSON.

JSON FORMAT:
{{
  "patient_summary": "",
  "overall_risk": "",
  "cbc_values": {{}},
  "chemistry_values": {{}},
  "flagged_abnormalities": [],
  "infection_indicators": [],
  "dehydration_markers": [],
  "kidney_markers": [],
  "liver_markers": [],
  "iron_deficiency_markers": [],
  "vitamin_deficiency_markers": [],
  "trend_analysis": [],
  "detailed_interpretation": [],
  "doctor_recommendations": [],
  "urgent_findings": [],
  "disclaimer": "This is an assistive AI analysis, not a medical diagnosis."
}}

Extracted text (may be incomplete):
{safe_text}
"""

        res = openai_client.responses.create(
            model="gpt-4o-mini",
            response_format={"type": "json"},
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:application/pdf;base64,{pdf_b64}"
                        }
                    ]
                }
            ]
        )

        # AI JSON output
        final = res.output[0].content[0].json
        return final

    except Exception as e:
        return {"error": f"AI Failure: {str(e)}"}


# ---------------------------------------------------------
# SAFE PDF LOADER → BASE64 + TEXT
# ---------------------------------------------------------
def load_pdf_for_ai(pdf_bytes):
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

    text = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
    except:
        text = ""

    return pdf_b64, text.strip()


# ---------------------------------------------------------
# POLLING WORKER LOOP
# ---------------------------------------------------------
def process_next_report():
    # Get the oldest queued report
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not job.data:
        return None

    report = job.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Processing report: {report_id}")

    # Mark processing
    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

    # Download PDF from storage
    pdf_file = supabase.storage.from_("reports").download(file_path)

    if not pdf_file:
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_results": {"error": "File missing in storage"}}
        ).eq("id", report_id).execute()
        return

    pdf_bytes = pdf_file
    pdf_b64, extracted_text = load_pdf_for_ai(pdf_bytes)

    # Run AI
    result = call_ami_ai(pdf_b64, extracted_text)

    # Store results
    supabase.table("reports").update(
        {
            "ai_results": result,
            "ai_status": "completed",
            "extracted_text": extracted_text[:20000]
        }
    ).eq("id", report_id).execute()

    print(f"✅ Completed: {report_id}")
    return True


# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
while True:
    try:
        done = process_next_report()
        if not done:
            print("⏳ No reports. Waiting...")
        time.sleep(5)

    except Exception as e:
        print("WORKER ERROR:", str(e))
        time.sleep(5)
