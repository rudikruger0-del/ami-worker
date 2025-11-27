import time
import json
import base64
import os
from supabase import create_client, Client
from openai import OpenAI

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

print("🔧 AMI Worker Ready (GPT-4o-mini Vision, no PyMuPDF)")


# ---------------------------------------------------------
# AI CALL (Vision → JSON)
# ---------------------------------------------------------
def call_ami_ai(pdf_bytes):
    try:
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        prompt = """
You are AMI — Artificial Medical Intelligence.
Interpret a laboratory report with clinical accuracy.

Rules:
- ONLY use values visible in the PDF image.
- No hallucinations.
- If a section has no data, return empty values.
- Output valid JSON only.

JSON FORMAT:
{
  "patient_summary": "",
  "overall_risk": "",
  "cbc_values": {},
  "chemistry_values": {},
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
  "disclaimer": "This is an assistive AI interpretation, not a medical diagnosis."
}
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

        return res.output[0].content[0].json

    except Exception as e:
        return {"error": f"AI Error: {str(e)}"}


# ---------------------------------------------------------
# PROCESS REPORT
# ---------------------------------------------------------
def process_next_report():
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not job.data:
        return False

    report = job.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Processing: {report_id}")

    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

    # Download PDF
    pdf_bytes = supabase.storage.from_("reports").download(file_path)

    if not pdf_bytes:
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_results": {"error": "File missing"}}
        ).eq("id", report_id).execute()
        return True

    # AI analysis
    ai_result = call_ami_ai(pdf_bytes)

    # Save to DB
    supabase.table("reports").update(
        {
            "ai_results": ai_result,
            "ai_status": "completed"
        }
    ).eq("id", report_id).execute()

    print(f"✅ Completed: {report_id}")
    return True


# ---------------------------------------------------------
# LOOP
# ---------------------------------------------------------
while True:
    try:
        if not process_next_report():
            print("⏳ No tasks. Waiting...")
        time.sleep(5)

    except Exception as e:
        print("Worker crash:", e)
        time.sleep(5)
        
