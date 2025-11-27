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

CHECK_INTERVAL = 5  # seconds


# --------------------------
# GPT-4o Vision: Extract Data
# --------------------------
def process_pdf_with_gpt_vision(pdf_base64: str):
    """
    Sends the PDF directly to GPT-4o Vision.
    Returns structured JSON.
    """

    prompt = """
You are AMI — an advanced medical AI assistant.

Extract ALL clinically relevant information from this lab report.
This PDF may include chemistry, CBC, hormones, urine, or special tests.

RETURN JSON ONLY in this EXACT structure:

{
  "patient_info": {
    "name": "",
    "age": "",
    "sex": "",
    "lab_date": ""
  },
  "cbc": {
    "wbc": "",
    "rbc": "",
    "hemoglobin": "",
    "hematocrit": "",
    "mcv": "",
    "mch": "",
    "mchc": "",
    "platelets": "",
    "neutrophils": "",
    "lymphocytes": "",
    "monocytes": "",
    "eosinophils": "",
    "basophils": ""
  },
  "chemistry": {},
  "hormones": {},
  "urine": {},
  "other_tests": {},
  
  "interpretation": {
    "risk_level": "",
    "summary": "",
    "flagged_results": [],
    "trend_summary": "",
    "recommendations": "",
    "urgent_findings": ""
  },

  "disclaimer": "This AI interpretation is assistive and not a medical diagnosis."
}

Be very detailed in the interpretation. 
If something is normal, say WHY.
If something is abnormal, explain WHAT it means.
Avoid medical jargon that a layperson won't understand.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Here is the lab PDF."
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:application/pdf;base64,{pdf_base64}"
                        }
                    ]
                }
            ]
        )

        ai_json = response.output[0].content[0].text
        return json.loads(ai_json)

    except Exception as e:
        return {"error": str(e)}


# --------------------------
# WORKER LOOP
# --------------------------
def worker_loop():
    print("🚀 AMI Worker Started (GPT-4o Vision PDF mode)…")

    while True:
        try:
            # pull the next queued report
            result = (
                supabase.table("reports")
                .select("*")
                .eq("ai_status", "queued")
                .order("created_at", desc=False)
                .limit(1)
                .execute()
            )

            if not result.data:
                print("⏳ No queued reports…")
                time.sleep(CHECK_INTERVAL)
                continue

            report = result.data[0]
            report_id = report["id"]
            filename = report_id + ".pdf"

            print(f"\n📄 Processing report: {report_id}")

            # update status to processing
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

            # download PDF from Supabase Storage
            file_res = supabase.storage.from_("reports").download(filename)

            if file_res is None:
                raise Exception("PDF missing in storage")

            pdf_bytes = file_res
            pdf_base64 = base64.b64encode(pdf_bytes).decode()

            # send to GPT-4o Vision
            ai_data = process_pdf_with_gpt_vision(pdf_base64)

            # determine status
            new_status = "completed" if "error" not in ai_data else "failed"

            # save results
            supabase.table("reports").update({
                "ai_status": new_status,
                "ai_results": ai_data
            }).eq("id", report_id).execute()

            print(f"✅ Completed: {report_id}\n")

        except Exception as error:
            print("❌ Worker Error:", error)

        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    worker_loop()
