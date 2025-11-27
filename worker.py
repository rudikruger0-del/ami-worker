import time
import json
import os
import base64
import io

from supabase import create_client, Client
from openai import OpenAI

# ----------------------------
# ENVIRONMENT
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing required environment variables")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# GPT-4o VISION PDF PROCESSING
# ----------------------------
def vision_extract_text(pdf_bytes: bytes) -> str:
    """
    Sends the ENTIRE PDF to GPT-4o Vision as a file input.
    GPT-4o Vision reads image PDFs perfectly (text or scanned).
    """
    try:
        # GPT-4o Vision requires a base64 file-like input
        b64_pdf = base64.b64encode(pdf_bytes).decode()

        response = openai_client.responses.create(
            model="gpt-4o",   # Vision enabled
            max_output_tokens=1200,
            input=[
                {
                    "role": "system",
                    "content": "You are a text extraction assistant. Extract ALL text from this PDF, preserving lab values clearly."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "input_file": {
                                "data": b64_pdf,
                                "mime_type": "application/pdf"
                            }
                        },
                        {
                            "type": "text",
                            "text": "Extract ALL readable text from this PDF clearly. Do not summarize. Only return raw extracted text."
                        }
                    ],
                }
            ]
        )

        extracted = response.output_text.strip()
        return extracted[:20000]  # safety cap

    except Exception as e:
        print("❌ GPT-4o Vision extraction error:", e)
        return ""


# ----------------------------
# GPT-4o MEDICAL INTERPRETATION
# ----------------------------
def run_medical_interpretation(extracted_text: str) -> dict:
    """
    Full detailed medical interpretation using GPT-4o.
    Returns structured JSON for frontend printing.
    """

    system_prompt = """
You are AMI — an advanced medical laboratory interpretation AI.

Your task: produce a complete, clinically structured interpretation for doctors.

RULES:
- Use ONLY the values in the extracted text.
- Never invent anything.
- If a value is missing, mark it “Not provided”.
- Output STRICT JSON ONLY.

OUTPUT STRUCTURE:
{
  "summary": [],
  "main_findings": [],
  "interpretation": [],
  "flagged_results": [],
  "cbc_values": {},
  "chemistry_values": {},
  "recommendations": [],
  "urgent_warnings": [],
  "risk_level": "",
  "report_text": "",
  "disclaimer": "This is not medical advice."
}
"""

    user_prompt = f"""
Extracted laboratory text:

\"\"\"{extracted_text}\"\"\"

Now produce the full JSON object ONLY.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",
            max_output_tokens=1800,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        ai = response.output_text.strip()
        ai = ai.replace("```json", "").replace("```", "")

        # extract pure JSON
        ai = ai[ai.index("{"): ai.rindex("}") + 1]

        return json.loads(ai)

    except Exception as e:
        print("❌ Interpretation error:", e)
        return {"error": str(e)}


# ----------------------------
# PROCESS SINGLE REPORT
# ----------------------------
def process_report(report):
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Processing report {report_id}")

    # DOWNLOAD PDF FROM SUPABASE STORAGE
    try:
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ PDF download failure:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report_id).execute()
        return

    # TEXT EXTRACTION USING GPT-4o VISION
    extracted = vision_extract_text(pdf_bytes)

    if len(extracted.strip()) < 20:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": "Unable to extract text from PDF."},
        }).eq("id", report_id).execute()
        return

    # MEDICAL INTERPRETATION
    ai_json = run_medical_interpretation(extracted)

    if "error" in ai_json:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json,
        }).eq("id", report_id).execute()
        return

    # SAVE BACK TO SUPABASE
    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": extracted,
        "ai_results": ai_json,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("main_findings", [])
    }).eq("id", report_id).execute()

    print(f"✅ Completed report {report_id}")


# ----------------------------
# MAIN LOOP
# ----------------------------
print("🚀 AMI Worker V3 (Vision Mode) started…")

while True:
    try:
        res = supabase.table("reports") \
            .select("*") \
            .eq("ai_status", "queued") \
            .limit(1).execute()

        if res.data:
            rep = res.data[0]
            # Lock the report
            supabase.table("reports").update({"ai_status": "processing"}).eq("id", rep["id"]).execute()
            process_report(rep)
        else:
            print("⏳ No queued reports...")
    except Exception as e:
        print("💥 Worker error:", e)

    time.sleep(4)
