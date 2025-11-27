import time
import json
import base64
import io
import os

from supabase import create_client, Client
from openai import OpenAI
from PyPDF2 import PdfReader


# ----------------------------------------
# ENVIRONMENT
# ----------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------------------
# AI CALL — HIGH ACCURACY VERSION
# ----------------------------------------
def call_ami_ai(extracted_text: str, pdf_base64: str):
    """
    Calls GPT-4o with a very strict prompt ensuring:
    - No hallucinations
    - Proper CBC & chemistry parsing
    - JSON-only output
    """

    # Clean and trim extracted text
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 20000:
            extracted_text = extracted_text[:20000]

    # Only include PDF data if text is missing or incomplete
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 80:
        pdf_chunk = pdf_base64[:60000]

    system_prompt = """
You are AMI — an advanced medical laboratory interpretation AI.
You analyse blood test reports ONLY from provided text/PDF content.

STRICT RULES:
- NEVER invent values or diagnoses.
- ONLY extract numbers and markers that appear in the text.
- If a value is missing, do NOT guess it.
- If infection markers (WBC, neutrophils, CRP, ESR) are missing, clearly state this.
- Output STRICT JSON ONLY. No explanations outside the JSON.

OUTPUT FORMAT:
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

    user_prompt = f"""
Extracted Lab Text:
{extracted_text}

PDF Fallback (optional):
{pdf_chunk}

Return ONLY valid JSON.
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

        ai_text = response.output_text.strip()

        # Strip backticks if present
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        # Ensure JSON boundaries
        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"):ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI ERROR:", e)
        return {"error": str(e)}


# ----------------------------------------
# PDF TEXT EXTRACTION
# ----------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text = ""
        for page in reader.pages:
            content = page.extract_text() or ""
            text += content + "\n"

        print("📄 Extracted text length:", len(text))
        return text.strip()

    except Exception as e:
        print("❌ PDF EXTRACT ERROR:", e)
        return ""


# ----------------------------------------
# PROCESS REPORT
# ----------------------------------------
def process_next_report():
    print("🔍 Looking for QUEUED reports...")

    # Pull next queued report
    response = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "queued") \
        .limit(1) \
        .execute()

    if not response.data:
        return None  # Nothing to process

    report = response.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Found report to process: {report_id}")

    # Mark as processing
    supabase.table("reports") \
        .update({"ai_status": "processing"}) \
        .eq("id", report_id) \
        .execute()

    # ------------------------------
    # DOWNLOAD PDF FROM STORAGE
    # ------------------------------
    try:
        print("⬇️ Downloading PDF from Supabase Storage:", file_path)
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ STORAGE DOWNLOAD ERROR:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": "File download failed"}
        }).eq("id", report_id).execute()
        return None

    # ------------------------------
    # EXTRACT TEXT
    # ------------------------------
    extracted_text = extract_text_from_pdf(pdf_bytes)
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    # ------------------------------
    # CALL AI
    # ------------------------------
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    if "error" in ai_json:
        print("❌ AI FAILED:", ai_json["error"])
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json
        }).eq("id", report_id).execute()
        return None

    # ------------------------------
    # SAVE RESULTS
    # ------------------------------
    print("💾 Saving AI results into Supabase…")

    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json,
        "extracted_text": extracted_text,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", [])
    }).eq("id", report_id).execute()

    print(f"✅ Report completed: {report_id}")


# ----------------------------------------
# MAIN LOOP
# ----------------------------------------
print("🚀 Worker started successfully…")

while True:
    try:
        process_next_report()
    except Exception as e:
        print("⚠️ Worker crashed but recovered:", e)

    time.sleep(5)
