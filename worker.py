import time
import json
import base64
import io

from supabase import create_client, Client
from openai import OpenAI
from PyPDF2 import PdfReader

# ----------------------------
# Environment
# ----------------------------
import os

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# AI CALL
# ----------------------------
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
You analyse blood tests (CBC, chemistry, markers of infection/inflammation).

CRITICAL RULES:
1. Base your interpretation ONLY on values found in the provided text/PDF.
2. NEVER invent values.
3. If markers are missing, state clearly that they cannot be evaluated.
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
Extracted Lab Text:
{extracted_text}

PDF Fallback (limited):
{pdf_chunk}

Return ONLY the JSON object described.
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


# ----------------------------
# PDF → Extract Text
# ----------------------------
def extract_text_from_pdf(pdf_bytes):
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        return text.strip()

    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""


# ----------------------------
# PROCESS NEXT REPORT
# ----------------------------
def process_next_report():
    print("🔍 Checking for queued reports...")

    # Pull next report in queue
    response = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "processing") \
        .limit(1) \
        .execute()

    if not response.data:
        return None

    report = response.data[0]
    print("📄 Processing report:", report["id"])

    file_path = report["file_path"]

    # ----------------------------
    # DOWNLOAD PDF FROM STORAGE
    # ----------------------------
    try:
        print("⬇️ Downloading PDF from Supabase...")
        file_res = supabase.storage.from_("reports").download(file_path)
        pdf_bytes = file_res
    except Exception as e:
        print("❌ Could not download PDF:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report["id"]).execute()
        return None

    # ----------------------------
    # Extract text
    # ----------------------------
    extracted_text = extract_text_from_pdf(pdf_bytes)

    # PDF base64 for fallback
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    # ----------------------------
    # Call AI
    # ----------------------------
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    if "error" in ai_json:
        print("❌ AI failed:", ai_json["error"])
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json
        }).eq("id", report["id"]).execute()
        return None

    # ----------------------------
    # Save back to Supabase
    # ----------------------------
    print("💾 Saving results to database...")

    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json,
        "extracted_text": extracted_text,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", [])
    }).eq("id", report["id"]).execute()

    print("✅ Report completed:", report["id"])


# ----------------------------
# MAIN LOOP
# ----------------------------
print("🚀 Worker started...")

while True:
    try:
        process_next_report()
    except Exception as e:
        print("❌ Worker crash prevented:", e)

    time.sleep(5)
