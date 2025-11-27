import time
import json
import base64
import io
import os

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader   # ✅ CORRECT LIBRARY

# ----------------------------
# ENVIRONMENT VARIABLES
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# AI CALL (HIGH ACCURACY)
# ----------------------------
def call_ami_ai(extracted_text, pdf_base64):

    # Clean text
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 20000:
            extracted_text = extracted_text[:20000]

    # Fallback PDF chunk
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 80:
        pdf_chunk = pdf_base64[:60000]

    system_message = """
You are AMI — a highly accurate laboratory interpretation AI.
You analyse CBC, chemistry panels, renal, liver, infection markers.

STRICT RULES:
- NEVER guess missing values.
- NEVER invent lab markers.
- ONLY use values found in text/PDF.
- If infection markers are missing, explicitly state this.
- Output STRICT JSON ONLY.

JSON FORMAT:
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
Extracted Text:
{extracted_text}

PDF Fallback:
{pdf_chunk}

Return ONLY the JSON structure described.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4.1",   # ✅ MUCH BETTER THAN 4o-mini
            max_output_tokens=2000,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )

        ai_text = response.output_text.strip()
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        # Extract pure JSON
        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"): ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}

# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(pdf_bytes):
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text = ""
        for page in reader.pages:
            extracted = page.extract_text() or ""
            text += extracted + "\n"

        return text.strip()

    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""

# ----------------------------
# PROCESS NEXT REPORT
# ----------------------------
def process_next_report():
    print("🔍 Checking for queued reports...")

    response = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "processing") \
        .limit(1) \
        .execute()

    if not response.data:
        return None

    report = response.data[0]
    print("📄 Processing:", report["id"])

    file_path = report["file_path"]

    # Download PDF
    try:
        print("⬇️ Downloading PDF...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ Download error:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report["id"]).execute()
        return None

    # Extract text
    extracted_text = extract_text_from_pdf(pdf_bytes)

    # Base64 fallback
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    # AI processing
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    if "error" in ai_json:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json
        }).eq("id", report["id"]).execute()
        return None

    # Save results
    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json,
        "extracted_text": extracted_text,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", [])
    }).eq("id", report["id"]).execute()

    print("✅ Completed:", report["id"])

# ----------------------------
# MAIN LOOP
# ----------------------------
print("🚀 Worker running...")

while True:
    try:
        process_next_report()
    except Exception as e:
        print("❌ Crash prevented:", e)

    time.sleep(5)
