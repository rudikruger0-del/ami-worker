import time
import json
import base64
import io
import os

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader   # correct and installed


# ----------------------------
# ENVIRONMENT
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# PDF → TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(pdf_bytes):
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text = ""
        for page in reader.pages:
            t = page.extract_text() or ""
            text += t + "\n"

        return text.strip()
    except Exception as e:
        print("❌ PDF extraction error:", e)
        return ""


# ----------------------------
# AI CALL
# ----------------------------
def call_ami_ai(extracted_text: str, pdf_base64: str):
    # CLEAN TEXT
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 15000:
            extracted_text = extracted_text[:15000]

    # FALLBACK: SEND SMALL PIECE OF PDF IF TEXT TOO SHORT
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 50:
        pdf_chunk = pdf_base64[:30000]  # tiny, safe for GPT-4o

    system_prompt = """
You are AMI — expert medical laboratory analysis system.

CRITICAL RULES:
- ONLY use values found in the text/PDF.
- NEVER invent values or diagnoses.
- If markers (WBC, NEUT, CRP, ESR) are missing, clearly state that.
- Output STRICT JSON ONLY.

JSON format:
{
  "summary": [],
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

PDF Fallback (may be empty):
{pdf_chunk}

Return ONLY the JSON structure exactly as described.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini",     # safe & fast, low tokens
            max_output_tokens=1200,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        ai_out = response.output_text.strip()

        # CLEAN
        ai_out = ai_out.replace("```json", "").replace("```", "").strip()
        ai_out = ai_out[ai_out.index("{"): ai_out.rindex("}") + 1]

        return json.loads(ai_out)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# ----------------------------
# PROCESS NEXT REPORT
# ----------------------------
def process_next_report():
    print("🔍 Checking for queued reports...")

    # 1️⃣ LOOK FOR QUEUED REPORTS (FIXED!)
    response = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "queued") \
        .order("id", desc=False) \
        .limit(1) \
        .execute()

    if not response.data:
        return None

    report = response.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print("📄 Found report:", report_id)

    # 2️⃣ LOCK IT
    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

    # 3️⃣ DOWNLOAD
    try:
        print("⬇️ Downloading PDF...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ Download error:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report_id).execute()
        return

    # 4️⃣ EXTRACT TEXT
    extracted_text = extract_text_from_pdf(pdf_bytes)
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    # 5️⃣ AI
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    if "error" in ai_json:
        print("❌ AI failed:", ai_json["error"])
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json
        }).eq("id", report_id).execute()
        return

    # 6️⃣ SAVE
    print("💾 Saving AI results...")

    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", []),
        "extracted_text": extracted_text
    }).eq("id", report_id).execute()

    print("✅ Report completed:", report_id)


# ----------------------------
# MAIN LOOP
# ----------------------------
print("🚀 AMI Worker started (text-only, safe tokens)…")

while True:
    try:
        process_next_report()
    except Exception as e:
        print("❌ Worker crash prevented:", e)
    time.sleep(5)
