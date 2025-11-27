import time
import json
import base64
import io
import os

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader   # ✅ CORRECT LIBRARY


# =========================
# ENVIRONMENT
# =========================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# =========================
# AI CALL (HIGH ACCURACY)
# =========================

def call_ami_ai(extracted_text: str, pdf_base64: str):
    """
    Send lab text + (optional) PDF snippet to OpenAI and return parsed JSON.
    """

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
2. NEVER invent lab values, NEVER guess diagnoses.
3. If markers are missing, state clearly that they cannot be evaluated.
4. Be conservative and explicit about uncertainty.
5. Output STRICT JSON ONLY — no markdown, no prose outside JSON.

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
Extracted Lab Text (cleaned, possibly truncated):
{extracted_text}

PDF Fallback (may be empty, limited in size):
{pdf_chunk}

Instructions:
- Extract all CBC and chemistry values you can find.
- Comment on infection/inflammation ONLY if markers are present.
- If data is incomplete or non-specific, say so.
- Return ONLY the JSON object described above.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",              # ✅ higher accuracy model
            max_output_tokens=1800,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        ai_text = response.output_text.strip()

        # Clean up any accidental markdown wrappers
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        # Ensure we only keep the JSON object
        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"): ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# =========================
# PDF → TEXT EXTRACT
# =========================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF bytes blob using pypdf.
    """
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text_chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_chunks.append(page_text)

        return "\n".join(text_chunks).strip()

    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""


# =========================
# PROCESS NEXT REPORT
# =========================

def process_next_report():
    print("🔍 Checking for queued reports...")

    # 1️⃣ Get one queued report
    queued = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .limit(1)
        .execute()
    )

    if not queued.data:
        print("⏳ No queued reports.")
        return

    report = queued.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Found queued report: {report_id} ({file_path})")

    # 2️⃣ Mark as processing
    supabase.table("reports").update(
        {"ai_status": "processing"}
    ).eq("id", report_id).execute()

    # 3️⃣ Download PDF from Supabase Storage
    try:
        print("⬇️ Downloading PDF from Supabase storage...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ Could not download PDF:", e)
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": {"error": f"PDF download failed: {str(e)}"},
            }
        ).eq("id", report_id).execute()
        return

    # 4️⃣ Extract text from PDF
    print("📖 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_bytes)

    # 5️⃣ Encode PDF as base64 (for fallback OCR if needed)
    pdf_base64 = base64.b64encode(pdf_bytes).decode()

    # 6️⃣ Call AMI AI
    print("🧠 Calling AMI AI…")
    ai_json = call_ami_ai(extracted_text, pdf_base64)

    # 7️⃣ Handle errors
    if isinstance(ai_json, dict) and "error" in ai_json:
        print("❌ AI failed:", ai_json["error"])
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": ai_json,
                "extracted_text": extracted_text,
            }
        ).eq("id", report_id).execute()
        return

    # 8️⃣ Save results back to Supabase
    print("💾 Saving AI results to database...")
    supabase.table("reports").update(
        {
            "ai_status": "completed",
            "ai_results": ai_json,
            "extracted_text": extracted_text,
            "cbc_json": ai_json.get("cbc_values", {}),
            "trend_json": ai_json.get("trend_summary", []),
        }
    ).eq("id", report_id).execute()

    print(f"✅ Report completed: {report_id}")


# =========================
# MAIN LOOP
# =========================

if __name__ == "__main__":
    print("🚀 AMI Worker started (pypdf + gpt-4o)…")
    while True:
        try:
            process_next_report()
        except Exception as e:
            # Never let the worker die
            print("❌ Worker crash prevented:", e)

        time.sleep(5)
