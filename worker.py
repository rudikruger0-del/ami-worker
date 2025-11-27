import time
import json
import os
import io

from supabase import create_client
from openai import OpenAI
from PyPDF2 import PdfReader

# ------------------------------
# Load environment
# ------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing environment variables.")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# PDF → TEXT
# ------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text.strip())

        text = "\n\n".join(chunks)
        if len(text) > 8000:
            text = text[:8000] + "\n\n[Truncated for length]"

        return text
    except Exception as e:
        print("❌ PDF extraction error:", e)
        return ""

# ------------------------------
# GPT-4o AI CALL
# ------------------------------
def run_ai(extracted_text: str) -> dict:
    if len(extracted_text.strip()) < 30:
        return {"error": "No usable text extracted."}

    system_prompt = """
You are AMI, a medical AI.
Provide structured, detailed interpretation of lab text.
Do NOT hallucinate values.
Output STRICT JSON only.
"""

    user_prompt = f"""
Extracted lab report text:
\"\"\"{extracted_text}\"\"\"

Generate structured JSON with:
summary,
trend_summary,
flagged_results,
interpretation,
risk_level,
recommendations,
when_to_seek_urgent_care,
cbc_values,
chemistry_values,
disclaimer.
"""

    try:
        res = openai_client.responses.create(
            model="gpt-4o",
            max_output_tokens=1800,
            response_format={"type": "json"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        return res.output[0].content[0].json

    except Exception as e:
        print("❌ AI ERROR:", e)
        return {"error": str(e)}

# ------------------------------
# PROCESS REPORT
# ------------------------------
def process_one():
    print("🔍 Checking for queued reports...")

    res = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "queued") \
        .order("created_at", desc=False) \
        .limit(1) \
        .execute()

    if not res.data:
        print("No queued reports.")
        return

    report = res.data[0]
    rid = report["id"]
    file_path = report["file_path"]

    print("📄 Processing:", rid)

    supabase.table("reports").update({"ai_status": "processing"}).eq("id", rid).execute()

    # Download PDF
    try:
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ Could not download PDF:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": "Could not download PDF."}
        }).eq("id", rid).execute()
        return

    # Extract text
    text = extract_text_from_pdf(pdf_bytes)
    if not text:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": "PDF contains no readable text."}
        }).eq("id", rid).execute()
        return

    # Run AI
    print("🤖 Running GPT-4o...")
    ai = run_ai(text)

    if "error" in ai:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai
        }).eq("id", rid).execute()
        return

    # Save to DB
    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai,
        "extracted_text": text,
        "cbc_json": ai.get("cbc_values", {}),
        "trend_json": ai.get("trend_summary", [])
    }).eq("id", rid).execute()

    print("✅ Completed:", rid)

# ------------------------------
# MAIN LOOP
# ------------------------------
if __name__ == "__main__":
    print("🚀 AMI Worker started")

    while True:
        try:
            process_one()
        except Exception as e:
            print("💥 Worker crash (continuing):", e)

        time.sleep(5)
