import time
import json
import base64
import io
import os

from supabase import create_client, Client
from openai import OpenAI

from pypdf import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image


# -------------------------------------------
# ENV
# -------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -------------------------------------------
# OCR FUNCTION
# -------------------------------------------
def ocr_pdf(pdf_bytes):
    try:
        pages = convert_from_bytes(pdf_bytes)
        text = ""

        for p in pages:
            ocr_text = pytesseract.image_to_string(p)
            text += ocr_text + "\n"

        return text.strip()

    except Exception as e:
        print("❌ OCR error:", e)
        return ""


# -------------------------------------------
# Extract text with BOTH PdfReader + OCR
# -------------------------------------------
def extract_text(pdf_bytes):
    text = ""

    # Try normal text extraction
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

    except:
        pass

    # If NO text, run OCR
    if len(text.strip()) < 50:
        print("⚠️ No text found — using OCR…")
        text = ocr_pdf(pdf_bytes)

    return text.strip()


# -------------------------------------------
# GPT PROCESSING
# -------------------------------------------
def call_ai(extracted_text):

    if len(extracted_text) > 15000:
        extracted_text = extracted_text[:15000]

    system_prompt = """
You are AMI — a medical lab interpretation AI.

Rules:
- Only use values found in the text.
- If missing, say “not provided”.
- Output strict JSON in this format:

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
-----------------------
{extracted_text}
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=1200
        )

        txt = response.output_text.strip()

        # clean JSON
        if "{" in txt and "}" in txt:
            txt = txt[txt.index("{"): txt.rindex("}") + 1]

        return json.loads(txt)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# -------------------------------------------
# PROCESSING LOOP
# -------------------------------------------
def process_next():
    print("🔍 Checking for queued reports...")

    row = supabase.table("reports") \
        .select("*") \
        .eq("ai_status", "queued") \
        .limit(1) \
        .execute()

    if not row.data:
        return

    report = row.data[0]
    report_id = report["id"]

    print("📄 Processing:", report_id)

    # download pdf
    try:
        file_bytes = supabase.storage.from_("reports").download(report["file_path"])
    except Exception as e:
        print("❌ Download error:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report_id).execute()
        return

    # extract text
    extracted_text = extract_text(file_bytes)

    # call ai
    ai_json = call_ai(extracted_text)

    if "error" in ai_json:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": ai_json
        }).eq("id", report_id).execute()
        return

    # save
    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": ai_json,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", []),
        "extracted_text": extracted_text
    }).eq("id", report_id).execute()

    print("✅ Completed:", report_id)


print("🚀 AMI Worker (OCR + text) started…")

while True:
    try:
        process_next()
        time.sleep(4)
    except Exception as e:
        print("❌ Worker crash prevented:", e)
