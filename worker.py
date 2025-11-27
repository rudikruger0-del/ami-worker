import time
import json
import base64
import io
import os
import fitz   # PyMuPDF for PDF parsing

from supabase import create_client, Client
from openai import OpenAI

# ----------------------------
# ENVIRONMENT
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ============================================================
# 🧠 MAIN AI INTERPRETATION — TEXT MODEL (CHEAP + ACCURATE)
# ============================================================

def call_ami_from_text(text: str):
    system_msg = """
You are AMI — an advanced AI specializing in blood test interpretation.
You MUST output **medical-grade detail** without inventing values.

Rules:
1. Only use values that exist in the text.
2. If a marker is missing, state that it is missing.
3. Provide a full medical interpretation: patterns, risks, trends.
4. Return STRICT JSON only with NO explanations.

JSON format:
{
 "risk_level": "",
 "summary": [],
 "interpretation": [],
 "flagged_results": [],
 "recommendations": [],
 "when_to_seek_urgent_care": [],
 "cbc_values": {},
 "chemistry_values": {},
 "disclaimer": "This is assistive and not medical advice."
}
"""

    user_msg = f"Extracted lab report:\n{text}"

    response = openai_client.responses.create(
        model="gpt-4o",
        max_output_tokens=3000,
        input=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    raw = response.output_text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()

    # Clean extract JSON
    if raw.startswith("{"):
        raw = raw[: raw.rfind("}") + 1]

    return json.loads(raw)


# ============================================================
# 🧠 FALLBACK: VISION MODEL (GPT-4o-mini-vision)
# ============================================================

def extract_with_vision(pdf_bytes):
    """
    Sends each PDF page as an image to GPT-4o-mini-vision.
    More accurate than OCR and lighter than GPT-4o vision.
    """

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=180)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode()

        response = openai_client.responses.create(
            model="gpt-4o-mini-vision",
            max_output_tokens=2000,
            input=[
                {"role": "system", "content": "Extract all text exactly from this lab report page."},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Read ALL text."},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{b64}"
                        }
                    ]
                }
            ]
        )

        pages_text.append(response.output_text)

    return "\n".join(pages_text)


# ============================================================
# PDF → Extract TEXT FIRST (cheap, fast)
# ============================================================

def extract_text(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        all_text = ""

        for page in doc:
            t = page.get_text()
            if t:
                all_text += t + "\n"

        return all_text.strip()

    except Exception as e:
        print("PDF extract error:", e)
        return ""


# ============================================================
# PROCESS NEXT REPORT
# ============================================================

def process_next_report():
    print("🔍 Checking for queued reports...")

    # Get one queued
    q = supabase.table("reports").select("*").eq("ai_status", "processing").limit(1).execute()
    if not q.data:
        return None

    report = q.data[0]
    report_id = report["id"]
    file_path = report["file_path"]

    print(f"📄 Processing report: {report_id}")

    # Download PDF from Supabase
    try:
        file_res = supabase.storage.from_("reports").download(file_path)
        pdf_bytes = file_res
    except Exception as e:
        print("❌ PDF download error:", e)
        supabase.table("reports").update({"ai_status": "failed"}).eq("id", report_id).execute()
        return

    # 1 — TRY TEXT EXTRACTION
    extracted_text = extract_text(pdf_bytes)
    print("📌 Text extracted length:", len(extracted_text))

    # If text is too short → fallback to vision
    if len(extracted_text) < 50:
        print("⚠️ Not enough text → using Vision fallback")
        extracted_text = extract_with_vision(pdf_bytes)

    # 2 — CALL AI TEXT MODEL FOR INTERPRETATION
    try:
        ai_json = call_ami_from_text(extracted_text)
    except Exception as e:
        print("❌ Text model failed:", e)
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_results": {"error": str(e)}
        }).eq("id", report_id).execute()
        return

    # 3 — SAVE BACK TO DB
    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": extracted_text,
        "ai_results": ai_json,
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", [])
    }).eq("id", report_id).execute()

    print(f"✅ Completed: {report_id}")


# ============================================================
# MAIN LOOP
# ============================================================

print("🚀 AMI Worker started (hybrid text+vision)…")

while True:
    try:
        process_next_report()
    except Exception as e:
        print("❌ Worker crash prevented:", e)

    time.sleep(4)
