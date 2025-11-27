import time
import json
import base64
import fitz  # PyMuPDF
import os
from supabase import create_client, Client
from openai import OpenAI

# ENV
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------------------------------------
# Convert PDF pages → base64 PNG images
# ------------------------------------------------------------
def pdf_to_images(base64_pdf):
    pdf_bytes = base64.b64decode(base64_pdf)
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    image_list = []
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(dpi=200)  # high quality
        img_bytes = pix.tobytes("png")
        encoded = base64.b64encode(img_bytes).decode()
        image_list.append(encoded)

    return image_list

# ------------------------------------------------------------
# GPT-4o Vision call
# ------------------------------------------------------------
def call_ami_ai(images_base64):
    messages = [
        {
            "role": "system",
            "content": (
                "You are AMI, a medical AI. Analyze the lab report images and extract:"
                "\n- CBC values"
                "\n- Chemistry values"
                "\n- Flagged abnormalities"
                "\n- Summary"
                "\n- Interpretation"
                "\n- Recommendations"
                "\n- Risk level (low / moderate / high)"
                "\nReturn JSON ONLY in this structure:"
                "{"
                '"summary": [],'
                '"trend_summary": [],'
                '"flagged_results": [],'
                '"interpretation": [],'
                '"risk_level": "",'
                '"recommendations": [],'
                '"cbc_values": {},'
                '"chemistry_values": {}'
                "}"
            )
        }
    ]

    # Add images
    for img in images_base64:
        messages.append({
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Analyze this page."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{img}"
                }
            ]
        })

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=2000
    )

    text = response.choices[0].message["content"]
    return json.loads(text)

# ------------------------------------------------------------
# Save JSON → Supabase
# ------------------------------------------------------------
def save_results(report_id, results):
    supabase.table("reports").update({
        "ai_status": "completed",
        "ai_results": results
    }).eq("id", report_id).execute()

# ------------------------------------------------------------
# Fail Handler
# ------------------------------------------------------------
def fail_report(report_id, msg):
    supabase.table("reports").update({
        "ai_status": "failed",
        "ai_results": {"error": msg}
    }).eq("id", report_id).execute()

# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------
print("🚀 AMI Worker started (VISION mode, gpt-4o)…")

while True:
    try:
        time.sleep(3)

        # Find queued reports
        res = supabase.table("reports").select("*").eq("ai_status", "queued").limit(1).execute()
        if len(res.data) == 0:
            print("🔍 No queued reports.")
            continue

        report = res.data[0]
        report_id = report["id"]
        print(f"📄 Processing: {report_id}")

        # Update status → processing
        supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

        # -------------------------------
        # Download PDF from storage
        # -------------------------------
        pdf_path = f"{report_id}.pdf"
        download = supabase.storage.from_("reports").download(pdf_path)

        if download is None:
            fail_report(report_id, "Unable to download PDF from bucket.")
            continue

        pdf_bytes = download
        encoded_pdf = base64.b64encode(pdf_bytes).decode()

        # Convert PDF → image frames
        images = pdf_to_images(encoded_pdf)

        if len(images) == 0:
            fail_report(report_id, "PDF contains no readable pages.")
            continue

        # AI call
        print("🤖 Calling GPT-4o Vision…")
        ai_json = call_ami_ai(images)

        # Save output
        save_results(report_id, ai_json)

        print(f"✅ Completed: {report_id}")

    except Exception as e:
        print("❌ Worker crashed:", str(e))
