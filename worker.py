import os
import json
import time
import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# ------------------------------
# PDF TEXT EXTRACTION
# ------------------------------
def extract_pdf_text(path):
    try:
        reader = PdfReader(path)
        text = ""
        for p in reader.pages:
            t = p.extract_text()
            if t:
                text += t + "\n"
        return text.strip()
    except Exception as e:
        print("PDF ERROR:", e)
        return ""

# ------------------------------
# AI INTERPRETATION
# ------------------------------
def run_ai(text):
    prompt = f"""
    You are AMI Health AI. Extract CBC values, trends, risks.
    Return ONLY valid JSON.

    LAB REPORT:
    -----------
    {text}
    """

    completion = client.chat.completions.create(
        model="gpt-4.1",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {
                    "cbc_values": {"type": "object"},
                    "trend_summary": {"type": "array", "items": {"type": "string"}},
                    "interpretation": {"type": "array", "items": {"type": "string"}},
                    "flagged_results": {"type": "array", "items": {"type": "string"}},
                    "risk_level": {"type": "string"},
                },
                "required": ["cbc_values", "trend_summary", "interpretation", "flagged_results", "risk_level"]
            },
        },
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return json.loads(completion.choices[0].message.content)

# ------------------------------
# SIGNED URL FETCH
# ------------------------------
def download_pdf_signed(path, local):
    signed = supabase.storage.from_("reports").create_signed_url(path, 3600)
    url = signed["signedURL"]

    r = requests.get(url)
    r.raise_for_status()

    with open(local, "wb") as f:
        f.write(r.content)

# ------------------------------
# PROCESS NEXT REPORT
# ------------------------------
def process_next():
    job = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .limit(1)
        .execute()
    )

    if not job.data:
        print("No jobs...")
        return

    report = job.data[0]
    rid = report["id"]
    fpath = report.get("file_path")

    if not fpath:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"Missing file_path for report {rid}"
        }).eq("id", rid).execute()
        return

    print("\n--- Processing", rid, " ---")

    # 1. Download PDF via signed URL
    local_pdf = f"/tmp/{rid}.pdf"
    try:
        download_pdf_signed(fpath, local_pdf)
    except Exception as e:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"PDF download failed: {e}"
        }).eq("id", rid).execute()
        return

    # 2. Extract text
    text = extract_pdf_text(local_pdf)
    if len(text) < 15:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": "Unreadable or empty PDF"
        }).eq("id", rid).execute()
        return

    # 3. Run AI
    try:
        result = run_ai(text)
    except Exception as e:
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": f"AI error: {e}"
        }).eq("id", rid).execute()
        return

    # 4. Save results
    supabase.table("reports").update({
        "ai_status": "completed",
        "extracted_text": text,
        "ai_results": result,
        "cbc_json": result.get("cbc_values", {}),
        "trend_json": result.get("trend_summary", []),
    }).eq("id", rid).execute()

    print("✔ Completed:", rid)

# ------------------------------
# MAIN LOOP
# ------------------------------
def main():
    print("AMI Worker running…")
    while True:
        try:
            process_next()
        except Exception as e:
            print("Worker crashed:", e)
        time.sleep(4)

if __name__ == "__main__":
    main()
