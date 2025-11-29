import os
import time
import json
import io
import traceback

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader

# -------- ENV & CLIENTS --------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# -------- CONFIDENCE ENGINE --------

def calculate_confidence_score(cbc):
    """
    Simple + safe confidence score based on:
    - extraction completeness
    - number of abnormalities
    - pattern clarity
    """

    score = 0

    # 1) Extraction quality (0–40)
    missing_values = [item for item in cbc if item.get("value") is None]
    extraction_quality = 40 - (len(missing_values) * 4)
    extraction_quality = max(5, extraction_quality)  # never 0 (unless unreadable)
    score += extraction_quality

    # 2) Pattern strength (0–40)
    abnormalities = [i for i in cbc if i.get("flag") in ["high", "low"]]

    if len(abnormalities) >= 4:
        score += 40
    elif len(abnormalities) == 3:
        score += 32
    elif len(abnormalities) == 2:
        score += 24
    elif len(abnormalities) == 1:
        score += 15
    else:
        score += 10  # normal CBC is still “clear”

    # 3) Report integrity (0–20)
    integrity = 20
    if len(cbc) < 10:
        integrity -= 8
    if any(i.get("units") is None for i in cbc):
        integrity -= 5

    integrity = max(0, integrity)
    score += integrity

    return min(score, 100)


# -------- HELPERS --------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF (no logging of content for POPIA safety)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        return "\n\n".join(pages)
    except Exception as e:
        print("PDF parse error:", e)
        return ""


def call_ai_on_report(text: str) -> dict:
    """
    Calls OpenAI with STRICT JSON return structure,
    now including expanded reasoning (but NOT diagnosis).
    """

    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You analyse CBC reports. You DO NOT diagnose. You only:\n"
        "• identify abnormal values\n"
        "• explain what these abnormalities are commonly associated with "
        "(e.g. infections, inflammation, anemia patterns, dehydration, viral illness, etc.)\n"
        "• suggest safe, non-diagnostic follow-up steps.\n\n"
        "You MUST return STRICT JSON:\n"
        "{\n"
        '  "patient": { "name": string|null, "age": number|null, "sex": "Male"|"Female"|"Unknown" },\n'
        '  "cbc": [ { "analyte": string, "value": number|null, "units": string|null,\n'
        '            "reference_low": number|null, "reference_high": number|null,\n'
        '            "flag": "low"|"normal"|"high"|"unknown" } ],\n'
        '  "summary": {\n'
        '      "impression": string,\n'
        '      "possible_associations": string,\n'
        '      "suggested_follow_up": string\n'
        '  }\n'
        "}\n"
        "Return ONLY that JSON. NO extra text."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


# -------- CORE JOB PROCESSING --------

def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path:
            raise ValueError(f"Missing file_path for report {report_id}")

        # Download PDF
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        # Extract text
        pdf_text = extract_text_from_pdf(pdf_bytes)
        merged_text = (l_text + "\n\n" + pdf_text).strip() if l_text else pdf_text

        if not merged_text.strip():
            raise ValueError("No extractable text")

        # Call AI
        ai_json = call_ai_on_report(merged_text)

        # --------------- ADD CONFIDENCE SCORE ---------------
        cbc = ai_json.get("cbc", [])
        confidence = calculate_confidence_score(cbc)
        ai_json["confidence_score"] = confidence
        # ----------------------------------------------------

        # Update DB
        supabase.table("reports").update(
            {
                "ai_status": "completed",
                "ai_results": ai_json,
                "ai_error": None,
            }
        ).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_json}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_error": err,
            }
        ).eq("id", report_id).execute()

        return {"error": err}


# -------- WORKER LOOP --------

def main():
    print("AMI Worker started… watching for jobs.")

    while True:
        try:
            res = (
                supabase.table("reports")
                .select("*")
                .eq("ai_status", "pending")
                .limit(1)
                .execute()
            )
            jobs = res.data or []

            if jobs:
                job = jobs[0]
                job_id = job["id"]
                print(f"🔎 Found job: {job_id}")

                supabase.table("reports").update(
                    {"ai_status": "processing"}
                ).eq("id", job_id).execute()

                process_report(job)
            else:
                print("No jobs...")

            time.sleep(5)

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
