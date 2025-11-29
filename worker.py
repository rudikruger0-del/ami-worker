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
    print("⚠️  OPENAI_API_KEY is missing — AI calls will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# -------- PDF HELPERS --------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from a PDF (safe: no logging of contents)."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n\n".join(pages)
    except Exception as e:
        print("PDF parse error:", e)
        return ""


# -------- AI CALL --------

def call_ai_on_report(text: str) -> dict:
    """
    Sends extracted text to OpenAI and receives structured JSON.
    NEW: Includes confidence_score + possible_relevant_patterns
    """

    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing a Complete Blood Count (CBC) report. "
        "You MUST NOT provide a diagnosis. Only describe abnormalities and broad physiological patterns.\n\n"

        "Return STRICT JSON with EXACTLY this structure:\n"
        "{\n"
        "  \"patient\": {\n"
        "    \"name\": string | null,\n"
        "    \"age\": number | null,\n"
        "    \"sex\": \"Male\" | \"Female\" | \"Unknown\"\n"
        "  },\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": string,\n"
        "      \"value\": number | null,\n"
        "      \"units\": string | null,\n"
        "      \"reference_low\": number | null,\n"
        "      \"reference_high\": number | null,\n"
        "      \"flag\": \"low\" | \"normal\" | \"high\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"impression\": string,\n"
        "    \"suggested_follow_up\": string,\n"
        "    \"possible_relevant_patterns\": [string],\n"
        "    \"confidence_score\": number\n"
        "  }\n"
        "}\n\n"

        "Rules:\n"
        "- 'possible_relevant_patterns' are broad categories ONLY: e.g. \"infection\", \"viral illness\", "
        "\"dehydration\", \"inflammation\", \"allergy\", \"stress response\".\n"
        "- These ARE NOT DIAGNOSES.\n"
        "- 'confidence_score' must be a float between 0.0 and 1.0.\n"
        "- Output ONLY valid JSON. No extra text."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,  # slightly higher for better pattern detection
    )

    content = response.choices[0].message.content

    if isinstance(content, list):
        raw = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    else:
        raw = content or ""

    if not raw.strip():
        raise ValueError("AI returned empty content")

    return json.loads(raw)


# -------- REPORT PROCESSOR --------

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

        merged_text = (l_text + "\n\n" + pdf_text).strip() if pdf_text else l_text

        if not merged_text.strip():
            raise ValueError("No text extracted from report")

        # AI analysis
        ai_json = call_ai_on_report(merged_text)

        # Save results
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
    print("⚡ AMI Worker started — watching for pending jobs...")

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

                print(f"🔎 New job found: {job_id}")

                supabase.table("reports").update(
                    {"ai_status": "processing"}
                ).eq("id", job_id).execute()

                process_report(job)
            else:
                print("No pending jobs...")

            time.sleep(5)

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
