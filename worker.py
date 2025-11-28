import os
import json
import time
import traceback
from typing import Optional

import requests
from pypdf import PdfReader
from dotenv import load_dotenv
from supabase import create_client, Client
from openai import OpenAI

# -------------------------------------------------
# ENV + CLIENTS
# -------------------------------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_KEY")
    or None
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL:
    raise Exception("Missing SUPABASE_URL")
if not SUPABASE_KEY:
    raise Exception("Missing SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY")
if not OPENAI_API_KEY:
    raise Exception("Missing OPENAI_API_KEY")

print("Supabase URL Loaded:", bool(SUPABASE_URL))
print("Supabase Key Loaded:", bool(SUPABASE_KEY))
print("OpenAI Key Loaded:", bool(OPENAI_API_KEY))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET_NAME = "reports"

# -------------------------------------------------
# STRICT JSON SCHEMA FOR STRUCTURED OUTPUTS
# -------------------------------------------------
JSON_SCHEMA = {
    "name": "ami_lab_interpretation",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "array",
                "items": {"type": "string"},
            },
            "trend_summary": {
                "type": "array",
                "items": {"type": "string"},
            },
            "flagged_results": {
                "type": "array",
                "items": {"type": "string"},
            },
            "interpretation": {
                "type": "array",
                "items": {"type": "string"},
            },
            "risk_level": {"type": "string"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
            },
            "when_to_seek_urgent_care": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cbc_values": {"type": "object"},
            "chemistry_values": {"type": "object"},
            "disclaimer": {"type": "string"},
        },
        "required": [
            "summary",
            "trend_summary",
            "flagged_results",
            "interpretation",
            "risk_level",
            "recommendations",
            "when_to_seek_urgent_care",
            "cbc_values",
            "chemistry_values",
            "disclaimer",
        ],
        "additionalProperties": False,
    },
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------


def download_pdf_from_supabase(file_path: str, local_path: str) -> None:
    """
    Download a PDF from the private Supabase bucket using the SERVICE ROLE key.
    No public URLs are used – this is fully POPIA-safe.
    """
    try:
        data: bytes = supabase.storage.from_(BUCKET_NAME).download(file_path)
    except Exception as e:
        # This error string is kept generic and used for ai_error – no PHI
        raise RuntimeError(f"Storage download failed: {e}") from e

    with open(local_path, "wb") as f:
        f.write(data)


def extract_pdf_text(local_path: str) -> str:
    """
    Extract text from the PDF using pypdf. Returns a single string.
    """
    try:
        reader = PdfReader(local_path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text.strip())
        text = "\n\n".join(chunks).strip()
        return text
    except Exception as e:
        print("PDF extraction error:", e)
        traceback.print_exc()
        return ""


def run_ai_interpretation(extracted_text: str) -> dict:
    """
    Call OpenAI Responses API with structured JSON output.
    """
    # Safety: cap length so we don’t blow the context
    if len(extracted_text) > 15000:
        extracted_text = extracted_text[:15000]

    prompt = (
        "You are AMI (Artificial Medical Intelligence), a clinical pathology "
        "assistant for doctors.\n"
        "You receive a blood test report (CBC +/- chemistry and other labs).\n"
        "Analyse patterns, disease profiles, and clinical risk. Focus on:\n"
        "- Flagged / abnormal values\n"
        "- Consistent patterns (e.g. iron deficiency, infection, liver injury)\n"
        "- Level of clinical concern\n"
        "- Recommendations for follow-up tests and clinical review\n\n"
        "IMPORTANT:\n"
        "- Do NOT invent tests that are not in the text.\n"
        "- Keep language doctor-level but concise and practical.\n"
        "- Do NOT include any patient identifiers in your output.\n"
        "- Output MUST follow the provided JSON schema exactly.\n\n"
        "Lab Report Text:\n"
        "----------------\n"
        f"{extracted_text}\n"
    )

    response = client.responses.create(
        model="gpt-4.1-mini",  # good balance of cost + quality
        input=prompt,
        max_output_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": JSON_SCHEMA,
        },
    )

    # For structured outputs, the model usually returns JSON as text in the first content block.
    # We keep this defensive in case the SDK layout changes.
    try:
        # Preferred: output_text if it exists and is JSON
        raw = getattr(response, "output_text", None)
        if not raw:
            raw = response.output[0].content[0].text  # type: ignore[attr-defined]
        parsed = json.loads(raw)
        return parsed
    except Exception:
        print("Failed to parse structured JSON output, dumping response:")
        print(response)
        raise


def update_report_status(
    report_id: str,
    *,
    ai_status: str,
    ai_results: Optional[dict] = None,
    extracted_text: Optional[str] = None,
    ai_error: Optional[str] = None,
    cbc_json: Optional[dict] = None,
    trend_json=None,
) -> None:
    """
    Central helper to update a report row. Never writes patient identifiers,
    only AI-related columns.
    """
    payload: dict = {"ai_status": ai_status}

    if ai_results is not None:
        payload["ai_results"] = ai_results
    if extracted_text is not None:
        payload["extracted_text"] = extracted_text
    if ai_error is not None:
        # Truncate to avoid overly long error logs in DB
        payload["ai_error"] = ai_error[:1000]
    if cbc_json is not None:
        payload["cbc_json"] = cbc_json
    if trend_json is not None:
        payload["trend_json"] = trend_json

    supabase.table("reports").update(payload).eq("id", report_id).execute()


# -------------------------------------------------
# MAIN JOB PROCESSOR
# -------------------------------------------------


def process_next_job() -> None:
    """
    Find the next queued report, download the PDF from the private bucket,
    extract text, run AI, and store the results.
    """
    job_resp = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not job_resp.data:
        print("No queued jobs.")
        return

    report = job_resp.data[0]
    report_id = report["id"]
    file_path = report.get("file_path")

    print("\n-----------------------------")
    print(f"Processing report: {report_id}")
    print("-----------------------------\n")

    if not file_path:
        print("Report has no file_path set.")
        update_report_status(
            report_id,
            ai_status="failed",
            ai_error="Missing file_path for report",
        )
        return

    local_pdf_path = "/tmp/report.pdf"

    # 1) Download PDF using Supabase Storage API (private, service-role)
    try:
        download_pdf_from_supabase(file_path, local_pdf_path)
    except Exception as e:
        print("Download error:", e)
        traceback.print_exc()
        update_report_status(
            report_id,
            ai_status="failed",
            ai_error=f"Download error: {e}",
        )
        return

    # 2) Extract text
    extracted_text = extract_pdf_text(local_pdf_path)

    if not extracted_text or len(extracted_text) < 20:
        print("PDF extraction produced insufficient text.")
        update_report_status(
            report_id,
            ai_status="failed",
            extracted_text=extracted_text,
            ai_error="PDF extraction produced no readable text",
        )
        return

    # 3) Run AI interpretation
    try:
        result_json = run_ai_interpretation(extracted_text)
    except Exception as e:
        print("AI error:", e)
        traceback.print_exc()
        update_report_status(
            report_id,
            ai_status="failed",
            extracted_text=extracted_text,
            ai_error=f"AI error: {e}",
        )
        return

    # 4) Save success back into Supabase
    cbc_values = result_json.get("cbc_values", {}) if isinstance(result_json, dict) else {}
    trend_summary = result_json.get("trend_summary", []) if isinstance(result_json, dict) else []

    update_report_status(
        report_id,
        ai_status="completed",
        ai_results=result_json,
        extracted_text=extracted_text,
        cbc_json=cbc_values,
        trend_json=trend_summary,
    )

    print(f"Report completed: {report_id}")


# -------------------------------------------------
# WORKER LOOP
# -------------------------------------------------


def main() -> None:
    print("AMI Worker started. Ready for reports...")
    while True:
        try:
            process_next_job()
        except Exception as e:
            # Catch any unexpected crash, log generic info only
            print("Worker runtime error:", e)
            traceback.print_exc()
        time.sleep(4)


if __name__ == "__main__":
    main()
