import time
import json
import os
import io

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader


# ----------------------------
# Environment + clients
# ----------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY / OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# PDF → TEXT
# ----------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extracts text from all pages of the PDF.
    If the PDF is image-only, this will return little / no text.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        chunks = []

        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(text)

        return "\n\n".join(chunks).strip()

    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""


def truncate_lab_text(text: str, max_chars: int = 12000) -> str:
    """
    Keep the most important parts of the report while staying small.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    head = text[:8000]
    tail = text[-4000:]
    return head + "\n...\n[TRUNCATED]\n...\n" + tail


# ----------------------------
# AI CALL (HIGH ACCURACY)
# ----------------------------

def call_ami_ai(extracted_text: str) -> dict:
    """
    Call GPT-4o with STRICT JSON rules.
    We only send cleaned lab text (no base64), so we stay inside token limits.
    """

    if not extracted_text or len(extracted_text.strip()) < 200:
        # Not enough machine-readable text – likely a scanned image
        return {
            "error": "Not enough machine-readable text was found in this PDF for AI analysis."
        }

    lab_text = truncate_lab_text(extracted_text)

    system_message = """
You are AMI — an advanced laboratory interpretation assistant for clinicians.

You receive the plain text of laboratory reports (CBC, chemistry, inflammatory markers, etc.).
Your job is to extract the values and provide a cautious, structured interpretation.

SAFETY & SCOPE
- You are NOT the treating doctor and you do NOT make diagnoses.
- You highlight abnormal or concerning patterns in the lab values.
- You always recommend correlation with clinical findings and specialist review when appropriate.

ABSOLUTE RULES
1. Base everything strictly on the provided lab text. If a value is not present, do not guess it.
2. If an important test (e.g. WBC, neutrophils, CRP, ESR, creatinine, urea, ALT, AST, etc.) is missing,
   explicitly mention that it is not available.
3. If results are non-specific or borderline, say that clearly.
4. Never state a definitive diagnosis. Use language like "may suggest", "could be consistent with", etc.
5. Output STRICT JSON ONLY (no extra commentary, no markdown).

EXPECTED JSON SCHEMA
{
  "summary": [ "short bullet sentences of the main lab story" ],
  "trend_summary": [ "if there are multiple dates, describe the trend; otherwise keep this short" ],
  "flagged_results": [
    { "test": "WBC", "value": 15.2, "unit": "10^9/L", "flag": "high", "comment": "Mild leukocytosis, could indicate infection or inflammation" }
  ],
  "interpretation": [
    "Longer paragraphs (2–5) explaining what the overall pattern may indicate, always cautiously."
  ],
  "risk_level": "low" | "moderate" | "high",
  "recommendations": [
    "Example: Correlate with symptoms (fever, pain, weight loss).",
    "Example: Consider repeat CBC in 1–2 weeks if clinically indicated."
  ],
  "cbc_values": {
    "WBC": { "value": 7.4, "unit": "10^9/L", "ref_range": "4.0–11.0" },
    "RBC": { "value": 4.8, "unit": "10^12/L", "ref_range": "4.5–6.0" },
    "Hb":  { "value": 13.6, "unit": "g/dL", "ref_range": "13.0–17.0" },
    "Platelets": { "value": 250, "unit": "10^9/L", "ref_range": "150–450" }
  },
  "chemistry_values": {
    "Creatinine": { "value": 88, "unit": "µmol/L", "ref_range": "60–110" },
    "Urea": { "value": 5.2, "unit": "mmol/L", "ref_range": "3.0–8.0" },
    "CRP": { "value": 35, "unit": "mg/L", "ref_range": "<5" }
  },
  "disclaimer": "This AI interpretation is assistive and not a medical diagnosis. Always correlate with clinical findings and local laboratory reference ranges."
}
"""

    user_message = f"""
Here is the extracted laboratory report text:

\"\"\"{lab_text}\"\"\"

Please:
- Extract as many numeric lab results as you reliably can into cbc_values and chemistry_values.
- Flag clearly abnormal or clinically important values.
- If data is insufficient for strong conclusions, say so.
- Return ONLY a single JSON object exactly matching the schema described by the system message.
"""

    try:
        resp = openai_client.responses.create(
            model="gpt-4o",
            max_output_tokens=900,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        ai_text = resp.output_text.strip()

        # Clean any accidental markdown fences
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        # Ensure we only keep the JSON object
        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"): ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# ----------------------------
# PROCESS NEXT REPORT
# ----------------------------

def process_next_report():
    print("🔍 Checking for queued reports...")

    res = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "processing")
        .limit(1)
        .execute()
    )

    if not res.data:
        print("🟦 No queued reports.")
        return

    report = res.data[0]
    report_id = report["id"]
    file_path = report.get("file_path")

    print(f"📄 Found queued report: {report_id} ({file_path})")

    if not file_path:
        msg = "Missing file_path for report row."
        print("❌", msg)
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_results": {"error": msg}}
        ).eq("id", report_id).execute()
        return

    # Download PDF from Supabase storage
    try:
        print("⬇️ Downloading PDF from Supabase storage...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        msg = f"Could not download PDF: {e}"
        print("❌", msg)
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_results": {"error": msg}}
        ).eq("id", report_id).execute()
        return

    # Extract text from PDF
    extracted_text = extract_text_from_pdf(pdf_bytes)

    if not extracted_text or len(extracted_text.strip()) < 50:
        msg = (
            "No readable text could be extracted from this PDF. "
            "It may be a scanned image without OCR."
        )
        print("⚠️", msg)
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": {"error": msg},
                "extracted_text": extracted_text or "",
            }
        ).eq("id", report_id).execute()
        return

    print("📝 Extracted text length:", len(extracted_text))

    # Call AI
    ai_json = call_ami_ai(extracted_text)

    if "error" in ai_json:
        print("❌ AI failed:", ai_json["error"])
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": ai_json,
                "extracted_text": extracted_text,
            }
        ).eq("id", report_id).execute()
        return

    # Save results back to Supabase
    print("💾 Saving AI results to Supabase...")
    update_payload = {
        "ai_status": "completed",
        "ai_results": ai_json,
        "extracted_text": extracted_text,
        # These match the columns you showed in Supabase
        "cbc_json": ai_json.get("cbc_values") or {},
        "trend_json": ai_json.get("trend_summary") or [],
    }

    supabase.table("reports").update(update_payload).eq("id", report_id).execute()
    print("✅ Report processed:", report_id)


# ----------------------------
# MAIN LOOP
# ----------------------------

if __name__ == "__main__":
    print("🚀 AMI Worker started (text-only, gpt-4o)…")

    while True:
        try:
            process_next_report()
        except Exception as e:
            print("💥 Worker loop error (will continue):", e)

        time.sleep(5)
