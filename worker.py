import time
import json
import base64
import io
import os

from supabase import create_client, Client
from pypdf import PdfReader
import requests

# ----------------------------
# Environment
# ----------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY env vars")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = "gpt-4o-mini"  # can change to gpt-4.1-mini or gpt-4o later


# ----------------------------
# PDF → Extract Text (searchable PDFs)
# ----------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Use pypdf to extract text from each page.
    Works for real, text-based lab PDFs.
    Scanned image-only PDFs will return empty text.
    """
    try:
        stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(stream)

        all_text = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            all_text.append(page_text)

        text = "\n".join(all_text).strip()
        return text

    except Exception as e:
        print("❌ PDF extract error:", e)
        return ""


# ----------------------------
# AI CALL – EXTREME DETAIL, JSON ONLY
# ----------------------------

def call_ami_ai(extracted_text: str) -> dict:
    """
    Call OpenAI Responses API with:
    - detailed doctor-level interpretation
    - strict JSON output
    - token-friendly truncation
    """

    # Clean & truncate text for safety
    if extracted_text:
        # collapse massive whitespace
        compact = " ".join(extracted_text.split())
        # cap at ~12k chars (≈ 3k tokens)
        if len(compact) > 12000:
            compact = compact[:12000]
        extracted_text = compact
    else:
        extracted_text = ""

    system_prompt = """
You are AMI — an advanced clinical laboratory interpretation AI.

Context:
- You interpret blood tests (CBC, chemistry, inflammatory/infection markers).
- Your audience is DOCTORS and CLINICIANS.
- Your tone is clear, concise, and clinically useful.

TASK:
From the provided lab text, you must:

1) Extract key CBC and chemistry values (if present):
   - CBC: WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, NEU%, LYM%, MONO%, EOS%, BASO%, etc.
   - Chemistry: creatinine, urea, eGFR, sodium, potassium, CRP, ESR, etc.
   - Include value, unit, and whether it is HIGH / LOW / NORMAL when possible.

2) Identify and EXPLAIN clinically relevant patterns:
   - Infection (e.g., high WBC with neutrophilia)
   - Inflammation
   - Anaemia patterns
   - Renal impairment
   - Electrolyte disturbances
   - Any critical abnormalities

3) Produce a detailed DOCTOR-LEVEL interpretation:
   - 3–6 bullet points in "summary"
   - 3–6 bullet points in "interpretation" focusing on organ systems / problems
   - Clear "risk_level": "low", "moderate", "high", or "critical"
   - Practical "recommendations" (follow-up tests, clinical correlation, time-frame)
   - "when_to_seek_urgent_care" with specific warning symptoms if risk is not low

4) You MUST output STRICT JSON ONLY with this structure:

{
  "summary": [
    "High-level bullet point...",
    "Another key finding..."
  ],
  "trend_summary": [
    "If previous values are present, describe trend. If not, say trend cannot be assessed."
  ],
  "flagged_results": [
    {
      "name": "Haemoglobin",
      "value": "10.2",
      "unit": "g/dL",
      "flag": "low",
      "clinical_significance": "Suggestive of mild normocytic anaemia, clinical correlation required."
    }
  ],
  "interpretation": [
    "System-level or problem-based interpretation in doctor language.",
    "Mention likely causes, differentials, and what to consider next."
  ],
  "risk_level": "low",
  "recommendations": [
    "Concrete follow-up actions, repeat tests, or referrals, with rough time frames."
  ],
  "when_to_seek_urgent_care": [
    "Warning signs/symptoms that should prompt urgent review or ER referral."
  ],
  "cbc_values": {
    "WBC": { "value": "7.2", "unit": "x10^9/L", "flag": "normal" },
    "HGB": { "value": "14.1", "unit": "g/dL", "flag": "normal" }
  },
  "chemistry_values": {
    "Creatinine": { "value": "88", "unit": "µmol/L", "flag": "normal" }
  },
  "disclaimer": "This AI report is assistive and informational only and is NOT a medical diagnosis. Always correlate with clinical findings and local guidelines."
}

RULES:
- NEVER invent values or units that are not clearly present.
- If a parameter is missing, simply omit it from cbc_values/chemistry_values.
- If trends cannot be assessed, say so explicitly in trend_summary.
- If data is extremely limited, still fill the JSON fields but state the limitation clearly.
- Output MUST be valid JSON. Do not add explanations before or after the JSON.
"""

    user_prompt = f"""
LAB REPORT RAW TEXT (as extracted from PDF):

\"\"\"{extracted_text}\"\"\"

Use ONLY the information available here.
Return STRICT JSON exactly in the schema described above.
"""

    payload = {
        "model": OPENAI_MODEL,
        "response_format": {"type": "json_object"},
        "input": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": system_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt,
                    }
                ],
            },
        ],
        # soft cap to avoid runaway token usage
        "max_output_tokens": 1800,
    }

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(OPENAI_RESPONSES_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # New Responses API shape:
        # data["output"][0]["content"][0]["text"]
        try:
            text = data["output"][0]["content"][0]["text"]
        except Exception:
            # fallback – just return raw for debugging
            return {"error": "Unexpected OpenAI output format", "raw": data}

        # Ensure we have pure JSON
        text = text.strip()
        # sometimes the model might add ```json fences; strip them
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        try:
            parsed = json.loads(text)
        except Exception as e:
            return {
                "error": f"Failed to parse JSON from model: {e}",
                "raw_text": text,
            }

        return parsed

    except requests.HTTPError as e:
        print("❌ OpenAI HTTP error:", e, getattr(e, "response", None).text if hasattr(e, "response") else "")
        return {"error": f"OpenAI HTTP error: {str(e)}"}
    except Exception as e:
        print("❌ OpenAI general error:", e)
        return {"error": f"OpenAI error: {str(e)}"}


# ----------------------------
# PROCESS ONE REPORT
# ----------------------------

def fetch_next_report():
    """Get the next report with ai_status='queued' and mark as 'processing'."""
    res = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not res.data:
        return None

    report = res.data[0]

    # mark as processing to avoid double work
    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report["id"]).execute()

    return report


def process_report(report: dict):
    report_id = report["id"]
    file_path = report.get("file_path")

    print(f"📄 Processing report: {report_id}")

    if not file_path:
        msg = "Report has no file_path."
        print("❌", msg)
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_error": msg}
        ).eq("id", report_id).execute()
        return

    # 1) Download PDF from Supabase storage
    try:
        print("⬇️  Downloading PDF from Supabase storage...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        msg = f"Could not download PDF: {e}"
        print("❌", msg)
        supabase.table("reports").update(
            {"ai_status": "failed", "ai_error": msg}
        ).eq("id", report_id).execute()
        return

    # 2) Extract text
    print("📑 Extracting text from PDF...")
    extracted_text = extract_text_from_pdf(pdf_bytes)

    if not extracted_text or len(extracted_text.strip()) < 40:
        msg = "PDF contains no readable text (likely a scanned image only)."
        print("⚠️", msg)
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_error": msg,
                "extracted_text": extracted_text or "",
            }
        ).eq("id", report_id).execute()
        return

    # 3) Call AI
    print("🤖 Calling AMI AI (OpenAI)...")
    ai_json = call_ami_ai(extracted_text)

    if "error" in ai_json:
        print("❌ AI failed:", ai_json["error"])
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_error": ai_json["error"],
                "ai_results": ai_json,
                "extracted_text": extracted_text,
            }
        ).eq("id", report_id).execute()
        return

    # 4) Save back to Supabase
    print("💾 Saving AI results to database...")

    update_payload = {
        "ai_status": "completed",
        "ai_results": ai_json,
        "extracted_text": extracted_text,
        # optional structured fields for your frontend
        "cbc_json": ai_json.get("cbc_values", {}),
        "trend_json": ai_json.get("trend_summary", []),
    }

    supabase.table("reports").update(update_payload).eq("id", report_id).execute()

    print(f"✅ Report {report_id} completed.")


# ----------------------------
# MAIN WORKER LOOP
# ----------------------------

print("🚀 AMI Worker started (text-only PDFs, gpt-4o-mini).")

while True:
    try:
        report = fetch_next_report()
        if report is None:
            # no work to do
            time.sleep(5)
            continue

        process_report(report)

    except Exception as e:
        print("❌ Worker crash prevented:", e)

    # small delay to avoid hammering
    time.sleep(2)
