import time
import json
import os
import io

from supabase import create_client, Client
from openai import OpenAI
from PyPDF2 import PdfReader

# ----------------------------
# ENVIRONMENT
# ----------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY / OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ----------------------------
# PDF → TEXT (TEXT-ONLY, NO OCR)
# ----------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a digital (non-scanned) PDF using PyPDF2.
    If the PDF is a scanned image with no embedded text, this will return "".
    """
    try:
        pdf_stream = io.BytesIO(pdf_bytes)
        reader = PdfReader(pdf_stream)

        text_chunks = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_chunks.append(page_text.strip())

        full_text = "\n\n".join(text_chunks).strip()

        # Hard cap on characters to keep tokens under control
        MAX_CHARS = 8000  # ≈ a few thousand tokens, safe for gpt-4o + your limits
        if len(full_text) > MAX_CHARS:
            full_text = full_text[:MAX_CHARS] + "\n\n[Truncated for length – remaining text omitted.]"

        return full_text
    except Exception as e:
        print("❌ PDF text extraction error:", e)
        return ""


# ----------------------------
# AI CALL (GPT-4o, DETAILED REPORT)
# ----------------------------
def call_ami_ai(extracted_text: str) -> dict:
    """
    Send extracted lab text to gpt-4o and get a rich JSON interpretation.
    No PDF base64, to keep tokens lower and avoid rate limit errors.
    """

    if not extracted_text or len(extracted_text.strip()) < 50:
        # Not enough usable text
        return {
            "error": "No usable lab text could be extracted from this PDF. "
                     "It may be a scanned image or non-standard format."
        }

    system_message = """
You are AMI — an advanced clinical laboratory interpretation AI.
You analyse blood and chemistry tests for doctors.

Data you may see:
- CBC: WBC, RBC, HGB, HCT, MCV, MCH, MCHC, PLT, RDW, NEU%, LYM%, MONO%, EOS%, BASO%, abs counts, etc.
- Basic chemistry: Na, K, Cl, HCO3, urea, creatinine, eGFR, glucose, liver enzymes, etc.

CRITICAL RULES:
1. Base your reasoning ONLY on values and comments found in the provided text.
2. NEVER invent numbers or results that are not explicitly present.
3. If you are unsure, say you are unsure. Do NOT guess diagnoses.
4. If infection/inflammation markers (WBC, NEU, CRP, ESR, etc.) are missing, clearly state that limitation.
5. If you detect NO abnormal values, still give a clinically useful summary and reassurance.
6. Keep language suitable for DOCTORS, but understandable for patients if they read it.
7. Be detailed: aim for at least 3–6 bullet points where possible.
8. Output STRICT JSON ONLY — no markdown, no prose outside JSON.

JSON SCHEMA (use exactly these keys):

{
  "summary": [               // high-level overview, bullet points (strings)
  ],
  "trend_summary": [         // if prior values in text, comment on trends; else explain trend data not available
  ],
  "flagged_results": [       // each abnormal or clinically notable result
    {
      "name": "WBC",
      "value": "13.2 x10^9/L",
      "reference_range": "4.0 – 11.0 x10^9/L",
      "direction": "high or low",
      "severity": "mild | moderate | severe",
      "clinical_meaning": "Short explanation for doctor."
    }
  ],
  "interpretation": [        // deeper narrative: what the pattern suggests, differential thoughts, caveats
  ],
  "risk_level": "low | moderate | high | unknown",
  "recommendations": [       // practical next-steps: repeat labs, clinical correlation, lifestyle, further tests
  ],
  "when_to_seek_urgent_care": [  // red-flag symptoms or lab patterns that should trigger urgent review. If none, say why.
  ],
  "cbc_values": {            // key = short test name, value = object with parsed info when available
    "WBC": {
      "value": "13.2",
      "unit": "x10^9/L",
      "direction": "high | low | normal | unknown",
      "flagged": true
    }
  },
  "chemistry_values": {      // similar structure for chemistry / metabolic tests if present
  },
  "disclaimer": "This is not medical advice. Always interpret results in the context of the full clinical picture."
}

IMPORTANT BEHAVIOUR:
- If you cannot find ANY lab values, set arrays to [] and risk_level to "unknown",
  and in `interpretation` clearly state that no numeric results were available.
- Prefer concise bullet points over long paragraphs.
"""

    user_message = f"""
Lab report text (exact extraction from PDF):

\"\"\"{extracted_text}\"\"\"

Using this text only, build a detailed JSON report following the schema.
Do NOT add any keys that are not in the schema.
Output VALID JSON ONLY.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",              # ✅ full 4o, better medical reasoning
            max_output_tokens=1800,      # enough for rich JSON but still safe
            response_format={"type": "json"},
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
        )

        # openai==1.x: JSON is already parsed into .json
        ai_json = response.output[0].content[0].json
        return ai_json

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}


# ----------------------------
# PROCESS ONE REPORT
# ----------------------------
def process_next_report():
    print("🔍 Checking for queued reports...")

    # Get the oldest queued report
    resp = (
        supabase.table("reports")
        .select("*")
        .eq("ai_status", "queued")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )

    if not resp.data:
        print("ℹ️ No queued reports.")
        return

    report = resp.data[0]
    report_id = report["id"]
    file_path = report.get("file_path")

    print(f"📄 Found queued report: {report_id} ({file_path})")

    # Mark as processing to avoid duplicate work
    supabase.table("reports").update({"ai_status": "processing"}).eq("id", report_id).execute()

    # ----------------------------
    # Download PDF from Supabase storage
    # ----------------------------
    try:
        print("⬇️ Downloading PDF from Supabase storage...")
        pdf_bytes = supabase.storage.from_("reports").download(file_path)
    except Exception as e:
        print("❌ Could not download PDF:", e)
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": {"error": f"Could not download PDF: {str(e)}"},
            }
        ).eq("id", report_id).execute()
        return

    # ----------------------------
    # Extract text (digital PDFs only)
    # ----------------------------
    extracted_text = extract_text_from_pdf(pdf_bytes)

    if not extracted_text:
        msg = (
            "No readable text could be extracted from this PDF. "
            "It may be a scanned image or photo. Please upload a digital PDF."
        )
        print("⚠️", msg)
        supabase.table("reports").update(
            {
                "ai_status": "failed",
                "ai_results": {"error": msg},
                "extracted_text": "",
            }
        ).eq("id", report_id).execute()
        return

    # ----------------------------
    # Call AI
    # ----------------------------
    print("🤖 Calling AMI (gpt-4o) for interpretation...")
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

    # ----------------------------
    # Save back to Supabase
    # ----------------------------
    print("💾 Saving AI results to database...")

    supabase.table("reports").update(
        {
            "ai_status": "completed",
            "ai_results": ai_json,
            "extracted_text": extracted_text,
            # convenience fields for your UI:
            "cbc_json": ai_json.get("cbc_values", {}),
            "trend_json": ai_json.get("trend_summary", []),
        }
    ).eq("id", report_id).execute()

    print("✅ Report completed:", report_id)


# ----------------------------
# MAIN LOOP
# ----------------------------
if __name__ == "__main__":
    print("🚀 AMI Worker started (text-only, gpt-4o)…")

    while True:
        try:
            process_next_report()
        except Exception as e:
            print("💥 Worker loop error (caught, will continue):", e)

        # Poll every few seconds
        time.sleep(5)
