print(">>> AMI Worker starting... imports loading")

# -----------------------
# BASE IMPORTS
# -----------------------
import os, time, json, io, traceback, base64

print(">>> Base imports OK")

# -----------------------
# SUPABASE
# -----------------------
from supabase import create_client, Client
print(">>> Supabase imported")

# -----------------------
# OPENAI
# -----------------------
from openai import OpenAI
print(">>> OpenAI imported")

# -----------------------
# PDF PARSING
# -----------------------
from pypdf import PdfReader
print(">>> pypdf imported")

# -----------------------
# IMAGE CONVERSION (OCR)
# -----------------------
from pdf2image import convert_from_bytes
print(">>> pdf2image imported")

# ======================================================
# ENV + CLIENTS
# ======================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(">>> Environment variables loaded")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase env variables!")

client = OpenAI(api_key=OPENAI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET = "reports"


# ======================================================
# PDF TEXT EXTRACTION
# ======================================================
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            t = page.extract_text() or ""
            pages.append(t)
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("PDF parse error:", e)
        return ""


def is_scanned_pdf(text: str) -> bool:
    """Heuristic to detect scanned PDF."""
    if not text:
        return True
    if len(text.strip()) < 30:
        return True
    return False


# ======================================================
#   OCR (VISION) — Extract CBC from an image
# ======================================================
def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """Send PNG image to OpenAI Vision OCR and extract CBC values."""

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant for medical laboratory scans. "
        "Extract ALL CBC and chemistry analytes (WBC, neutrophils, lymphocytes, "
        "Hb, Hct, RBC, MCV, MCH, MCHC, RDW, platelets, CRP, urea, creatinine, "
        "electrolytes, liver enzymes, CK, CK-MB). "
        "Return STRICT JSON ONLY → { 'cbc': [ ... ] }"
    )

    # IMPORTANT: proper Vision format (NO 'input_image')
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            { "role": "system", "content": system_prompt },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{b64}"
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ======================================================
#   MAIN INTERPRETATION — CBC + ROUTES
# ======================================================
def call_ai_on_report(text: str) -> dict:
    """Interpret CBC using the full routes system."""

    MAX = 12000
    if len(text) > MAX:
        text = text[:MAX]

    system_prompt = (
        "You are AMI — an assistive tool interpreting CBC results for South African clinicians. "
        "SAFETY: never diagnose, never prescribe, use cautious language.\n\n"

        "YOU MUST OUTPUT STRICT JSON with fields:\n"
        "{\n"
        "  'patient': { 'name':..., 'age':..., 'sex':... },\n"
        "  'cbc': [ { 'analyte':..., 'value':..., 'units':..., "
        "            'reference_low':..., 'reference_high':..., 'flag':... } ],\n"
        "  'summary': {\n"
        "       'overall_severity': 'mild'|'moderate'|'high'|'critical',\n"
        "       'impression': string,\n"
        "       'suggested_follow_up': string\n"
        "  },\n"
        "  'routes': [\n"
        "       {\n"
        "         'id': string,\n"
        "         'name': string,\n"
        "         'trigger': string,\n"
        "         'pattern_summary': string,\n"
        "         'key_values': [string],\n"
        "         'suggested_tests': [string],\n"
        "         'priority': 'low'|'medium'|'high'\n"
        "       }\n"
        "  ]\n"
        "}\n\n"

        "ROUTES YOU MUST CONSIDER:\n"
        "- Microcytic anaemia route: Hb low + MCV low ± MCH low → ferritin, iron studies, reticulocyte count.\n"
        "- Normocytic anaemia route.\n"
        "- Macrocytic anaemia route.\n"
        "- High RDW route.\n"
        "- Low MCHC route.\n"
        "- Infection / bacterial inflammatory route: high neutrophils ± high CRP.\n"
        "- Viral pattern route.\n"
        "- Sepsis / severe inflammatory stress route.\n"
        "- Thrombocytosis route.\n"
        "- Thrombocytopenia route.\n"
        "- Eosinophilia route.\n"
        "- Monocytosis route.\n"
        "- Basophilia route.\n"
        "- Kidney function route.\n"
        "- Liver enzyme route.\n"
        "- Electrolyte imbalance routes.\n"
        "- NLR route.\n"
        "- Reticulocyte routes.\n\n"

        "Return JSON ONLY."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": text }
        ],
        response_format={"type": "json_object"},
        temperature=0.1
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ======================================================
# PROCESS A SINGLE REPORT
# ======================================================
def process_report(job: dict):
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path:
            err = f"Missing file_path for report {report_id}"
            supabase.table("reports").update({
                "ai_status": "failed", "ai_error": err
            }).eq("id", report_id).execute()
            return

        # Download PDF
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        # Extract text
        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)

        # ------------------------------------------------
        # SCANNED PDF → OCR
        # ------------------------------------------------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED → OCR")

            images = convert_from_bytes(pdf_bytes)
            combined = []

            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                try:
                    result = extract_cbc_from_image(img_bytes)
                    if "cbc" in result:
                        combined.extend(result["cbc"])
                except Exception as e:
                    print("OCR error:", e)

            if not combined:
                raise ValueError("Vision OCR could not extract CBC values")

            merged_text = json.dumps({ "cbc": combined })

        # ------------------------------------------------
        # TEXT PDF → parse normally
        # ------------------------------------------------
        else:
            print(f"📝 Report {report_id}: DIGITAL PDF → text parser")
            merged_text = text or l_text

        if not merged_text.strip():
            raise ValueError("No usable content extracted")

        # Final interpretation
        ai_json = call_ai_on_report(merged_text)

        # Store result
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": ai_json,
            "ai_error": None
        }).eq("id", report_id).execute()

        print(f"✅ Report {report_id} processed successfully")

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"❌ Error processing report {report_id}: {err}")
        traceback.print_exc()

        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": err
        }).eq("id", report_id).execute()


# ======================================================
# MAIN LOOP
# ======================================================
def main():
    print(">>> Entering worker loop…")
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
                print(f"🔎 Found job {job_id}")

                supabase.table("reports").update({
                    "ai_status": "processing"
                }).eq("id", job_id).execute()

                process_report(job)

            else:
                print("No jobs...")
            time.sleep(1)

        except Exception as e:
            print("Worker loop error:", e)
            traceback.print_exc()
            time.sleep(5)


if __name__ == "__main__":
    main()
