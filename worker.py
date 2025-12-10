print(">>> Worker starting, loading imports...")

# -----------------------
# BASE PYTHON IMPORTS
# -----------------------
try:
    import os
    import time
    import json
    import io
    import traceback
    import base64
    print(">>> Base imports loaded")
except Exception as e:
    print("❌ Failed loading base imports:", e)
    raise e

# -----------------------
# SUPABASE IMPORT
# -----------------------
try:
    from supabase import create_client, Client
    print(">>> Supabase imported")
except Exception as e:
    print("❌ Failed importing Supabase:", e)
    raise e

# -----------------------
# OPENAI IMPORT
# -----------------------
try:
    from openai import OpenAI
    print(">>> OpenAI imported")
except Exception as e:
    print("❌ Failed importing OpenAI:", e)
    raise e

# -----------------------
# PYPDF IMPORT
# -----------------------
try:
    from pypdf import PdfReader
    print(">>> pypdf imported")
except Exception as e:
    print("❌ Failed importing pypdf:", e)
    raise e

# -----------------------
# PDF2IMAGE IMPORT
# -----------------------
try:
    from pdf2image import convert_from_bytes
    print(">>> pdf2image imported")
except Exception as e:
    print("❌ Failed importing pdf2image:", e)
    raise e


# ======================================================
#   ENV & CLIENTS
# ======================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(">>> Environment variables loaded")
print("    SUPABASE_URL set? ", bool(SUPABASE_URL))
print("    SUPABASE_SERVICE_KEY set? ", bool(SUPABASE_KEY))
print("    OPENAI_API_KEY set? ", bool(OPENAI_API_KEY))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# ======================================================
#   HELPERS – PDF TEXT
# ======================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable text PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        text = "\n\n".join(pages).strip()
        print(f">>> PDF text length: {len(text)} chars")
        return text
    except Exception as e:
        print("PDF parse error:", e)
        return ""


def is_scanned_pdf(pdf_text: str) -> bool:
    """
    Heuristic: if there is almost no text, treat as scanned image PDF.
    """
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False


# ======================================================
#   HELPERS – IMAGE OCR → CBC
# ======================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Sends the scanned image to OpenAI Vision to extract CBC values.
    Returns a dict like: { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """

    print(">>> Running Vision OCR on one page image…")
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans. "
        "Your ONLY task is to read the image and extract CBC and related analytes "
        "(including chemistry, liver enzymes, CK, CK-MB, creatinine, electrolytes).\n\n"
        "Return STRICT JSON with this exact structure:\n"
        "{\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": string,\n"
        "      \"value\": number | null,\n"
        "      \"units\": string | null,\n"
        "      \"reference_low\": number | null,\n"
        "      \"reference_high\": number | null\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "- Use null for any numeric field you cannot read.\n"
        "- Do NOT add any extra top-level keys.\n"
        "- Output ONLY JSON, no explanations."
    )

    response = client.chat.completions.create(
        model="gpt-4o",   # vision-capable model
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
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

    print(">>> Vision raw JSON length:", len(raw))
    if not raw.strip():
        raise ValueError("Vision OCR returned empty content")

    return json.loads(raw)


# ======================================================
#   CLINICAL INTERPRETATION ENGINE
# ======================================================

def call_ai_on_report(text: str) -> dict:
    """
    Clinical reasoning layer (Option A – strong suggestions).

    Input:
      - Either plain report text
      - OR JSON string containing { "cbc": [...] }

    Output JSON:
    {
      "patient": { "name": string|null, "age": number|null, "sex": "Male"|"Female"|"Unknown" },
      "cbc": [ { analyte, value, units, reference_low, reference_high, flag } ],
      "summary": {
        "impression": string,
        "suggested_follow_up": string
      }
    }
    """
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are AMI, an assistive clinical tool that analyses Complete Blood Count (CBC) "
        "and related laboratory results (chemistry, CK, CK-MB, liver enzymes, creatinine, "
        "electrolytes, CRP, etc.).\n\n"

        "You MUST NOT make a formal diagnosis or prescribe treatment. "
        "You ONLY describe laboratory abnormalities, patterns and safe, general follow-up steps.\n\n"

        "The user input can be either:\n"
        "1) Plain text from a lab report; or\n"
        "2) A JSON string containing a top-level key \"cbc\" with a list of analytes.\n"
        "If JSON is provided, use it as the authoritative list of CBC/chemistry values.\n\n"

        "You are writing for South African clinicians (GPs, paediatricians, physicians). "
        "Be concise but clinically meaningful.\n\n"

        "--------------------------------------\n"
        "SEVERITY CLASSIFICATION\n"
        "--------------------------------------\n"
        "- Mild: values only slightly outside reference range.\n"
        "- Moderate: clearly outside reference but not critical.\n"
        "- Critical: clearly dangerous ranges, e.g. (examples only):\n"
        "  • Platelets < 80 x10^9/L (esp. < 50) – bleeding risk.\n"
        "  • Hb < ~8 g/dL – significant anaemia.\n"
        "  • Creatinine well above upper reference – possible renal stress/AKI.\n"
        "  • Potassium < 3.3 or > 5.5 mmol/L – arrhythmia risk.\n"
        "  • ALT or AST > ~3x upper limit – marked hepatocellular injury.\n"
        "  • CK > ~5,000 U/L – strong concern for acute muscle injury pattern.\n\n"

        "--------------------------------------\n"
        "OUTPUT JSON STRUCTURE (STRICT)\n"
        "--------------------------------------\n"
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
        "    \"suggested_follow_up\": string\n"
        "  }\n"
        "}\n\n"

        "--------------------------------------\n"
        "FILLING RULES\n"
        "--------------------------------------\n"
        "- Extract patient name/age/sex from the text if present; otherwise use null/\"Unknown\".\n"
        "- If the input is JSON with \"cbc\", map that list directly into the \"cbc\" array, "
        "adding appropriate flags and reference ranges when possible.\n"
        "- If the input is plain text, parse CBC/chemistry values yourself.\n"
        "- Use reference ranges from the text when available; otherwise, use reasonable adult "
        "ranges. If unsure, leave reference_low/reference_high as null.\n\n"

        "- In summary.impression:\n"
        "  • First sentence MUST start with exactly one of:\n"
        "    \"Overall severity: mild.\", \"Overall severity: moderate.\", "
        "or \"Overall severity: critical.\".\n"
        "  • Then describe key patterns (e.g. microcytic anaemia pattern, thrombocytopenia, "
        "leukocytosis suggestive of infection, renal stress, liver involvement, "
        "electrolyte disturbance, acute muscle injury pattern, etc.).\n"
        "  • Include possible causes ONLY as patterns, e.g. "
        "\"pattern could be in keeping with iron deficiency\", "
        "\"pattern may suggest infection\", "
        "\"pattern may be compatible with acute liver injury\". "
        "NEVER state a confirmed diagnosis.\n\n"

        "- In summary.suggested_follow_up:\n"
        "  • Provide practical, clinically useful next steps.\n"
        "  • ALWAYS include specific recommended additional tests when appropriate "
        "(e.g. ferritin, iron studies, reticulocyte count, CRP/ESR, liver ultrasound, "
        "renal function panel, repeat CBC, peripheral smear, viral serology, etc.).\n"
        "  • You MAY format this as short bullet-style lines separated by newlines.\n"
        "  • Include both investigations (tests) and general management direction like "
        "\"correlate clinically\", \"assess for bleeding\", \"monitor trend\".\n"
        "  • For critical patterns, emphasise urgency and recommend urgent clinical review "
        "or emergency assessment (without specifying treatment).\n\n"

        "Output ONLY valid JSON with the exact top-level keys: patient, cbc, summary. "
        "No markdown, no extra commentary."
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

    content = response.choices[0].message.content
    if isinstance(content, list):
        raw = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    else:
        raw = content or ""

    print(">>> Interpretation JSON length:", len(raw))
    if not raw.strip():
        raise ValueError("AI returned empty interpretation")

    return json.loads(raw)


# ======================================================
#   REPORT PROCESSOR
# ======================================================

def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        print(f">>> Processing report {report_id}")
        print("    file_path:", file_path)

        if not file_path or str(file_path).strip() == "":
            err = f"Missing file_path for report {report_id}"
            print("⚠️", err)
            supabase.table("reports").update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
            return {"error": err}

        # Download PDF bytes
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        print(">>> PDF bytes length:", len(pdf_bytes))

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        print(">>> is_scanned_pdf =", scanned)

        # merged_text will be what we send into the clinical engine
        merged_text = ""
        cbc_json = None

        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → Using OCR")

            images = convert_from_bytes(pdf_bytes)
            print(f">>> PDF rendered into {len(images)} page images")

            combined_cbc = []

            for idx, img in enumerate(images, start=1):
                print(f">>> OCR on page {idx}")
                img_bytes_io = io.BytesIO()
                img.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                try:
                    result = extract_cbc_from_image(img_bytes)
                    if isinstance(result, dict) and "cbc" in result and isinstance(result["cbc"], list):
                        combined_cbc.extend(result["cbc"])
                    else:
                        print("Vision OCR returned unexpected structure:", result)
                except Exception as e:
                    print("Vision OCR error:", e)

            if not combined_cbc:
                raise ValueError("Vision OCR failed to extract CBC values")

            cbc_json = {"cbc": combined_cbc}
            merged_text = json.dumps(cbc_json)
            print(">>> Combined OCR CBC entries:", len(combined_cbc))

        else:
            print(f"📝 Report {report_id}: Digital PDF detected → Using text parser")
            # Combine any manually stored text with extracted text
            if l_text and text:
                merged_text = (l_text + "\n\n" + text).strip()
            else:
                merged_text = (text or l_text or "").strip()

        if not merged_text.strip():
            raise ValueError("No usable content extracted for AI processing")

        # Final interpretation step – clinical engine
        ai_json = call_ai_on_report(merged_text)

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
            {"ai_status": "failed", "ai_error": err}
        ).eq("id", report_id).execute()

        return {"error": err}


# ======================================================
#   WORKER LOOP
# ======================================================

def main():
    print(">>> Entering main worker loop...")
    print("AMI Worker with Vision OCR + Clinical Engine started… watching for jobs.")

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

                supabase.table("reports").update(
                    {"ai_status": "processing"}
                ).eq("id", job_id).execute()

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
