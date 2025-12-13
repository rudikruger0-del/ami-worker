print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

import os, time, json, io, traceback, base64, re
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes

# -----------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

# -----------------------------------------------------
# PDF TEXT EXTRACTION
# -----------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n\n".join([(page.extract_text() or "") for page in reader.pages]).strip()
    except:
        return ""

def is_scanned_pdf(text: str) -> bool:
    return len(text.strip()) < 30

# -----------------------------------------------------
# CLEAN NUMBER
# -----------------------------------------------------
def clean_number(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[0]) if nums else None

# -----------------------------------------------------
# OCR — FIXED FOR OPENAI v2024+
# -----------------------------------------------------
def extract_cbc_from_image(image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode()

    system_prompt = (
        "You are an OCR assistant for lab reports. Extract ALL analytes you can see: "
        "CBC, diff, electrolytes, CRP, urea, creatinine, liver enzymes, CK. "
        "Return STRICT JSON:\n"
        "{ 'cbc': [ { 'analyte':'', 'value':'', 'units':'', 'reference_low':'', 'reference_high':'' } ] }"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        }
                    ]
                }
            ]
        )
        return resp.choices[0].message.parsed
    except Exception as e:
        print("OCR error:", e)
        return {"cbc": []}

# -----------------------------------------------------
# INTERPRETATION
# -----------------------------------------------------
def call_ai_on_report(text: str):
    system_prompt = (
        "You are AMI, a medical lab interpreter. "
        "Return STRICT JSON containing: patient, cbc, summary (impression + follow-up). "
        "Never diagnose, only describe patterns."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
    )
    return resp.choices[0].message.parsed

# -----------------------------------------------------
# CBC DICTIONARY NORMALISATION
# -----------------------------------------------------
def build_cbc_dict(ai_json):
    rows = ai_json.get("cbc") or []
    out = {}

    for r in rows:
        name = (r.get("analyte") or "").lower()

        if "haemo" in name or name == "hb": out["Hb"] = r
        if name.startswith("mcv"): out["MCV"] = r
        if name.startswith("mch"): out["MCH"] = r
        if "white" in name or "wbc" in name: out["WBC"] = r
        if "neut" in name: out["Neut"] = r
        if "lymph" in name: out["Lymph"] = r
        if "plate" in name: out["Plt"] = r
        if "creatinine" in name: out["Cr"] = r
        if "urea" in name: out["Urea"] = r
        if "crp" in name: out["CRP"] = r
        if "sodium" in name or name == "na": out["Na"] = r
        if "potassium" in name or name == "k": out["K"] = r

    return out

# -----------------------------------------------------
# ROUTE ENGINE V3 — Patterns → Route → Suggested Next Steps
# -----------------------------------------------------
def generate_routes(c):
    v = lambda k: clean_number(c.get(k, {}).get("value"))
    Hb, MCV, WBC, Neut, Lymph, Plt, Cr, Urea, CRP = (
        v("Hb"), v("MCV"), v("WBC"), v("Neut"), v("Lymph"), v("Plt"),
        v("Cr"), v("Urea"), v("CRP")
    )

    routes = []

    # --- ANAEMIA PATTERN ---
    if Hb is not None and Hb < 13:
        if MCV and MCV < 80:
            routes.append({
                "pattern": "Microcytic anaemia",
                "route": "Likely iron deficiency / chronic disease pattern",
                "next_steps": [
                    "Request ferritin & iron studies",
                    "Check reticulocyte count",
                    "Assess for chronic inflammation"
                ]
            })
        elif MCV and MCV > 100:
            routes.append({
                "pattern": "Macrocytic anaemia",
                "route": "Possible B12/folate deficiency / liver pattern",
                "next_steps": [
                    "Order B12 & folate",
                    "Review liver enzymes",
                    "Check medications (e.g., antiretrovirals)"
                ]
            })
        else:
            routes.append({
                "pattern": "Normocytic anaemia",
                "route": "Common in renal disease / early iron deficiency",
                "next_steps": [
                    "Check creatinine & eGFR",
                    "Order reticulocyte count",
                    "Review chronic inflammatory markers"
                ]
            })

    # --- INFECTION / INFLAMMATION ---
    if WBC and WBC > 12:
        if Neut and Neut > 70:
            routes.append({
                "pattern": "Neutrophilia",
                "route": "Bacterial infection physiology",
                "next_steps": [
                    "Correlate with fever, CRP",
                    "Look for localised infection",
                    "Consider sepsis markers if unwell"
                ]
            })

        if Lymph and Lymph > 45:
            routes.append({
                "pattern": "Lymphocytosis",
                "route": "Viral or recovery-phase pattern",
                "next_steps": [
                    "Check recent viral symptoms",
                    "Review past CBC for trends",
                    "Repeat CBC in 1–2 weeks"
                ]
            })

    if CRP and CRP > 10:
        routes.append({
            "pattern": "Raised CRP",
            "route": "Inflammation / infection",
            "next_steps": [
                "Correlate with WBC",
                "Assess clinical severity",
                "Consider imaging if focal symptoms exist"
            ]
        })

    # --- RENAL FUNCTION ---
    if Cr and Cr > 120:
        routes.append({
            "pattern": "Renal impairment physiology",
            "route": "Possible dehydration / CKD / medication effect",
            "next_steps": [
                "Repeat U&E",
                "Check hydration & meds",
                "Order eGFR"
            ]
        })

    # Default
    if not routes:
        routes.append({
            "pattern": "No major abnormalities",
            "route": "All key markers within expected physiology",
            "next_steps": ["Monitor and correlate clinically"]
        })

    return routes

# -----------------------------------------------------
# PROCESS REPORT
# -----------------------------------------------------
def process_report(job):
    rid = job["id"]
    file_path = job.get("file_path")
    print(f"Processing {rid}...")

    pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_bytes, "data"):
        pdf_bytes = pdf_bytes.data

    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)

    extracted_rows = []

    # --- SCANNED PDF → OCR ---
    if scanned:
        print("SCANNED PDF → OCR START")
        pages = convert_from_bytes(pdf_bytes)

        for i, img in enumerate(pages, 1):
            print(f"OCR page {i}...")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result = extract_cbc_from_image(buf.getvalue())
            rows = result.get("cbc", [])
            if rows:
                extracted_rows.extend(rows)

        if not extracted_rows:
            raise ValueError("No CBC extracted from scanned PDF")

        merged_text = json.dumps({"cbc": extracted_rows})

    else:
        merged_text = text or job.get("l_text") or ""

    # --- AI INTERPRETATION ---
    ai_json = call_ai_on_report(merged_text)

    # --- ROUTE ENGINE v3 ---
    cdict = build_cbc_dict(ai_json)
    ai_json["routes"] = generate_routes(cdict)

    supabase.table("reports").update(
        {"ai_status": "completed", "ai_results": ai_json, "ai_error": None}
    ).eq("id", rid).execute()

    print(f"✓ Completed {rid}")

# -----------------------------------------------------
# MAIN LOOP — FIXED
# -----------------------------------------------------
def main():
    print("Entering main loop...")
    while True:
        try:
            res = (
                supabase.table("reports")
                .select("*")
                .eq("ai_status", "pending")
                .limit(1)
                .execute()
            )

            jobs = res.model or []   # ← FIXED

            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            supabase.table("reports").update(
                {"ai_status": "processing"}
            ).eq("id", job["id"]).execute()

            process_report(job)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
