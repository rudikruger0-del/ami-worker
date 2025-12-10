print(">>> AMI Worker starting…")

import os, time, json, io, traceback, base64, re
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes

# ----------------------------------------------------
# ENV
# ----------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)
BUCKET = "reports"

# ----------------------------------------------------
# PDF TEXT EXTRACTOR
# ----------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages).strip()
    except:
        return ""

def is_scanned_pdf(text: str) -> bool:
    return len(text.strip()) < 30

# ----------------------------------------------------
# CLEAN NUMBER
# ----------------------------------------------------
def clean_number(val):
    if val is None: return None
    if isinstance(val, (int, float)): return float(val)
    s = str(val).replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    return float(nums[0]) if nums else None

# ----------------------------------------------------
# OCR FIXED — SAFE MULTI-PAGE READING
# ----------------------------------------------------
def extract_cbc_from_image(image_bytes: bytes):
    b64 = base64.b64encode(image_bytes).decode()

    system_prompt = (
        "You are an OCR assistant. Extract ANY lab analytes you see — CBC, diff, "
        "electrolytes, CRP, urea, creatinine, liver enzymes, CK, CK-MB.\n"
        "Return STRICT JSON:\n"
        "{ 'cbc': [ { 'analyte':'', 'value':'', 'units':'', 'reference_low':'', 'reference_high':'' } ] }"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        }
                    ]
                }
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        data = response.choices[0].message.parsed
        return data if isinstance(data, dict) else {"cbc": []}

    except Exception as e:
        print("OCR PAGE ERROR:", e)
        return {"cbc": []}   # ← Never break, return empty

# ----------------------------------------------------
# INTERPRETATION (unchanged)
# ----------------------------------------------------
def call_ai_on_report(text: str) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return strict JSON: patient, cbc, summary."},
            {"role": "user", "content": text}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    return resp.choices[0].message.parsed

# ----------------------------------------------------
# ROUTE ENGINE v2
# ----------------------------------------------------
def build_cbc_dict(ai_json):
    out = {}
    rows = ai_json.get("cbc") or []
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

def generate_routes(c):
    v = lambda k: clean_number(c.get(k, {}).get("value"))
    Hb, MCV, WBC, Neut, Lymph, Plt, Cr, Urea, CRP = (
        v("Hb"), v("MCV"), v("WBC"), v("Neut"), v("Lymph"), v("Plt"), v("Cr"), v("Urea"), v("CRP")
    )
    routes = []

    # Anaemia
    if Hb and Hb < 13:
        if MCV and MCV < 80:
            routes.append("Microcytic pattern → ferritin, iron studies, reticulocytes.")
        elif MCV and MCV > 100:
            routes.append("Macrocytic pattern → B12, folate, liver function tests.")
        else:
            routes.append("Normocytic anaemia → chronic disease, renal, early iron deficiency.")

    # Infection / inflammation
    if WBC and WBC > 12:
        if Neut and Neut > 70:
            routes.append("Neutrophilia → bacterial infection pattern.")
        if Lymph and Lymph > 45:
            routes.append("Lymphocytosis → viral / recovery pattern.")

    if CRP and CRP > 10:
        routes.append("CRP elevated → active inflammation/infection route.")

    # Renal
    if Cr and Cr > 120:
        routes.append("Creatinine high → renal assessment route (hydration, meds, U&E).")

    if not routes:
        routes.append("No major abnormalities detected.")

    return routes

# ----------------------------------------------------
# PROCESS REPORT
# ----------------------------------------------------
def process_report(job):
    rid = job["id"]
    file_path = job.get("file_path")
    ltext = job.get("l_text") or ""

    print(f"Processing {rid}…")

    pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_bytes, "data"): pdf_bytes = pdf_bytes.data

    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)

    all_rows = []

    # SCANNED PDF → OCR
    if scanned:
        print("SCANNED PDF → Starting OCR…")
        images = convert_from_bytes(pdf_bytes)

        for i, img in enumerate(images, 1):
            print(f"OCR on page {i}…")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            result = extract_cbc_from_image(buf.getvalue())

            rows = result.get("cbc", [])
            if rows:
                print(f" → Extracted {len(rows)} rows on page {i}")
                all_rows.extend(rows)
            else:
                print(f" → No CBC rows on page {i}, skipping…")

    else:
        all_rows = text

    if not all_rows:
        raise ValueError("No CBC extracted from any page.")

    # INTERPRETATION
    if isinstance(all_rows, list): 
        merged_text = json.dumps({"cbc": all_rows})
    else:
        merged_text = all_rows

    ai_json = call_ai_on_report(merged_text)

    # ROUTES
    cdict = build_cbc_dict(ai_json)
    ai_json["routes"] = generate_routes(cdict)

    supabase.table("reports").update(
        {"ai_status": "completed", "ai_results": ai_json, "ai_error": None}
    ).eq("id", rid).execute()

    print(f"✓ Completed {rid}")
    return ai_json

# ----------------------------------------------------
# LOOP
# ----------------------------------------------------
def main():
    print("AMI Worker ready.")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status", "pending").limit(1).execute()
            jobs = res.data or []
            if not jobs:
                print("No jobs…")
                time.sleep(1)
                continue

            job = jobs[0]
            supabase.table("reports").update({"ai_status":"processing"}).eq("id", job["id"]).execute()
            process_report(job)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
