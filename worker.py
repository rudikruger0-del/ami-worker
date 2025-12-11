print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps")

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

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing Supabase credentials")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

# ----------------------------------------------------
# PDF TEXT EXTRACTOR
# ----------------------------------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n\n".join(pages).strip()
    except:
        return ""

def is_scanned_pdf(text: str) -> bool:
    """Very low text = scanned PDF."""
    return len(text.strip()) < 30

# ----------------------------------------------------
# CLEAN NUMBERS
# ----------------------------------------------------
def clean_number(val):
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except:
        return None

# ----------------------------------------------------
# OCR FIXED — MULTI-PAGE
# ----------------------------------------------------
def extract_cbc_from_image(image_bytes: bytes):
    """Safe OCR wrapper — never crashes worker."""
    try:
        b64 = base64.b64encode(image_bytes).decode()
    except:
        return {"cbc": []}

    system_prompt = (
        "Extract all visible laboratory values: CBC, differential, platelets, CRP, "
        "electrolytes, renal markers, liver enzymes. "
        "Return STRICT JSON: { 'cbc':[{'analyte':'','value':'','units':'','reference_low':'','reference_high':''} ] }"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
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
            ],
            response_format={"type": "json_object"},
            temperature=0
        )

        data = resp.choices[0].message.parsed
        return data if isinstance(data, dict) else {"cbc": []}

    except Exception as e:
        print("OCR PAGE ERROR:", e)
        return {"cbc": []}

# ----------------------------------------------------
# INTERPRETATION MODEL
# ----------------------------------------------------
def call_ai_on_report(text: str) -> dict:
    """Returns clinical summary + structured CBC from text or OCR output."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content":
                        "Analyse CBC & chemistry. Identify patterns. Do NOT diagnose. "
                        "Return strict JSON: patient, cbc[], summary.impression, summary.suggested_follow_up."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return resp.choices[0].message.parsed

    except Exception as e:
        print("❌ AI INTERPRETATION ERROR:", e)
        raise

# ----------------------------------------------------
# CBC MAPPING
# ----------------------------------------------------
def build_cbc_dict(ai_json):
    out = {}
    for r in ai_json.get("cbc", []):
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

# ----------------------------------------------------
# ROUTE ENGINE v4 (Dr Riekert style)
# PATTERN → ROUTE → NEXT STEPS
# ----------------------------------------------------
def generate_routes(c):
    v = lambda k: clean_number(c.get(k, {}).get("value"))

    Hb = v("Hb")
    MCV = v("MCV")
    WBC = v("WBC")
    Neut = v("Neut")
    Lymph = v("Lymph")
    Plt = v("Plt")
    Cr = v("Cr")
    CRP = v("CRP")

    result = {
        "patterns": [],
        "routes": [],
        "next_steps": []
    }

    # ---------------------- ANAEMIA ----------------------
    if Hb and Hb < 13:
        if MCV and MCV < 80:
            result["patterns"].append("Microcytic anaemia pattern")
            result["routes"].append("Iron deficiency / chronic disease anaemia route")
            result["next_steps"].append("Ferritin, iron studies, transferrin saturation, reticulocyte count")

        elif MCV and MCV > 100:
            result["patterns"].append("Macrocytic anaemia pattern")
            result["routes"].append("B12 / folate deficiency route")
            result["next_steps"].append("Vitamin B12, folate, LFTs, thyroid screen")

        else:
            result["patterns"].append("Normocytic anaemia")
            result["routes"].append("Chronic inflammation / renal / early iron deficiency route")
            result["next_steps"].append("Reticulocyte count, ferritin, renal profile")

    # ---------------------- INFECTION ----------------------
    if WBC and WBC > 12:
        result["patterns"].append("Leukocytosis")
        result["routes"].append("Infection / inflammatory response route")

        if Neut and Neut > 70:
            result["patterns"].append("Neutrophil-predominant response")
            result["next_steps"].append("Consider bacterial source; correlate with CRP and symptoms")

        if Lymph and Lymph > 45:
            result["patterns"].append("Lymphocytosis")
            result["next_steps"].append("Consider viral illness or recovery phase")

    if CRP and CRP > 10:
        result["patterns"].append("Raised CRP")
        result["routes"].append("Active inflammatory route")
        result["next_steps"].append("Monitor CRP trend; evaluate infection source")

    # ---------------------- PLATELETS ----------------------
    if Plt and Plt > 450:
        result["patterns"].append("Thrombocytosis")
        result["routes"].append("Reactive thrombocytosis route")
        result["next_steps"].append("Check iron studies, inflammation markers")

    # ---------------------- RENAL ----------------------
    if Cr and Cr > 120:
        result["patterns"].append("Possible renal impairment")
        result["routes"].append("Renal function evaluation route")
        result["next_steps"].append("Repeat U&E, hydration assessment, medication review")

    # ----------------------------------------------
    if not result["patterns"]:
        result["patterns"].append("No significant abnormal patterns detected")
        result["routes"].append("No major diagnostic route triggered")
        result["next_steps"].append("Correlate with clinical picture")

    return result

# ----------------------------------------------------
# PROCESS REPORT
# ----------------------------------------------------
def process_report(job):
    rid = job["id"]
    file_path = job.get("file_path")

    print(f"Processing report {rid}…")

    pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_bytes, "data"):
        pdf_bytes = pdf_bytes.data

    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)

    all_rows = []

    if scanned:
        print("SCANNED PDF → OCR")
        pages = convert_from_bytes(pdf_bytes)
        for idx, img in enumerate(pages, 1):
            print(f"OCR page {idx}…")
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            ocr_res = extract_cbc_from_image(buf.getvalue())
            all_rows.extend(ocr_res.get("cbc", []))
    else:
        all_rows = text

    if not all_rows:
        raise ValueError("No CBC extracted.")

    merged_text = (
        json.dumps({"cbc": all_rows}) if isinstance(all_rows, list)
        else all_rows
    )

    ai_json = call_ai_on_report(merged_text)

    cdict = build_cbc_dict(ai_json)
    ai_json["clinical_engine"] = generate_routes(cdict)

    supabase.table("reports").update(
        {"ai_status": "completed", "ai_results": ai_json, "ai_error": None}
    ).eq("id", rid).execute()

    print(f"✓ Completed {rid}")
    return ai_json

# ----------------------------------------------------
# MAIN LOOP (FIXED FOR SUPABASE v2)
# ----------------------------------------------------
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

            # FIXED: New Supabase client returns .model
            rows = getattr(res, "model", None)
            jobs = rows if rows else []

            if not jobs:
                print("No jobs…")
                time.sleep(1)
                continue

            job = jobs[0]
            rid = job["id"]

            supabase.table("reports").update(
                {"ai_status": "processing"}
            ).eq("id", rid).execute()

            process_report(job)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
