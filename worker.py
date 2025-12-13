print(">>> AMI Worker starting…")

import os
import time
import json
import io
import traceback
import base64
import re

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes

# =====================================================
# ENV + CLIENTS
# =====================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set – OpenAI calls will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# =====================================================
# GENERIC HELPERS
# =====================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable text PDFs."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("PDF parse error:", e)
        return ""


def is_scanned_pdf(text: str) -> bool:
    """If very little text, assume scanned image PDF."""
    return len((text or "").strip()) < 30


def clean_number(val):
    """
    Safely convert lab values like '88.0%', '11,6 g/dL', '4.2*' → float.
    Returns None if conversion fails.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val)
    s = s.replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


def parse_json_from_message(message) -> dict:
    """
    OpenAI 1.51.0 + response_format={"type":"json_object"}:
    message.content can be a string or a list of parts with type="text".
    """
    content = message.content
    if isinstance(content, str):
        raw = content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        raw = "\n".join(texts)
    else:
        raise ValueError("Unsupported message.content type")

    try:
        return json.loads(raw)
    except Exception as e:
        print("JSON parse error:", e, "RAW:", raw[:300])
        raise


# =====================================================
# OCR – EXTRACT CBC TABLE FROM SCANNED IMAGES
# =====================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Sends a scanned image to OpenAI Vision to extract CBC + chemistry.
    Returns dict: { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR assistant for medical lab reports.\n"
        "Read the image and extract ANY numeric analytes you see – full blood count, "
        "differential, platelets, CRP, ESR, electrolytes, urea, creatinine, liver enzymes, "
        "CK, CK-MB, etc.\n"
        "Return STRICT JSON:\n"
        "{ 'cbc': [ { 'analyte':'', 'value':'', 'units':'', "
        "'reference_low':'', 'reference_high':'' } ] }.\n"
        "Do not add explanations or free text."
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
                            "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }
                        }
                    ]
                },
            ],
        )

        data = parse_json_from_message(resp.choices[0].message)
        if not isinstance(data, dict):
            return {"cbc": []}
        if "cbc" not in data or not isinstance(data["cbc"], list):
            data["cbc"] = []
        return data

    except Exception as e:
        print("OCR PAGE ERROR:", e)
        # Never crash – just say “no rows”
        return {"cbc": []}


# =====================================================
# INTERPRETATION MODEL
# =====================================================

def call_ai_on_report(text: str) -> dict:
    """
    Main interpretation model. Produces a structured JSON:
    patient, cbc, narrative summary, and basic follow-up text.
    Route engine will enrich this.
    """
    MAX_CHARS = 12000
    if text and len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool interpreting full blood count (CBC) and "
        "basic chemistry results.\n\n"
        "CRITICAL:\n"
        "- You do NOT diagnose.\n"
        "- You do NOT prescribe.\n"
        "- Use phrases like 'may suggest', 'can be compatible with'.\n\n"
        "INPUT is either:\n"
        "  • Raw extracted lab text, OR\n"
        "  • A JSON-like object with a 'cbc' array from OCR.\n\n"
        "Return STRICT JSON only, with at least:\n"
        "{\n"
        "  'patient': { 'name': null, 'age': null, 'sex': 'Unknown' },\n"
        "  'cbc': [\n"
        "    {\n"
        "      'analyte': '', 'value': '', 'units': '',\n"
        "      'reference_low': '', 'reference_high': '',\n"
        "      'flag': 'low|normal|high|unknown'\n"
        "    }\n"
        "  ],\n"
        "  'summary': {\n"
        "    'impression': '',\n"
        "    'suggested_follow_up': ''\n"
        "  }\n"
        "}\n"
        "If there are no numeric lab values, return an EMPTY 'cbc' array and describe "
        "that no CBC data could be extracted."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text or "No readable text provided."}
        ],
    )

    data = parse_json_from_message(resp.choices[0].message)
    if not isinstance(data, dict):
        raise ValueError("Interpretation result is not a JSON object")

    if "cbc" not in data or not isinstance(data["cbc"], list):
        data["cbc"] = []

    if "summary" not in data or not isinstance(data["summary"], dict):
        data["summary"] = {
            "impression": "",
            "suggested_follow_up": ""
        }

    return data


# =====================================================
# BUILD CANONICAL CBC DICT FOR ROUTES
# =====================================================

def build_cbc_dict(ai_json: dict) -> dict:
    """
    Turn ai_json['cbc'] list into a dict with stable keys:
      Hb, MCV, MCH, WBC, Neut, Lymph, Plt,
      Cr, Urea, Na, K, CRP, ALT, AST, ALP, GGT, Bili, CK
    """
    out = {}
    rows = ai_json.get("cbc") or []
    if not isinstance(rows, list):
        return out

    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("analyte") or r.get("test") or "").lower().strip()
        if not name:
            continue

        def set_if_empty(key):
            if key not in out:
                out[key] = r

        # RBC / Anaemia
        if "haemoglobin" in name or "hemoglobin" in name or name == "hb":
            set_if_empty("Hb")
        elif name.startswith("mcv"):
            set_if_empty("MCV")
        elif name.startswith("mch"):
            set_if_empty("MCH")
        elif "red cell" in name or "rbc" in name:
            set_if_empty("RBC")

        # White cells
        elif "white cell" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            set_if_empty("WBC")
        elif "neut" in name:
            set_if_empty("Neut")
        elif "lymph" in name:
            set_if_empty("Lymph")
        elif "mono" in name:
            set_if_empty("Mono")
        elif "eosin" in name:
            set_if_empty("Eos")
        elif "baso" in name:
            set_if_empty("Baso")

        # Platelets
        elif "platelet" in name or name == "plt":
            set_if_empty("Plt")

        # Renal
        elif "creatinine" in name:
            set_if_empty("Cr")
        elif "urea" in name:
            set_if_empty("Urea")

        # Inflammation
        elif name.startswith("crp") or "c-reactive" in name:
            set_if_empty("CRP")
        elif "esr" in name or "sed rate" in name:
            set_if_empty("ESR")

        # Electrolytes
        elif "sodium" in name or name == "na":
            set_if_empty("Na")
        elif "potassium" in name or name == "k":
            set_if_empty("K")
        elif "chloride" in name or name == "cl":
            set_if_empty("Cl")
        elif "bicarbonate" in name or "hco3" in name or "tco2" in name or "co2" == name:
            set_if_empty("HCO3")

        # Liver
        elif name == "alt" or "alanine aminotransferase" in name:
            set_if_empty("ALT")
        elif name == "ast" or "aspartate aminotransferase" in name:
            set_if_empty("AST")
        elif name == "alp" or "alkaline phosphatase" in name:
            set_if_empty("ALP")
        elif "ggt" in name or "gamma glutamyl" in name:
            set_if_empty("GGT")
        elif "bilirubin" in name:
            set_if_empty("Bili")

        # Muscle
        elif name == "ck" or "creatine kinase" in name:
            set_if_empty("CK")
        elif "ck-mb" in name or "ck mb" in name:
            set_if_empty("CKMB")

    return out


# =====================================================
# ROUTE ENGINE V3: PATTERN → ROUTE → NEXT STEPS
# =====================================================

def make_route(name, pattern, route, steps):
    return {
        "name": name,
        "pattern": pattern,
        "route": route,
        "steps": steps,
    }


def generate_routes(c: dict):
    """
    Takes canonical CBC dict → returns a list of
    {name, pattern, route, steps[]} objects.
    """
    v = lambda k: clean_number(c.get(k, {}).get("value"))

    Hb   = v("Hb")
    MCV  = v("MCV")
    MCH  = v("MCH")
    WBC  = v("WBC")
    Neut = v("Neut")
    Lymph = v("Lymph")
    Plt  = v("Plt")
    Cr   = v("Cr")
    Urea = v("Urea")
    Na   = v("Na")
    K    = v("K")
    CRP  = v("CRP")
    ALT  = v("ALT")
    AST  = v("AST")
    ALP  = v("ALP")
    GGT  = v("GGT")
    Bili = v("Bili")
    CK   = v("CK")

    routes = []

    # ---------------- ANAEMIA ROUTES ----------------
    if Hb is not None and Hb < 13:
        if MCV is not None and MCV < 80:
            routes.append(make_route(
                name="Microcytic anaemia pattern",
                pattern=f"Hb {Hb} low with MCV {MCV} low ± low MCH",
                route="Pattern is compatible with iron deficiency or other microcytic anaemias.",
                steps=[
                    "Check ferritin and iron studies.",
                    "Order reticulocyte count.",
                    "Review menstrual/blood loss history, diet and chronic disease.",
                ]
            ))
        elif MCV is not None and MCV > 100:
            routes.append(make_route(
                name="Macrocytic anaemia pattern",
                pattern=f"Hb {Hb} low with MCV {MCV} high",
                route="Pattern may suggest B12/folate deficiency, liver disease or medication effect.",
                steps=[
                    "Check vitamin B12 and folate.",
                    "Review liver function and alcohol/medication history.",
                    "Consider thyroid function tests if clinically indicated.",
                ]
            ))
        else:
            routes.append(make_route(
                name="Normocytic anaemia pattern",
                pattern=f"Hb {Hb} low with MCV in normal range",
                route="Pattern may be compatible with anaemia of chronic disease, renal disease or early iron deficiency.",
                steps=[
                    "Check creatinine/eGFR and CRP/ESR.",
                    "Consider ferritin and reticulocyte count.",
                    "Correlate with chronic inflammatory or renal history.",
                ]
            ))

    # --------------- WHITE CELL / INFECTION ROUTES ---------------
    if WBC is not None:
        if WBC > 12:
            if Neut is not None and Neut > 70:
                routes.append(make_route(
                    name="Neutrophil-predominant leukocytosis",
                    pattern=f"WBC {WBC} with neutrophils {Neut}%",
                    route="Pattern is compatible with an acute bacterial or inflammatory response.",
                    steps=[
                        "Look for clinical source of infection or inflammation.",
                        "Correlate with CRP and temperature.",
                        "Consider cultures or imaging depending on clinical picture.",
                    ]
                ))
            elif Lymph is not None and Lymph > 45:
                routes.append(make_route(
                    name="Lymphocyte-predominant leukocytosis",
                    pattern=f"WBC {WBC} with lymphocytes {Lymph}%",
                    route="Pattern can be seen in viral infections or recovery from infection.",
                    steps=[
                        "Review history for recent viral illness or atypical infections.",
                        "Repeat CBC if clinically indicated.",
                        "Consider specific viral testing if symptoms suggest.",
                    ]
                ))
            else:
                routes.append(make_route(
                    name="Raised white cell count",
                    pattern=f"WBC {WBC} elevated",
                    route="Non-specific leukocytosis; may reflect infection, inflammation, stress or steroid effect.",
                    steps=[
                        "Review medications (especially steroids).",
                        "Correlate with CRP, temperature and symptoms.",
                        "Repeat CBC if picture remains unclear.",
                    ]
                ))
        elif WBC < 4:
            routes.append(make_route(
                name="Leukopenia route",
                pattern=f"WBC {WBC} reduced",
                route="Low white cell count can be seen with viral suppression, marrow disorders or medication effects.",
                steps=[
                    "Review recent viral illness and medications (e.g. chemotherapy, immunosuppressants).",
                    "Check differential count and previous results.",
                    "Consider repeat CBC and further evaluation if persistent.",
                ]
            ))

    # --------------- PLATELET ROUTES ---------------
    if Plt is not None:
        if Plt < 150:
            routes.append(make_route(
                name="Thrombocytopenia route",
                pattern=f"Platelet count {Plt} x10^9/L",
                route="Low platelet count may increase bleeding risk and can be immune, consumptive, marrow or drug related.",
                steps=[
                    "Review medication history and infections.",
                    "Check for bruising/bleeding and correlate with coagulation tests.",
                    "Consider repeat count and haematology opinion if significantly reduced.",
                ]
            ))
        elif Plt > 450:
            routes.append(make_route(
                name="Thrombocytosis route",
                pattern=f"Platelet count {Plt} x10^9/L",
                route="Raised platelets are often reactive (infection, inflammation, iron deficiency) but can be clonal.",
                steps=[
                    "Look for infection, inflammation or iron deficiency.",
                    "Review previous platelet counts.",
                    "Consider haematology opinion if persistently very high.",
                ]
            ))

    # --------------- RENAL ROUTES ---------------
    if Cr is not None:
        if Cr > 120:
            routes.append(make_route(
                name="Renal impairment route",
                pattern=f"Creatinine {Cr} elevated",
                route="Pattern suggests reduced renal function (acute or chronic).",
                steps=[
                    "Review baseline creatinine and eGFR.",
                    "Assess hydration status, blood pressure and nephrotoxic drugs.",
                    "Repeat U&E and monitor trend.",
                ]
            ))

    if Urea is not None and Urea > 8:
        routes.append(make_route(
            name="Raised urea route",
            pattern=f"Urea {Urea} elevated",
            route="May reflect dehydration, high protein load or renal impairment.",
            steps=[
                "Interpret together with creatinine and clinical hydration.",
                "Review gastrointestinal bleeding risk if symptoms suggest.",
            ]
        ))

    # --------------- ELECTROLYTE ROUTES ---------------
    if K is not None:
        if K < 3.3:
            routes.append(make_route(
                name="Hypokalaemia route",
                pattern=f"Potassium {K} mmol/L (low)",
                route="Low potassium can increase arrhythmia risk.",
                steps=[
                    "Review medications (diuretics, β-agonists, insulin).",
                    "Check for vomiting/diarrhoea and magnesium status.",
                    "Obtain ECG if symptomatic or markedly low.",
                ]
            ))
        elif K > 5.5:
            routes.append(make_route(
                name="Hyperkalaemia route",
                pattern=f"Potassium {K} mmol/L (high)",
                route="High potassium may be life-threatening due to arrhythmia risk.",
                steps=[
                    "Repeat potassium urgently to exclude sample artefact.",
                    "Obtain ECG and correlate with symptoms.",
                    "Consider urgent treatment pathway according to local protocol.",
                ]
            ))

    if Na is not None:
        if Na < 133:
            routes.append(make_route(
                name="Hyponatraemia route",
                pattern=f"Sodium {Na} mmol/L (low)",
                route="Hyponatraemia can reflect fluid overload, endocrine causes or diuretic use.",
                steps=[
                    "Assess volume status (hypo/eu/hypervolaemia).",
                    "Review medications (diuretics, antidepressants).",
                    "Check serum and urine osmolality if workup is needed.",
                ]
            ))
        elif Na > 146:
            routes.append(make_route(
                name="Hypernatraemia route",
                pattern=f"Sodium {Na} mmol/L (high)",
                route="Hypernatraemia usually reflects water loss exceeding sodium loss.",
                steps=[
                    "Assess fluid intake and losses (fever, diarrhoea, diabetes insipidus).",
                    "Correct gradually with careful fluid management.",
                ]
            ))

    # --------------- INFLAMMATION ROUTES ---------------
    if CRP is not None:
        if CRP > 10:
            routes.append(make_route(
                name="Raised CRP route",
                pattern=f"CRP {CRP} mg/L elevated",
                route="Elevated CRP indicates an inflammatory or infective process.",
                steps=[
                    "Correlate with WBC, temperature and symptoms.",
                    "Look for source of infection or inflammation.",
                    "Use trend (rising/falling) to monitor response to treatment.",
                ]
            ))

    # --------------- LIVER / MUSCLE ROUTES ---------------
    liver_vals = [ALT, AST, ALP, GGT, Bili]
    if any(vv is not None and vv > 1.5 for vv in liver_vals):
        routes.append(make_route(
            name="Liver function abnormal route",
            pattern="One or more liver markers elevated",
            route="Pattern suggests possible hepatic or cholestatic process.",
            steps=[
                "Review medications, alcohol intake and viral risk factors.",
                "Examine for jaundice or hepatomegaly.",
                "Consider liver ultrasound or further serology if clinically indicated.",
            ]
        ))

    if CK is not None and CK > 300:
        routes.append(make_route(
            name="Muscle injury / CK route",
            pattern=f"CK {CK} U/L elevated",
            route="Raised CK may indicate muscle injury (exercise, trauma, myositis, statins, etc.).",
            steps=[
                "Review recent exercise, trauma and medications (e.g. statins).",
                "Consider repeat CK and renal function monitoring if markedly raised.",
            ]
        ))

    if not routes:
        routes.append(make_route(
            name="No major route triggered",
            pattern="No clearly abnormal CBC or chemistry thresholds met.",
            route="No specific route-level pattern identified from available results.",
            steps=[
                "Correlate with history, examination and prior results.",
                "Repeat testing if symptoms change or persist.",
            ]
        ))

    return routes


# =====================================================
# PROCESS A SINGLE REPORT
# =====================================================

def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path or str(file_path).strip() == "":
            err = f"Missing file_path for report {report_id}"
            print("⚠️", err)
            supabase.table("reports").update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
            return {"error": err}

        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        print(f"Report {report_id}: scanned={scanned}, text_length={len(text)}")

        # --------- SCANNED PDF: OCR PATH ----------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF → OCR pipeline")
            images = convert_from_bytes(pdf_bytes)
            all_rows = []

            for idx, img in enumerate(images, start=1):
                print(f"OCR on page {idx}…")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                ocr_data = extract_cbc_from_image(buf.getvalue())
                rows = ocr_data.get("cbc") or []
                if rows:
                    print(f" → {len(rows)} rows extracted on page {idx}")
                    all_rows.extend(rows)
                else:
                    print(f" → No CBC rows recognised on page {idx}")

            if all_rows:
                merged_text = json.dumps({"cbc": all_rows}, ensure_ascii=False)
            else:
                print("⚠️ OCR found no numeric CBC data; falling back to text (if any).")
                merged_text = text or l_text or "No numeric lab values detected."

        # --------- DIGITAL PDF: TEXT PATH ----------
        else:
            print(f"📝 Report {report_id}: Digital PDF → text interpreter")
            merged_text = text or l_text or "No readable text detected."

        # --------- MAIN INTERPRETATION ----------
        ai_json = call_ai_on_report(merged_text)

        # --------- ROUTE ENGINE ----------
        try:
            cbc_dict = build_cbc_dict(ai_json)
            routes = generate_routes(cbc_dict)
            ai_json["routes"] = routes

            # Also add a short one-line summary of routes for UIs that only show text
            route_names = [r["name"] for r in routes] if routes else []
            if route_names:
                extra = "Routes considered: " + "; ".join(route_names) + "."
                summary = ai_json.get("summary", {})
                old_follow = summary.get("suggested_follow_up", "") or ""
                summary["suggested_follow_up"] = (old_follow + "\n" + extra).strip()
                ai_json["summary"] = summary
        except Exception as e:
            print("Route engine error:", e)
            traceback.print_exc()

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


# =====================================================
# WORKER LOOP
# =====================================================

def main():
    print("AMI Worker with OCR + Route Engine V3 started – watching for jobs.")
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

            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            job_id = job["id"]
            print(f"🔎 Found job {job_id}")

            supabase.table("reports").update(
                {"ai_status": "processing"}
            ).eq("id", job_id).execute()

            process_report(job)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(3)


if __name__ == "__main__":
    main()
