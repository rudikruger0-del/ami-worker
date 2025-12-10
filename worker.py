print(">>> AMI Worker starting...")

# -----------------------
# BASE IMPORTS
# -----------------------
import os
import time
import json
import io
import traceback
import base64

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes

print(">>> Imports OK")

# -----------------------
# ENV & CLIENTS
# -----------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# ===================================================================
#                       PDF TEXT EXTRACTION
# ===================================================================

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a non-scanned PDF (selectable text)."""
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
    """Heuristic: if almost no text, assume scan/image PDF."""
    if not text:
        return True
    if len(text.strip()) < 30:
        return True
    return False


# ===================================================================
#                     OCR — CBC VALUE EXTRACTION
# ===================================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    OpenAI Vision OCR — extract CBC + chemistry from a single PNG image.
    Returns JSON: { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """

    b64 = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans. "
        "Extract ALL CBC and chemistry analytes including: WBC, differential counts, RBC, Hb, Hct, "
        "MCV, MCH, MCHC, RDW, platelets, CRP, urea, creatinine, eGFR, electrolytes, liver enzymes, CK, CK-MB, glucose. "
        "Return STRICT JSON ONLY with structure:\n"
        "{\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": \"\",\n"
        "      \"value\": \"\",\n"
        "      \"units\": \"\",\n"
        "      \"reference_low\": \"\",\n"
        "      \"reference_high\": \"\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not include any extra keys or text outside this JSON."
    )

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
                        },
                    }
                ],
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


# ===================================================================
#                     NUMERIC CLEANING HELPERS
# ===================================================================

def clean_num(v):
    """
    Convert things like '88.0%', '5,6', ' 13 ' into floats.
    Returns None if not parseable.
    """
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        # remove percent if present
        s = s.replace("%", "")
        # convert comma decimal to dot
        s = s.replace(",", ".")
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def build_cbc_list(cbc_raw):
    """
    Normalise cbc into a list of items:
      { analyte, value, units, reference_low, reference_high, flag }
    Accepts either:
      - list of items
      - dict[analyte_name] -> { ... }
    """
    if isinstance(cbc_raw, list):
        return cbc_raw

    if isinstance(cbc_raw, dict):
        out = []
        for name, obj in cbc_raw.items():
            if isinstance(obj, dict):
                item = obj.copy()
                item.setdefault("analyte", name)
                out.append(item)
        return out

    return []


def find_analyte(cbc_list, name_keywords):
    """
    Find first analyte whose label contains any keyword (case-insensitive).
    """
    for item in cbc_list:
        label = (item.get("analyte") or "").lower()
        for kw in name_keywords:
            if kw in label:
                return item
    return None


def get_value(cbc_list, name_keywords):
    """
    Return (value, units, label) for first matching analyte.
    """
    item = find_analyte(cbc_list, name_keywords)
    if not item:
        return None, None, None
    value = clean_num(item.get("value"))
    units = item.get("units")
    label = item.get("analyte")
    return value, units, label


def add_route(routes, id_, name, trigger, pattern_summary, key_values, suggested_tests, priority):
    """
    Add a structured 'route' object.
    priority: "low" | "medium" | "high"
    """
    routes.append(
        {
            "id": id_,
            "name": name,
            "trigger": trigger,
            "pattern_summary": pattern_summary,
            "key_values": [kv for kv in key_values if kv],  # drop empties
            "suggested_tests": suggested_tests,
            "priority": priority,
        }
    )


# ===================================================================
#            FULL CBC + CHEMISTRY ROUTE ENGINE (ROETES)
# ===================================================================

def generate_clinical_routes(cbc_raw):
    """
    Generates doctor-style 'roetes' for CBC + chemistry patterns.
    This is where we implement what Dr Riekert asked for:
    e.g. Hb laag → MCV & MCH pas by ystertekort → voorstel ferritien en retikulosiet telling.
    """

    cbc_list = build_cbc_list(cbc_raw)
    routes = []
    if not cbc_list:
        return routes

    # ---- Get key values ----
    Hb, _, Hb_label = get_value(cbc_list, ["hb", "haemoglobin", "hemoglobin", "hgb"])
    MCV, _, MCV_label = get_value(cbc_list, ["mcv"])
    MCH, _, MCH_label = get_value(cbc_list, ["mch"])
    MCHC, _, MCHC_label = get_value(cbc_list, ["mchc"])
    WBC, _, WBC_label = get_value(cbc_list, ["wbc", "white cell", "white blood cell"])
    Neut_val, Neut_units, Neut_label = get_value(cbc_list, ["neut", "neutrophil"])
    Lymph_val, Lymph_units, Lymph_label = get_value(cbc_list, ["lymph"])
    Plt, _, Plt_label = get_value(cbc_list, ["plt", "platelet"])
    CRP, _, CRP_label = get_value(cbc_list, ["crp"])
    Urea, _, Urea_label = get_value(cbc_list, ["urea"])
    Creat, _, Creat_label = get_value(cbc_list, ["creat", "creatinine"])
    eGFR, _, eGFR_label = get_value(cbc_list, ["egfr", "gfr"])
    Na, _, Na_label = get_value(cbc_list, ["sodium", " na"])
    K, _, K_label = get_value(cbc_list, ["potassium", "k+"])
    Cl, _, Cl_label = get_value(cbc_list, ["chloride", " cl"])
    Glucose, _, Glu_label = get_value(cbc_list, ["glucose"])
    ALT, _, ALT_label = get_value(cbc_list, ["alt"])
    AST, _, AST_label = get_value(cbc_list, ["ast"])
    GGT, _, GGT_label = get_value(cbc_list, ["ggt"])
    ALP, _, ALP_label = get_value(cbc_list, ["alp"])
    Bili, _, Bili_label = get_value(cbc_list, ["bili", "bilirubin"])
    CK, _, CK_label = get_value(cbc_list, ["ck"])
    CKMB, _, CKMB_label = get_value(cbc_list, ["ck-mb", "ck mb", "ckmb"])

    # =========================
    # 1. ANAEMIA / Hb ROUTES
    # =========================
    if Hb is not None and Hb < 12:
        trigger = f"{Hb_label or 'Hb'} low ({Hb})"

        # Microcytic – iron physiology route (this is exactly Dr Riekert’s example)
        if MCV is not None and MCV < 80:
            add_route(
                routes,
                "microcytic_iron_route",
                "Microcytic anaemia (iron physiology) route",
                trigger + f", {MCV_label or 'MCV'} low ({MCV})",
                "Pattern consistent with microcytic / iron-deficiency type physiology.",
                [
                    f"{Hb_label or 'Hb'} {Hb}",
                    f"{MCV_label or 'MCV'} {MCV}",
                    f"{MCH_label or 'MCH'} {MCH}" if MCH is not None else "",
                ],
                [
                    "Ferritin",
                    "Iron studies (serum iron, transferrin saturation)",
                    "Reticulocyte count",
                ],
                "high",
            )
        # Macrocytic – B12 / folate physiology
        elif MCV is not None and MCV > 100:
            add_route(
                routes,
                "macrocytic_route",
                "Macrocytic anaemia route",
                trigger + f", {MCV_label or 'MCV'} high ({MCV})",
                "Macrocytic physiology pattern, as can be seen with B12/folate deficiency or other causes.",
                [
                    f"{Hb_label or 'Hb'} {Hb}",
                    f"{MCV_label or 'MCV'} {MCV}",
                ],
                [
                    "Vitamin B12",
                    "Folate",
                    "Reticulocyte count",
                    "Liver function tests if clinically indicated",
                ],
                "medium",
            )
        # Normocytic – chronic disease/early iron/blood loss
        else:
            add_route(
                routes,
                "normocytic_route",
                "Normocytic anaemia route",
                trigger + (f", {MCV_label or 'MCV'} normal ({MCV})" if MCV is not None else ""),
                (
                    "Anaemia with normal cell size – may fit anaemia of chronic disease, early iron deficiency, "
                    "renal disease or blood loss physiology depending on context."
                ),
                [
                    f"{Hb_label or 'Hb'} {Hb}",
                    f"{MCV_label or 'MCV'} {MCV}" if MCV is not None else "",
                ],
                [
                    "Reticulocyte count",
                    "Inflammatory markers (CRP/ESR)",
                    "Renal profile (urea/creatinine, eGFR)",
                ],
                "medium",
            )

    # Raised Hb
    if Hb is not None and Hb > 16:
        add_route(
            routes,
            "raised_hb",
            "Raised haemoglobin route",
            f"{Hb_label or 'Hb'} high ({Hb})",
            "Raised haemoglobin can be seen with dehydration or polycythaemia physiology in the right setting.",
            [f"{Hb_label or 'Hb'} {Hb}"],
            [
                "Repeat CBC with good hydration",
                "Consider clinical assessment for symptoms of hyperviscosity",
            ],
            "medium",
        )

    # =========================
    # 2. WBC / DIFFERENTIAL / CRP
    # =========================
    neut_is_percent = bool(Neut_units and "%" in str(Neut_units))
    lymph_is_percent = bool(Lymph_units and "%" in str(Lymph_units))

    # Overall high WBC
    if WBC is not None and WBC > 11:
        add_route(
            routes,
            "raised_wcc",
            "Raised white cell count route",
            f"{WBC_label or 'WBC'} high ({WBC})",
            "Raised white cell count suggesting active inflammatory or infectious physiology.",
            [f"{WBC_label or 'WBC'} {WBC}"],
            [
                "Clinical correlation for infection (fever, focus, vital signs)",
                "Repeat CBC if clinically indicated",
            ],
            "medium",
        )

    # Neutrophilia – bacterial-type physiology
    if Neut_val is not None:
        if neut_is_percent and Neut_val > 75:
            trig = f"{Neut_label or 'Neutrophils'} high percentage ({Neut_val}%)"
            add_route(
                routes,
                "neutrophilia_percent",
                "Neutrophil-predominant (bacterial-type) route",
                trig,
                "Neutrophil-predominant pattern, often seen with bacterial infection or acute stress physiology.",
                [trig],
                ["CRP", "Clinical assessment for bacterial source (lungs, urine, skin, abdomen)"],
                "high",
            )
        elif not neut_is_percent and Neut_val > 7.5:
            trig = f"{Neut_label or 'Neutrophils'} high ({Neut_val})"
            add_route(
                routes,
                "neutrophilia_abs",
                "Neutrophil-predominant (bacterial-type) route",
                trig,
                "Neutrophil-predominant pattern, often seen with bacterial infection or acute stress physiology.",
                [trig],
                ["CRP", "Clinical assessment for bacterial source"],
                "high",
            )

    # Lymphocytosis – viral-type physiology
    if Lymph_val is not None:
        if lymph_is_percent and Lymph_val > 40:
            trig = f"{Lymph_label or 'Lymphocytes'} high percentage ({Lymph_val}%)"
            add_route(
                routes,
                "lymphocytosis_percent",
                "Lymphocyte-predominant (viral-type) route",
                trig,
                "Lymphocyte-predominant pattern often seen with viral infection physiology.",
                [trig],
                ["Clinical correlation for viral syndrome (URTI, GI, systemic)", "Repeat CBC if indicated"],
                "medium",
            )
        elif not lymph_is_percent and Lymph_val > 4:
            trig = f"{Lymph_label or 'Lymphocytes'} high ({Lymph_val})"
            add_route(
                routes,
                "lymphocytosis_abs",
                "Lymphocyte-predominant (viral-type) route",
                trig,
                "Lymphocyte-predominant pattern often seen with viral infection physiology.",
                [trig],
                ["Clinical correlation for viral syndrome", "Repeat CBC if indicated"],
                "medium",
            )

    # CRP inflammation route
    if CRP is not None and CRP > 10:
        severity = "moderately" if CRP <= 100 else "markedly"
        add_route(
            routes,
            "crp_route",
            "CRP inflammatory route",
            f"{CRP_label or 'CRP'} {severity} raised ({CRP})",
            "Raised CRP indicating active inflammatory or infectious physiology.",
            [f"{CRP_label or 'CRP'} {CRP}"],
            ["Trend CRP if monitoring response", "Correlate with clinical course and antibiotics if used"],
            "high" if CRP > 100 else "medium",
        )

    # =========================
    # 3. PLATELETS
    # =========================
    if Plt is not None and Plt > 450:
        add_route(
            routes,
            "thrombocytosis",
            "Thrombocytosis route",
            f"{Plt_label or 'Platelets'} high ({Plt})",
            "Raised platelet count which can be reactive (inflammation, iron deficiency physiology) or primary.",
            [f"{Plt_label or 'Platelets'} {Plt}"],
            ["Look for inflammatory or iron-deficiency features", "Repeat platelet count if persistent"],
            "medium",
        )

    if Plt is not None and Plt < 150:
        prio = "high" if Plt < 80 else "medium"
        add_route(
            routes,
            "thrombocytopenia",
            "Thrombocytopenia route",
            f"{Plt_label or 'Platelets'} low ({Plt})",
            "Low platelet count pattern – may increase bleeding tendency, depending on the level and clinical picture.",
            [f"{Plt_label or 'Platelets'} {Plt}"],
            [
                "Review for bleeding/bruising, mucosal bleeding",
                "Repeat platelet count",
                "Consider peripheral smear if persistent",
            ],
            prio,
        )

    # =========================
    # 4. KIDNEY / RENAL FUNCTION
    # =========================
    if (Creat is not None and Creat > 100) or (Urea is not None and Urea > 8):
        key_vals = []
        if Urea is not None:
            key_vals.append(f"{Urea_label or 'Urea'} {Urea}")
        if Creat is not None:
            key_vals.append(f"{Creat_label or 'Creatinine'} {Creat}")
        trig = ", ".join(key_vals)
        add_route(
            routes,
            "renal_stress",
            "Renal function / kidney stress route",
            trig,
            "Pattern consistent with possible renal stress or reduced kidney function physiology.",
            key_vals,
            [
                "Check eGFR if not already provided",
                "Review blood pressure and hydration status",
                "Repeat renal profile if indicated",
            ],
            "high" if (Creat is not None and Creat > 200) else "medium",
        )

    if eGFR is not None and eGFR < 60:
        add_route(
            routes,
            "low_egfr",
            "Reduced eGFR route",
            f"{eGFR_label or 'eGFR'} low ({eGFR})",
            "Reduced estimated GFR suggesting decreased kidney filtration capacity.",
            [f"{eGFR_label or 'eGFR'} {eGFR}"],
            [
                "Correlate with chronic kidney disease history",
                "Monitor renal function and blood pressure",
            ],
            "high" if eGFR < 30 else "medium",
        )

    # =========================
    # 5. ELECTROLYTES
    # =========================
    if Na is not None and (Na < 135 or Na > 145):
        add_route(
            routes,
            "sodium_route",
            "Sodium balance route",
            f"{Na_label or 'Sodium'} {Na}",
            "Abnormal sodium level which can alter fluid balance and neurological status.",
            [f"{Na_label or 'Sodium'} {Na}"],
            [
                "Assess volume status",
                "Repeat electrolytes",
                "Look for causes of hypo/hypernatraemia (medications, dehydration, SIADH, etc.)",
            ],
            "high" if Na < 130 or Na > 150 else "medium",
        )

    if K is not None and (K < 3.3 or K > 5.5):
        add_route(
            routes,
            "potassium_route",
            "Potassium route",
            f"{K_label or 'Potassium'} {K}",
            "Abnormal potassium which can impact cardiac rhythm physiology.",
            [f"{K_label or 'Potassium'} {K}"],
            [
                "Repeat potassium urgently to confirm",
                "ECG if significantly abnormal",
                "Review medications and renal function",
            ],
            "high",
        )

    # =========================
    # 6. LIVER ENZYMES
    # =========================
    liver_vals = []
    if ALT is not None and ALT > 40:
        liver_vals.append(f"{ALT_label or 'ALT'} {ALT}")
    if AST is not None and AST > 40:
        liver_vals.append(f"{AST_label or 'AST'} {AST}")
    if GGT is not None and GGT > 60:
        liver_vals.append(f"{GGT_label or 'GGT'} {GGT}")
    if ALP is not None and ALP > 130:
        liver_vals.append(f"{ALP_label or 'ALP'} {ALP}")
    if Bili is not None and Bili > 20:
        liver_vals.append(f"{Bili_label or 'Bilirubin'} {Bili}")

    if liver_vals:
        add_route(
            routes,
            "liver_route",
            "Liver enzyme / hepatobiliary route",
            ", ".join(liver_vals),
            "Pattern of raised liver enzymes / bilirubin suggesting hepatocellular or cholestatic physiology.",
            liver_vals,
            [
                "Correlate with alcohol, medications and viral hepatitis risk",
                "Consider abdominal ultrasound if clinically indicated",
            ],
            "medium",
        )

    # =========================
    # 7. CK / CK-MB (MUSCLE / CARDIAC)
    # =========================
    if CK is not None and CK > 200:
        prio = "high" if CK > 5000 else "medium"
        add_route(
            routes,
            "ck_route",
            "Muscle injury / CK route",
            f"{CK_label or 'CK'} high ({CK})",
            "Raised CK suggesting muscle injury or stress physiology; very high levels carry risk of renal stress.",
            [f"{CK_label or 'CK'} {CK}"],
            [
                "Review for muscle pain, trauma, seizures or prolonged immobilisation",
                "Monitor renal function (urea/creatinine, eGFR)",
                "Ensure adequate hydration if clinically appropriate",
            ],
            prio,
        )

    if CKMB is not None and CKMB > 5:
        add_route(
            routes,
            "ckmb_route",
            "CK-MB route",
            f"{CKMB_label or 'CK-MB'} raised ({CKMB})",
            "Raised CK-MB fraction which may reflect myocardial-related CK fraction in the right clinical context.",
            [f"{CKMB_label or 'CK-MB'} {CKMB}"],
            [
                "Correlate with chest pain, ECG and troponin (if clinically indicated)",
            ],
            "high",
        )

    # =========================
    # 8. GLUCOSE
    # =========================
    if Glucose is not None and (Glucose < 3.5 or Glucose > 7.8):
        prio = "high" if Glucose < 3.0 or Glucose > 15 else "medium"
        add_route(
            routes,
            "glucose_route",
            "Glucose regulation route",
            f"{Glu_label or 'Glucose'} {Glucose}",
            "Abnormal glucose level suggesting hypo- or hyperglycaemia physiology.",
            [f"{Glu_label or 'Glucose'} {Glucose}"],
            [
                "Repeat fasting or random glucose",
                "Consider HbA1c in non-acute setting",
            ],
            prio,
        )

    # Drop routes where there are no key values at all
    routes = [r for r in routes if any(r["key_values"])]
    return routes


# ===================================================================
#                     LLM INTERPRETATION STEP
# ===================================================================

def call_ai_on_report(text: str) -> dict:
    """
    Call OpenAI to produce structured CBC + summary.
    Routes are added by our own engine, not by the model.
    """
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing a Complete Blood Count (CBC) and related "
        "laboratory values (CRP, kidney function, liver enzymes, CK, CK-MB, electrolytes, glucose). "
        "You MUST NOT make a diagnosis or prescribe treatment. Describe physiology and patterns only.\n\n"
        "You are writing for South African clinicians (GPs, paediatricians, physicians).\n\n"
        "Output STRICT JSON with this structure:\n"
        "{\n"
        "  \"patient\": {\n"
        "    \"name\": string | null,\n"
        "    \"age\": number | null,\n"
        "    \"sex\": \"Male\" | \"Female\" | \"Unknown\"\n"
        "  },\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": string,\n"
        "      \"value\": number | string | null,\n"
        "      \"units\": string | null,\n"
        "      \"reference_low\": number | null,\n"
        "      \"reference_high\": number | null,\n"
        "      \"flag\": \"low\" | \"normal\" | \"high\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"overall_severity\": \"mild\" | \"moderate\" | \"high\" | \"critical\",\n"
        "    \"impression\": string,\n"
        "    \"suggested_follow_up\": string\n"
        "  }\n"
        "}\n\n"
        "If the input already contains a JSON object with a 'cbc' array, reuse those values as far as possible.\n"
        "Return ONLY this JSON object. No markdown, no explanation."
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

    raw = response.choices[0].message.content
    return json.loads(raw)


# ===================================================================
#                        PROCESS A SINGLE REPORT
# ===================================================================

def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    try:
        if not file_path:
            err = f"Missing file_path for report {report_id}"
            print("⚠️", err)
            supabase.table("reports").update(
                {"ai_status": "failed", "ai_error": err}
            ).eq("id", report_id).execute()
            return {"error": err}

        # Download original PDF from Supabase storage
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        # Check if PDF has selectable text
        pdf_text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(pdf_text)

        # ---------- SCANNED PIPELINE (OCR) ----------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → OCR pipeline")
            pages = convert_from_bytes(pdf_bytes)
            combined_cbc = []

            for img in pages:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

                try:
                    ocr_result = extract_cbc_from_image(img_bytes)
                    if isinstance(ocr_result, dict) and "cbc" in ocr_result:
                        combined_cbc.extend(ocr_result["cbc"])
                except Exception as e:
                    print("Vision OCR error:", e)

            if not combined_cbc:
                raise ValueError("Vision OCR could not extract CBC values")

            merged_text = "Structured CBC JSON from OCR:\n" + json.dumps({"cbc": combined_cbc})

        # ---------- DIGITAL PIPELINE ----------
        else:
            print(f"📝 Report {report_id}: Digital PDF detected → using extracted text")
            merged_text = pdf_text
            if l_text:
                merged_text = (pdf_text + "\n\nClinical notes:\n" + l_text).strip()

        if not merged_text.strip():
            raise ValueError("No usable text extracted for AI analysis")

        # LLM interpretation step
        ai_json = call_ai_on_report(merged_text)

        # Deterministic route engine (CBC + chemistry)
        cbc_raw = ai_json.get("cbc") or []
        routes = generate_clinical_routes(cbc_raw)
        ai_json["routes"] = routes

        # Save results
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


# ===================================================================
#                           WORKER LOOP
# ===================================================================

def main():
    print("AMI Worker with OCR + CBC/chemistry routes started… watching for jobs.")
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
