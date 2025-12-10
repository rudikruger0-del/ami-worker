print(">>> AMI Worker starting, loading imports...")

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
    import re
    print(">>> Base imports loaded")
except Exception as e:
    print("❌ Failed loading base imports:", e)
    raise e

# -----------------------
# THIRD PARTY IMPORTS
# -----------------------
try:
    from supabase import create_client, Client
    print(">>> Supabase imported")
except Exception as e:
    print("❌ Failed importing Supabase:", e)
    raise e

try:
    from openai import OpenAI
    print(">>> OpenAI imported")
except Exception as e:
    print("❌ Failed importing OpenAI:", e)
    raise e

try:
    from pypdf import PdfReader
    print(">>> pypdf imported")
except Exception as e:
    print("❌ Failed importing pypdf:", e)
    raise e

try:
    from pdf2image import convert_from_bytes
    print(">>> pdf2image imported")
except Exception as e:
    print("❌ Failed importing pdf2image:", e)
    raise e


# ======================================================
#   ENV + CLIENTS
# ======================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(">>> Environment variables loaded")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️  OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"


# ======================================================
#   PDF HELPERS
# ======================================================

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


def is_scanned_pdf(pdf_text: str) -> bool:
    """If there’s almost no text, assume the PDF is scanned."""
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False


# ======================================================
#   SMALL UTILITIES
# ======================================================

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
    # Replace comma decimal + extract first numeric piece
    s = s.replace(",", ".")
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except Exception:
        return None


# ======================================================
#   VISION OCR: EXTRACT CBC TABLE FROM SCANNED IMAGE
#   (USING chat.completions WITH gpt-4o)
# ======================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Sends a scanned image to OpenAI Vision to extract CBC + chemistry values.
    Returns dict like { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans. "
        "Extract ALL CBC and chemistry analytes you can see, including: full blood count, "
        "differential, platelets, ESR (if present), electrolytes, urea, creatinine, eGFR, "
        "bicarbonate / CO2, liver enzymes, CK, CK-MB, CRP and other inflammatory markers. "
        "Return STRICT JSON with this structure:\n"
        "{ \"cbc\": [ { \"analyte\": \"\", \"value\": \"\", \"units\": \"\", "
        "\"reference_low\": \"\", \"reference_high\": \"\" } ] }\n"
        "Do not summarise, do not interpret – just the table."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Here is a scanned lab report page. Extract all lab table rows you can see."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                },
            ],
            temperature=0.1,
        )
    except Exception as e:
        print("❌ OCR API call failed:", e)
        raise

    content = resp.choices[0].message.content

    if isinstance(content, list):
        raw = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    else:
        raw = content or ""

    if not raw.strip():
        raise ValueError("OCR returned empty JSON content")

    try:
        data = json.loads(raw)
    except Exception as e:
        print("❌ OCR JSON parse error:", e, "RAW:", raw[:200])
        raise

    if not isinstance(data, dict):
        raise ValueError("OCR result is not a JSON object")

    return data


# ======================================================
#   INTERPRETATION MODEL (TEXT OR STRUCTURED JSON)
# ======================================================

def call_ai_on_report(text: str) -> dict:
    """
    Main interpretation model.
    Input may be:
      - Raw lab report text, OR
      - A JSON string like { "cbc": [ ... ] } from OCR.

    Returns JSON:
    {
      "patient": {...},
      "cbc": [ ... ],
      "summary": { "impression": "", "suggested_follow_up": "" }
    }
    The route engine will later add a "routes" field.
    """
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing full blood count (CBC) and chemistry results.\n"
        "You do NOT diagnose or prescribe. You only describe patterns and possible clinical routes.\n\n"
        "INPUT FORMAT:\n"
        "- You may receive either:\n"
        "  (a) raw lab report text, OR\n"
        "  (b) a JSON string with a 'cbc' array containing analytes extracted by OCR.\n\n"
        "TASK:\n"
        "- Normalise all available results into a structured CBC/chemistry list.\n"
        "- Identify which are low / normal / high.\n"
        "- Write a short clinical impression and suggested follow-up.\n\n"
        "OUTPUT FORMAT (STRICT JSON):\n"
        "{\n"
        "  \"patient\": { \"name\": null, \"age\": null, \"sex\": \"Unknown\" },\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": \"\",\n"
        "      \"value\": \"\",\n"
        "      \"units\": \"\",\n"
        "      \"reference_low\": \"\",\n"
        "      \"reference_high\": \"\",\n"
        "      \"flag\": \"low\" | \"normal\" | \"high\" | \"unknown\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "      \"impression\": \"\",\n"
        "      \"suggested_follow_up\": \"\"\n"
        "  }\n"
        "}\n"
        "If you cannot find some values, just omit them – never invent numbers."
    )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.1,
    )

    content = resp.choices[0].message.content

    if isinstance(content, list):
        raw = "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict)
        )
    else:
        raw = content or ""

    if not raw.strip():
        raise ValueError("Interpretation returned empty JSON content")

    try:
        ai_json = json.loads(raw)
    except Exception as e:
        print("❌ Interpretation JSON parse error:", e, "RAW:", raw[:200])
        raise

    if not isinstance(ai_json, dict):
        raise ValueError("Interpretation result is not a dict JSON object")

    return ai_json


# ======================================================
#   BUILD CANONICAL CBC VALUE DICT FOR ROUTES
# ======================================================

def build_cbc_value_dict(ai_json: dict) -> dict:
    """
    Turn ai_json["cbc"] (list) into a dict like:
    {
      "Hb": {"value": "...", ...},
      "WBC": {...},
      "MCV": {...},
      ...
    }
    so the route engine can work on consistent names.
    """
    cbc_values = {}
    rows = ai_json.get("cbc") or []
    if not isinstance(rows, list):
        return cbc_values

    for row in rows:
        if not isinstance(row, dict):
            continue
        name_raw = row.get("analyte") or row.get("test") or ""
        name = name_raw.lower().strip()
        if not name:
            continue

        def put(key):
            if key not in cbc_values:
                cbc_values[key] = row

        # Red cells
        if name in ("hb", "hgb") or "haemoglobin" in name or "hemoglobin" in name:
            put("Hb")
        elif name.startswith("mcv"):
            put("MCV")
        elif name.startswith("mch"):
            put("MCH")
        elif name.startswith("mchc"):
            put("MCHC")
        elif "red cell" in name or "rbc" in name:
            put("RBC")
        elif "hct" in name or "haematocrit" in name or "hematocrit" in name:
            put("Hct")

        # White cells
        elif "white cell" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            put("WBC")
        elif "neutrophil" in name:
            put("Neutrophils")
        elif "lymphocyte" in name:
            put("Lymphocytes")
        elif "eosinophil" in name:
            put("Eosinophils")
        elif "monocyte" in name:
            put("Monocytes")
        elif "basophil" in name:
            put("Basophils")

        # Platelets
        elif "platelet" in name or "plt" in name:
            put("Platelets")

        # Inflammation
        elif name.startswith("crp") or "c-reactive" in name:
            put("CRP")
        elif "esr" in name or "sedimentation" in name:
            put("ESR")

        # Renal
        elif "creatinine" in name:
            put("Creatinine")
        elif name.startswith("urea"):
            put("Urea")
        elif "egfr" in name:
            put("eGFR")

        # LFT
        elif name == "alt" or "alanine aminotransferase" in name:
            put("ALT")
        elif name == "ast" or "aspartate aminotransferase" in name:
            put("AST")
        elif name == "alp" or "alkaline phosphatase" in name:
            put("ALP")
        elif "ggt" in name or "gamma glutamyl" in name:
            put("GGT")
        elif "bilirubin" in name:
            put("Bilirubin")
        elif "albumin" in name:
            put("Albumin")

        # Muscle / cardiac
        elif name == "ck" or "creatine kinase" in name:
            put("CK")
        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")
        elif "troponin" in name:
            put("Troponin")

        # Electrolytes
        elif "sodium" in name or name == "na":
            put("Sodium")
        elif "potassium" in name or name == "k":
            put("Potassium")
        elif "chloride" in name or name in ("cl", "cl-"):
            put("Chloride")
        elif "bicarbonate" in name or "hco3" in name or "co2" in name:
            put("Bicarbonate")
        elif "calcium" in name or name.startswith("ca"):
            put("Calcium")
        elif "magnesium" in name or name.startswith("mg"):
            put("Magnesium")

    return cbc_values


# ======================================================
#   CBC + CHEMISTRY ROUTE ENGINE (V2, HYBRID STYLE C)
# ======================================================

def add_route_block(routes, title, findings=None, interpretation=None, actions=None):
    """Helper: build one hybrid route block."""
    block_lines = [f"ROUTE: {title}"]

    if findings:
        block_lines.append("Key findings:")
        for f in findings:
            block_lines.append(f"• {f}")

    if interpretation:
        block_lines.append("Interpretation:")
        block_lines.append(interpretation.strip())

    if actions:
        block_lines.append("Suggested next steps:")
        for a in actions:
            block_lines.append(f"• {a}")

    routes.append("\n".join(block_lines).strip())


def generate_clinical_routes(cbc_values: dict):
    """
    Takes parsed CBC + chemistry values → generates hybrid routes (Style C):
    - Short bullet structure
    - With expanded explanation + suggested next steps
    """

    routes = []

    def val(name):
        return clean_number(cbc_values.get(name, {}).get("value"))

    # CBC core
    Hb   = val("Hb")
    Hct  = val("Hct")
    MCV  = val("MCV")
    MCH  = val("MCH")
    WBC  = val("WBC")
    RBC  = val("RBC")
    Plt  = val("Platelets")

    Neut = val("Neutrophils")
    Lymph = val("Lymphocytes")
    Eos  = val("Eosinophils")
    Mono = val("Monocytes")

    # Chemistry
    Cr   = val("Creatinine")
    Urea = val("Urea")
    eGFR = val("eGFR")

    ALT  = val("ALT")
    AST  = val("AST")
    ALP  = val("ALP")
    GGT  = val("GGT")
    Bili = val("Bilirubin")
    Alb  = val("Albumin")

    CK   = val("CK")
    CKMB = val("CK-MB")
    Trop = val("Troponin")

    Na   = val("Sodium")
    K    = val("Potassium")
    Cl   = val("Chloride")
    HCO3 = val("Bicarbonate")

    Ca   = val("Calcium")
    Mg   = val("Magnesium")

    CRP  = val("CRP")
    ESR  = val("ESR")

    # Helper: rough flags
    def low(v, thr):
        return v is not None and v < thr

    def high(v, thr):
        return v is not None and v > thr

    # -----------------------------
    # 1. ANAEMIA / RED CELL ROUTES
    # -----------------------------
    if Hb is not None and Hb < 13:
        # Rough classification by MCV
        if MCV is not None and MCV < 80:
            add_route_block(
                routes,
                "Microcytic anaemia",
                findings=[
                    f"Hb low (~{Hb} g/dL)",
                    f"MCV low (~{MCV} fL)",
                ],
                interpretation=(
                    "This pattern is consistent with a microcytic anaemia, most often seen with "
                    "iron deficiency or chronic inflammatory states. Thalassaemia traits can also "
                    "show microcytosis with relatively preserved Hb."
                ),
                actions=[
                    "Order ferritin (first-line test for iron deficiency).",
                    "If ferritin is normal or high, consider CRP/ESR to look for inflammatory masking.",
                    "If strong suspicion of thalassaemia trait, consider haemoglobin electrophoresis.",
                    "Check reticulocyte count to assess marrow response."
                ],
            )
        elif MCV is not None and 80 <= MCV <= 100:
            add_route_block(
                routes,
                "Normocytic anaemia",
                findings=[
                    f"Hb low (~{Hb} g/dL)",
                    f"MCV in normal range (~{MCV} fL)",
                ],
                interpretation=(
                    "Normocytic anaemia is often associated with chronic disease, renal impairment, "
                    "acute blood loss or early iron deficiency before microcytosis appears."
                ),
                actions=[
                    "Review renal function (urea, creatinine, eGFR) and chronic disease history.",
                    "Check CRP/ESR for chronic inflammatory states.",
                    "Consider reticulocyte count to distinguish underproduction vs blood loss/haemolysis.",
                    "Correlate with medication history and recent bleeding."
                ],
            )
        elif MCV is not None and MCV > 100:
            add_route_block(
                routes,
                "Macrocytic anaemia",
                findings=[
                    f"Hb low (~{Hb} g/dL)",
                    f"MCV raised (~{MCV} fL)",
                ],
                interpretation=(
                    "Macrocytic anaemia can occur with B12 or folate deficiency, alcohol use, liver disease, "
                    "hypothyroidism, or certain medications (e.g. methotrexate, AZT)."
                ),
                actions=[
                    "Check vitamin B12 and folate levels.",
                    "Review liver function tests, thyroid function and medication history.",
                    "Consider reticulocyte count and blood film if concern for marrow pathology.",
                ],
            )

    # Pancytopenia / bicytopenia pattern
    if (
        Hb is not None and Hb < 11 and
        WBC is not None and WBC < 3.5 and
        Plt is not None and Plt < 150
    ):
        add_route_block(
            routes,
            "Pancytopenia pattern",
            findings=[
                f"Hb low (~{Hb})",
                f"WBC low (~{WBC})",
                f"Platelets low (~{Plt})",
            ],
            interpretation=(
                "Concurrent depression of red cells, white cells and platelets suggests a pancytopenia pattern. "
                "This can be seen in marrow failure, infiltrative disease, severe sepsis, nutritional deficiency "
                "or drug/toxin effects."
            ),
            actions=[
                "Urgently correlate with clinical status (fever, bleeding, infection).",
                "Consider urgent blood film and haematology opinion.",
                "Review medications, alcohol, nutritional status and viral history.",
            ],
        )

    # -----------------------------
    # 2. WHITE CELL + DIFFERENTIAL
    # -----------------------------
    if WBC is not None:
        if WBC > 12:
            findings = [f"WBC raised (~{WBC} x10^9/L)"]
            if Neut is not None and Neut > 70:
                findings.append(f"Neutrophils high (~{Neut}%)")
            if Lymph is not None and Lymph > 45:
                findings.append(f"Lymphocytes high (~{Lymph}%)")

            interp_lines = []
            interp_lines.append(
                "Raised white cell count is consistent with an inflammatory or infective process."
            )
            if Neut is not None and Neut > 70:
                interp_lines.append(
                    "Neutrophil predominance often fits a bacterial or acute stress pattern."
                )
            if Lymph is not None and Lymph > 45:
                interp_lines.append(
                    "Lymphocyte predominance can be seen with viral infections or recovery phases."
                )

            actions = [
                "Correlate with symptoms (fever, localising signs, chest/urine/abdominal symptoms).",
                "Consider CRP and procalcitonin (if available) for further inflammatory assessment.",
                "Review medications (steroids, adrenergic agents) that may raise WBC.",
            ]

            add_route_block(
                routes,
                "Leukocytosis / inflammatory route",
                findings=findings,
                interpretation=" ".join(interp_lines),
                actions=actions,
            )
        elif WBC < 4:
            add_route_block(
                routes,
                "Leukopenia route",
                findings=[f"WBC low (~{WBC} x10^9/L)"],
                interpretation=(
                    "Low white cell count can reflect viral suppression, drug effects, marrow dysfunction, "
                    "autoimmune disease or severe sepsis (late)."
                ),
                actions=[
                    "Review recent viral illness, medications (e.g. chemotherapy, immunosuppressants).",
                    "Consider HIV testing where appropriate.",
                    "If associated with fever or severe symptoms, treat as potentially serious and escalate.",
                ],
            )

    if Eos is not None and Eos > 5:
        add_route_block(
            routes,
            "Eosinophilia route",
            findings=[f"Eosinophils raised (~{Eos}%)"],
            interpretation=(
                "Eosinophilia can occur with allergy, asthma, parasitic infection, drug reactions or certain "
                "autoimmune and haematological conditions."
            ),
            actions=[
                "Review allergic history, asthma, eczema and medication changes.",
                "Consider stool microscopy or parasitic screen if clinically indicated.",
                "If persistent or marked, consider further haematological evaluation.",
            ],
        )

    # -----------------------------
    # 3. PLATELET ROUTES
    # -----------------------------
    if Plt is not None:
        if Plt < 150:
            add_route_block(
                routes,
                "Thrombocytopenia route",
                findings=[f"Platelets low (~{Plt} x10^9/L)"],
                interpretation=(
                    "Low platelets increase bleeding risk and can be due to reduced production, increased destruction, "
                    "splenic sequestration, infection or drugs."
                ),
                actions=[
                    "Assess for bruising, petechiae, mucosal bleeding.",
                    "Review recent infections, medications and alcohol use.",
                    "Consider blood film, HIV testing and haematology review if significantly low.",
                ],
            )
        elif Plt > 450:
            add_route_block(
                routes,
                "Thrombocytosis route",
                findings=[f"Platelets raised (~{Plt} x10^9/L)"],
                interpretation=(
                    "Raised platelets are often reactive (infection, inflammation, iron deficiency, post-surgery) but "
                    "can rarely reflect a myeloproliferative process."
                ),
                actions=[
                    "Correlate with CRP/ESR, iron studies and recent surgery/trauma.",
                    "If persistent without clear cause, consider haematology opinion.",
                ],
            )

    # -----------------------------
    # 4. KIDNEY FUNCTION ROUTES
    # -----------------------------
    if Cr is not None or eGFR is not None or Urea is not None:
        findings = []
        if Cr is not None:
            findings.append(f"Creatinine ~{Cr}")
        if eGFR is not None:
            findings.append(f"eGFR ~{eGFR} mL/min/1.73m²")
        if Urea is not None:
            findings.append(f"Urea ~{Urea}")

        interpretation = (
            "Kidney function should be interpreted in context of baseline eGFR, age, hydration and medications."
        )

        if Cr is not None and Cr > 120:
            interpretation += " Raised creatinine suggests possible renal impairment or acute kidney injury."
        if eGFR is not None and eGFR < 60:
            interpretation += " eGFR <60 mL/min suggests at least moderate chronic kidney disease if persistent >3 months."

        actions = [
            "Compare with previous kidney function to distinguish acute vs chronic change.",
            "Review nephrotoxic drugs (NSAIDs, ACE-inhibitors, diuretics, contrast).",
            "Check blood pressure, fluid status and urine dipstick.",
            "Consider renal ultrasound or nephrology input if progressive or severe.",
        ]

        add_route_block(
            routes,
            "Renal function route",
            findings=findings,
            interpretation=interpretation,
            actions=actions,
        )

    # -----------------------------
    # 5. LIVER FUNCTION ROUTES
    # -----------------------------
    if any(x is not None for x in (ALT, AST, ALP, GGT, Bili)):
        findings = []
        if ALT is not None:
            findings.append(f"ALT ~{ALT}")
        if AST is not None:
            findings.append(f"AST ~{AST}")
        if ALP is not None:
            findings.append(f"ALP ~{ALP}")
        if GGT is not None:
            findings.append(f"GGT ~{GGT}")
        if Bili is not None:
            findings.append(f"Bilirubin ~{Bili}")
        if Alb is not None:
            findings.append(f"Albumin ~{Alb}")

        interpretation = (
            "Liver tests can show hepatocellular (ALT/AST predominant) vs cholestatic (ALP/GGT, bilirubin) patterns."
        )

        if ALT is not None and ALT > 3 * 40:
            interpretation += " Marked ALT elevation suggests hepatocellular injury (e.g. viral hepatitis, toxins, ischaemia)."
        if ALP is not None and ALP > 120:
            interpretation += " Raised ALP fits a cholestatic or bone-related pattern."
        if Bili is not None and Bili > 20:
            interpretation += " Raised bilirubin warrants evaluation for haemolysis, hepatic or obstructive causes."

        actions = [
            "Correlate with symptoms (jaundice, RUQ pain, pruritus, systemic features).",
            "Review alcohol intake, viral risk factors and medication/toxin exposure.",
            "Consider abdominal ultrasound if cholestatic or obstructive pattern suspected.",
            "Check INR and albumin if concern for synthetic dysfunction.",
        ]

        add_route_block(
            routes,
            "Liver function route",
            findings=findings,
            interpretation=interpretation,
            actions=actions,
        )

    # -----------------------------
    # 6. MUSCLE / RHABDO / CARDIAC ROUTES
    # -----------------------------
    if CK is not None and CK > 300:
        findings = [f"CK raised (~{CK})"]
        if Cr is not None:
            findings.append(f"Creatinine ~{Cr}")

        interpretation = (
            "Raised CK indicates muscle injury, which may be due to exercise, trauma, seizures, medications or "
            "inflammatory muscle disease."
        )
        if CK > 2000:
            interpretation += " Very high CK raises concern for rhabdomyolysis physiology and risk of kidney injury."

        actions = [
            "Ask about trauma, seizures, prolonged immobilisation, statins or other myotoxic drugs.",
            "Monitor renal function and urine output if CK markedly elevated.",
            "Encourage adequate hydration as clinically appropriate.",
        ]

        add_route_block(
            routes,
            "Muscle injury / rhabdomyolysis route",
            findings=findings,
            interpretation=interpretation,
            actions=actions,
        )

    if Trop is not None:
        findings = [f"Troponin ~{Trop}"]
        interpretation = (
            "Troponin elevation is specific for myocardial injury but not specific for cause "
            "(e.g. acute coronary syndrome, tachyarrhythmia, myocarditis, demand ischaemia, sepsis)."
        )
        actions = [
            "Always correlate troponin pattern with chest pain history, ECG changes and risk factors.",
            "Consider serial troponins to assess rise/fall pattern.",
            "Follow local ACS and chest pain protocols for urgent management.",
        ]
        add_route_block(
            routes,
            "Myocardial injury / troponin route",
            findings=findings,
            interpretation=interpretation,
            actions=actions,
        )

    # -----------------------------
    # 7. ELECTROLYTES ROUTES
    # -----------------------------
    if K is not None:
        if K < 3.3:
            add_route_block(
                routes,
                "Hypokalaemia route",
                findings=[f"Potassium low (~{K} mmol/L)"],
                interpretation=(
                    "Low potassium increases the risk of cardiac arrhythmia and can be due to diuretics, vomiting, "
                    "diarrhoea, endocrine causes or inadequate intake."
                ),
                actions=[
                    "Assess for symptoms (weakness, palpitations) and check ECG if significantly low.",
                    "Review diuretics and other medications affecting potassium.",
                    "Correct potassium under appropriate monitoring and local protocols.",
                ],
            )
        elif K > 5.5:
            add_route_block(
                routes,
                "Hyperkalaemia route",
                findings=[f"Potassium high (~{K} mmol/L)"],
                interpretation=(
                    "High potassium can cause life-threatening arrhythmias and may result from renal impairment, "
                    "medications (ACE-inhibitors, ARBs, spironolactone), acidosis or cell breakdown."
                ),
                actions=[
                    "Treat as time-sensitive if K is markedly raised or ECG changes present.",
                    "Review kidney function and medications that raise potassium.",
                    "Follow local hyperkalaemia treatment protocol.",
                ],
            )

    if Na is not None:
        if Na < 133:
            add_route_block(
                routes,
                "Hyponatraemia route",
                findings=[f"Sodium low (~{Na} mmol/L)"],
                interpretation=(
                    "Hyponatraemia is common and may be related to excess free water, SIADH, heart failure, "
                    "cirrhosis, endocrine disorders or medications."
                ),
                actions=[
                    "Evaluate volume status (hypovolaemic, euvolaemic, hypervolaemic).",
                    "Review medications (diuretics, antidepressants, anticonvulsants).",
                    "Avoid rapid correction; follow local hyponatraemia guidelines.",
                ],
            )
        elif Na > 146:
            add_route_block(
                routes,
                "Hypernatraemia route",
                findings=[f"Sodium high (~{Na} mmol/L)"],
                interpretation=(
                    "Hypernatraemia usually reflects water loss greater than sodium loss (dehydration) or excess "
                    "sodium intake/administration."
                ),
                actions=[
                    "Assess for dehydration, impaired thirst or limited access to water.",
                    "Correct sodium gradually with appropriate fluids according to local protocol.",
                ],
            )

    # -----------------------------
    # 8. INFLAMMATION ROUTES (CRP / ESR)
    # -----------------------------
    if CRP is not None:
        if CRP > 10:
            add_route_block(
                routes,
                "Inflammatory marker route",
                findings=[f"CRP raised (~{CRP} mg/L)"],
                interpretation=(
                    "Raised CRP supports the presence of active inflammation or infection but is non-specific "
                    "regarding location or cause."
                ),
                actions=[
                    "Correlate with clinical features and other markers (WBC, neutrophils, temperature).",
                    "Use CRP trend, not a single value, to follow response to treatment where appropriate.",
                ],
            )
        else:
            add_route_block(
                routes,
                "CRP low/normal route",
                findings=[f"CRP not raised (~{CRP} mg/L)"],
                interpretation=(
                    "A normal or low CRP makes significant systemic inflammation less likely but does not fully exclude "
                    "early infection or localised pathology."
                ),
                actions=[
                    "Interpret in context of symptom duration and immune status.",
                    "Consider repeating if clinical suspicion remains high.",
                ],
            )

    # -----------------------------
    # Fallback: No major routes
    # -----------------------------
    if not routes:
        routes.append(
            "ROUTE: No major abnormalities\n"
            "Interpretation:\n"
            "From the available values, no strong route-level abnormalities were triggered.\n"
            "Suggested next steps:\n"
            "• Correlate with the full clinical picture and any previous results.\n"
            "• Repeat testing if new symptoms develop or if there is ongoing concern."
        )

    return routes


# ======================================================
#   CORE JOB PROCESSING
# ======================================================

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

        # Download PDF
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        print(f"Report {report_id}: scanned={scanned}, text_length={len(text)}")

        # ------- SCANNED PDF → OCR TABLE -------
        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → OCR pipeline")
            images = convert_from_bytes(pdf_bytes)
            combined_rows = []

            for idx, img in enumerate(images):
                img_bytes_io = io.BytesIO()
                img.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                try:
                    ocr_result = extract_cbc_from_image(img_bytes)
                    if isinstance(ocr_result, dict) and "cbc" in ocr_result:
                        combined_rows.extend(ocr_result["cbc"])
                    else:
                        print(f"OCR page {idx} returned no 'cbc' key.")
                except Exception as e:
                    print("Vision OCR error:", e)

            if not combined_rows:
                raise ValueError("Vision OCR could not extract CBC values")

            # Pass structured CBC JSON as text to interpreter
            merged_text = json.dumps({"cbc": combined_rows}, ensure_ascii=False)

        # ------- DIGITAL PDF → DIRECT TEXT -------
        else:
            print(f"📝 Report {report_id}: Digital PDF detected → text interpreter")
            merged_text = text or l_text

        if not merged_text.strip():
            raise ValueError("No usable content extracted for AI processing")

        # ------- MAIN INTERPRETATION -------
        ai_json = call_ai_on_report(merged_text)

        # ------- ROUTE ENGINE V2 (Hybrid) -------
        try:
            cbc_values = build_cbc_value_dict(ai_json)
            routes = generate_clinical_routes(cbc_values)
            ai_json["routes"] = routes
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


# ======================================================
#   WORKER LOOP
# ======================================================

def main():
    print(">>> Entering main worker loop…")
    print("AMI Worker with Vision OCR + Route Engine V2 started – watching for jobs.")

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
