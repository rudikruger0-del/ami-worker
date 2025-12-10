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
    import re
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
#   ENV / CLIENTS
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
#   HELPER: SAFE JSON FROM OPENAI MESSAGE
# ======================================================

def _extract_json_from_message(msg) -> dict:
    """
    Handles different OpenAI SDK shapes:
    - message.parsed (new)
    - message.content as JSON string
    - message.content as list of {type,text}
    """
    # Newer SDK: .parsed when response_format={"type":"json_object"}
    parsed = getattr(msg, "parsed", None)
    if parsed is not None:
        return parsed

    content = getattr(msg, "content", None)

    if isinstance(content, str):
        if not content.strip():
            return {}
        return json.loads(content)

    if isinstance(content, list):
        # concatenate any text parts
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and "text" in part:
                    parts.append(part["text"])
            else:
                parts.append(str(part))
        text = "".join(parts).strip()
        if not text:
            return {}
        return json.loads(text)

    if content is None:
        return {}

    # last-resort: try to json-load whatever representation
    return json.loads(str(content))


def openai_json(model: str, messages: list, temperature: float = 0.1) -> dict:
    """Wrapper around OpenAI Chat Completions that always returns a JSON object."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    return _extract_json_from_message(resp.choices[0].message)


# ======================================================
#   PDF TEXT EXTRACTION
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
    """If there’s almost no text, assume scanned PDF."""
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 30:
        return True
    return False


# ======================================================
#   IMAGE OCR → CBC JSON
# ======================================================

def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Sends a PNG page to OpenAI Vision to extract CBC values.
    Returns dict: { "cbc": [ { analyte, value, units, reference_low, reference_high } ] }
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    system_prompt = (
        "You are an OCR and data extraction assistant for medical laboratory PDF scans.\n"
        "Extract ALL blood-related analytes including CBC and chemistry (electrolytes, renal, "
        "liver, CRP, CK, CK-MB where visible).\n\n"
        "Return STRICT JSON ONLY of this shape:\n"
        "{\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": \"\",\n"
        "      \"value\": \"\",   // as printed on report\n"
        "      \"units\": \"\",\n"
        "      \"reference_low\": \"\",\n"
        "      \"reference_high\": \"\",\n"
        "      \"flag\": \"high\" | \"low\" | \"normal\" | \"\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Do not add any other top-level fields. If uncertain, still include the analyte with an empty value.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Extract all analytes and values from this laboratory report image.",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    },
                },
            ],
        },
    ]

    data = openai_json("gpt-4o", messages)
    if not isinstance(data, dict):
        return {}
    return data


# ======================================================
#   MAIN INTERPRETATION MODEL (TEXT)
# ======================================================

def call_ai_on_report(text: str) -> dict:
    """
    Asks GPT to interpret the results and output:
      - patient (optional)
      - cbc (array of analytes, including chemistry)
      - summary.impression
      - summary.suggested_follow_up
    """
    MAX_CHARS = 12000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    system_prompt = (
        "You are an assistive clinical tool analysing complete blood count (CBC) and related "
        "chemistry results. You DO NOT diagnose. You only describe patterns and safe next steps.\n\n"
        "Input: raw lab report text or JSON from OCR.\n\n"
        "Output STRICT JSON with this structure:\n"
        "{\n"
        "  \"patient\": {\n"
        "    \"name\": \"\",\n"
        "    \"age\": null,\n"
        "    \"sex\": \"\"\n"
        "  },\n"
        "  \"cbc\": [\n"
        "    {\n"
        "      \"analyte\": \"\",\n"
        "      \"value\": \"\",\n"
        "      \"units\": \"\",\n"
        "      \"reference_low\": \"\",\n"
        "      \"reference_high\": \"\",\n"
        "      \"flag\": \"high\" | \"low\" | \"normal\" | \"\"\n"
        "    }\n"
        "  ],\n"
        "  \"summary\": {\n"
        "    \"impression\": \"A few short paragraphs describing the main lab patterns. "
        "Use language like 'may suggest', 'is consistent with', 'can indicate'. "
        "Avoid diagnostic labels such as 'iron deficiency anaemia' without qualifiers.\",\n"
        "    \"suggested_follow_up\": \"Bullet-style text describing safe follow-up: "
        "repeat labs, consider ferritin/reticulocytes/CRP/ESR/renal/liver tests, clinical correlation, "
        "and when to seek urgent review. No medication or treatment advice.\"\n"
        "  }\n"
        "}\n"
        "Do NOT add any other top-level fields. Never mention drug names or prescribe treatment.\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    data = openai_json("gpt-4o-mini", messages)
    if not isinstance(data, dict):
        return {}
    return data


# ======================================================
#   VALUE NORMALISATION HELPERS
# ======================================================

def normalise_number(raw):
    """
    Turn things like '88.0%', '13.4*10^9/L', ' <5 ', '7,200' into a float.
    Returns None if no numeric part found.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # remove thousands separators
    s = s.replace(",", "")
    # find first numeric substring
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def build_value_maps(cbc_list):
    """
    From ai_json['cbc'] (list of rows) build:
      - canonical_map: { key -> numeric_value }
      - raw_map:       { key -> row_dict }
    Keys are canonical biological concepts like 'hb', 'wbc', 'crp', 'na', 'creatinine', etc.
    """
    canonical_map = {}
    raw_map = {}

    if not isinstance(cbc_list, list):
        return canonical_map, raw_map

    # Define simple string matching rules
    mapping = {
        "hb": ["hb", "haemoglobin", "hemoglobin"],
        "hct": ["hct", "haematocrit", "hematocrit", "packed cell"],
        "rbc": ["rbc", "erythrocyte"],
        "wbc": ["wbc", "white cell", "leucocyte", "leukocyte"],
        "plt": ["platelet", "plt"],
        "mcv": ["mcv"],
        "mch": ["mch"],
        "mchc": ["mchc"],
        "rdw": ["rdw"],
        "neut_abs": ["neutrophil", "neut "],
        "lymph_abs": ["lymphocyte", "lymph "],
        "mono_abs": ["monocyte", "mono "],
        "eos_abs": ["eosinophil", "eos "],
        "baso_abs": ["basophil", "baso "],
        "neut_pct": ["neut %", "neutrophils %"],
        "lymph_pct": ["lymph %", "lymphocytes %"],
        "eos_pct": ["eos %", "eosinophils %"],
        "crp": ["crp", "c-reactive protein"],
        "esr": ["esr", "erythrocyte sedimentation"],
        "ferritin": ["ferritin"],
        "na": ["sodium", "na "],
        "k": ["potassium", "k "],
        "cl": ["chloride", "cl "],
        "bicarb": ["t-co2", "bicarbonate", "hco3", "co2"],
        "urea": ["urea", "bun"],
        "creatinine": ["creatinine"],
        "glucose": ["glucose"],
        "alt": ["alt", "alanine aminotransferase"],
        "ast": ["ast", "aspartate aminotransferase"],
        "alp": ["alkaline phosphatase", "alp"],
        "ggt": ["ggt", "gamma gt"],
        "bilirubin": ["bilirubin"],
        "albumin": ["albumin"],
    }

    for row in cbc_list:
        name = str(row.get("analyte", "")).lower()
        val = normalise_number(row.get("value"))
        for key, patterns in mapping.items():
            if key in canonical_map:
                continue  # keep first match
            for pat in patterns:
                if pat in name:
                    canonical_map[key] = val
                    raw_map[key] = row
                    break

    return canonical_map, raw_map


def make_evidence(label, row):
    if not row:
        return None
    v = row.get("value", "")
    u = row.get("units") or ""
    ref_low = row.get("reference_low") or ""
    ref_high = row.get("reference_high") or ""
    parts = [f"{label}: {v} {u}".strip()]
    if ref_low or ref_high:
        parts.append(f"(ref {ref_low}–{ref_high})".strip())
    return " ".join(parts)


# ======================================================
#   ROUTE ENGINE
# ======================================================

def generate_clinical_routes(ai_json: dict, job: dict) -> list:
    """
    Build a list of 'routes' – structured clinical decision paths based on CBC & chemistry.
    This is what makes AMI feel like a 'roete' engine.
    """
    routes = []

    cbc_list = ai_json.get("cbc") or []
    canonical, raw_map = build_value_maps(cbc_list)

    age = job.get("age") or ai_json.get("patient", {}).get("age")
    sex = (job.get("sex") or ai_json.get("patient", {}).get("sex") or "").lower()

    # Convenience getters
    g = lambda k: canonical.get(k)

    Hb = g("hb")
    MCV = g("mcv")
    MCH = g("mch")
    MCHC = g("mchc")
    RDW = g("rdw")
    WBC = g("wbc")
    Neut = g("neut_abs")
    Lymph = g("lymph_abs")
    Eos = g("eos_abs")
    Plt = g("plt")
    CRP = g("crp")
    Urea = g("urea")
    Creat = g("creatinine")
    Na = g("na")
    K = g("k")
    Bicarb = g("bicarb")
    AST = g("ast")
    ALT = g("alt")
    ALP = g("alp")
    GGT = g("ggt")
    Bili = g("bilirubin")

    # --------------------------------------------------
    # 1) ANAEMIA ROUTES
    # --------------------------------------------------
    if Hb is not None and Hb < 12.0:  # generic adult-ish threshold
        anaemia_evidence = []
        ev_hb = make_evidence("Haemoglobin", raw_map.get("hb"))
        if ev_hb:
            anaemia_evidence.append(ev_hb)

        # Microcytic pattern → possible iron deficiency
        if MCV is not None and MCV < 80:
            ev_mcv = make_evidence("MCV", raw_map.get("mcv"))
            ev_mch = make_evidence("MCH", raw_map.get("mch"))
            ev_rdw = make_evidence("RDW", raw_map.get("rdw"))
            for ev in [ev_mcv, ev_mch, ev_rdw]:
                if ev:
                    anaemia_evidence.append(ev)

            routes.append({
                "name": "Microcytic pattern – possible iron deficiency route",
                "category": "anaemia",
                "priority": "high" if Hb < 9.0 else "moderate",
                "pattern": "Low haemoglobin with low MCV and MCH, often with raised RDW.",
                "evidence": anaemia_evidence,
                "suggested_follow_up": [
                    "Consider iron studies (serum ferritin, transferrin saturation, serum iron).",
                    "Consider reticulocyte count to assess bone marrow response.",
                    "Screen for chronic blood loss (e.g. gastrointestinal, heavy menstrual bleeding) based on history.",
                    "Correlate with diet, pregnancy status, and chronic disease.",
                ],
                "notes": [
                    "Pattern is consistent with iron-restricted erythropoiesis but is not diagnostic.",
                    "Use local reference ranges and clinical context.",
                ],
            })

        # Macrocytic pattern → B12 / folate / liver / thyroid
        elif MCV is not None and MCV > 100:
            ev_mcv = make_evidence("MCV", raw_map.get("mcv"))
            if ev_mcv:
                anaemia_evidence.append(ev_mcv)

            routes.append({
                "name": "Macrocytic pattern – B12/folate/liver/thyroid route",
                "category": "anaemia",
                "priority": "moderate",
                "pattern": "Low haemoglobin with raised MCV.",
                "evidence": anaemia_evidence,
                "suggested_follow_up": [
                    "Consider vitamin B12 and folate levels.",
                    "Review liver function tests and thyroid function where clinically indicated.",
                    "Review alcohol intake, medication history (e.g. some chemotherapeutic or antiepileptic agents).",
                ],
                "notes": [
                    "Macrocytosis can have many causes; this route only highlights common patterns.",
                ],
            })

        # Normocytic → chronic disease / renal / mixed
        else:
            # Normocytic range ~80–100
            routes.append({
                "name": "Normocytic anaemia – chronic disease / renal route",
                "category": "anaemia",
                "priority": "moderate",
                "pattern": "Low haemoglobin with MCV in the normal range.",
                "evidence": anaemia_evidence,
                "suggested_follow_up": [
                    "Correlate with markers of inflammation (CRP, ESR) and chronic disease.",
                    "Review renal function (urea, creatinine, eGFR) for anaemia of chronic kidney disease.",
                    "Check reticulocyte count if available (low reticulocytes suggest under-production).",
                ],
                "notes": [
                    "Pattern is often seen in anaemia of chronic disease or mixed causes.",
                ],
            })

    # --------------------------------------------------
    # 2) WHITE CELL / INFECTION / INFLAMMATION ROUTES
    # --------------------------------------------------

    infection_evidence = []

    # Bacterial / neutrophilic pattern
    if (Neut is not None and Neut > 7.5) or (WBC is not None and WBC > 11) or (CRP is not None and CRP > 10):
        if raw_map.get("wbc"):
            ev_wbc = make_evidence("WBC", raw_map.get("wbc"))
            if ev_wbc:
                infection_evidence.append(ev_wbc)
        if raw_map.get("neut_abs"):
            ev_neut = make_evidence("Neutrophils", raw_map.get("neut_abs"))
            if ev_neut:
                infection_evidence.append(ev_neut)
        if raw_map.get("crp"):
            ev_crp = make_evidence("CRP", raw_map.get("crp"))
            if ev_crp:
                infection_evidence.append(ev_crp)

        routes.append({
            "name": "Neutrophilia / raised CRP – possible bacterial infection route",
            "category": "infection_inflammation",
            "priority": "high" if (CRP is not None and CRP > 100) or (WBC is not None and WBC > 20) else "moderate",
            "pattern": "Raised CRP and neutrophil-predominant leukocytosis may indicate acute bacterial infection or significant inflammation.",
            "evidence": infection_evidence,
            "suggested_follow_up": [
                "Correlate with fever, localising symptoms and examination.",
                "Consider targeted cultures or imaging if there is suspicion of focal infection.",
                "Trend CRP and WBC over time to monitor response.",
            ],
            "notes": [
                "Inflammatory markers are non-specific and must be interpreted in full clinical context.",
            ],
        })

    # Viral / lymphocytic pattern
    if Lymph is not None and Lymph > 4.0:
        ev_ly = make_evidence("Lymphocytes", raw_map.get("lymph_abs"))
        routes.append({
            "name": "Lymphocytosis – possible viral / reactive route",
            "category": "infection_inflammation",
            "priority": "moderate",
            "pattern": "Raised lymphocyte count may be seen in viral infections or reactive states.",
            "evidence": [ev_ly] if ev_ly else [],
            "suggested_follow_up": [
                "Correlate with recent viral symptoms (upper respiratory, glandular fever-like illness, etc.).",
                "If persistent or marked lymphocytosis, consider peripheral smear or haematology review.",
            ],
            "notes": [
                "Short-term lymphocytosis is often benign; persistent or extreme changes warrant further assessment.",
            ],
        })

    # Eosinophilia route
    if Eos is not None and Eos > 0.5:
        ev_eos = make_evidence("Eosinophils", raw_map.get("eos_abs"))
        routes.append({
            "name": "Eosinophilia route – allergy / parasitic / drug reaction",
            "category": "infection_inflammation",
            "priority": "moderate",
            "pattern": "Raised eosinophils may be associated with allergy, asthma, parasitic infection or drug reactions.",
            "evidence": [ev_eos] if ev_eos else [],
            "suggested_follow_up": [
                "Review history for atopy, asthma, new medications and travel.",
                "Consider stool studies or serology for parasites if clinically suspected.",
            ],
            "notes": [
                "Marked or persistent eosinophilia may require specialist input.",
            ],
        })

    # Leukopenia / neutropenia
    if WBC is not None and WBC < 4.0:
        ev_w = make_evidence("WBC", raw_map.get("wbc"))
        routes.append({
            "name": "Leukopenia route",
            "category": "white_cells",
            "priority": "high" if WBC < 2.0 else "moderate",
            "pattern": "Low white cell count may increase infection risk.",
            "evidence": [ev_w] if ev_w else [],
            "suggested_follow_up": [
                "Review recent viral illnesses, medications (e.g. chemotherapy, immunosuppressants).",
                "If severe or persistent, discuss with haematology.",
            ],
            "notes": [
                "Neutrophil count is usually the key parameter for infection risk; correlate with absolute neutrophil count if available.",
            ],
        })

    if Neut is not None and Neut < 1.5:
        ev_n = make_evidence("Neutrophils", raw_map.get("neut_abs"))
        routes.append({
            "name": "Neutropenia route",
            "category": "white_cells",
            "priority": "high" if Neut < 0.5 else "moderate",
            "pattern": "Low neutrophil count significantly increases risk of bacterial infection.",
            "evidence": [ev_n] if ev_n else [],
            "suggested_follow_up": [
                "If febrile and significantly neutropenic, follow local neutropenic sepsis guidelines.",
                "Review recent chemotherapy or medications and consider specialist input.",
            ],
            "notes": [
                "Management must follow local emergency protocols; this route only highlights the pattern.",
            ],
        })

    # --------------------------------------------------
    # 3) PLATELET ROUTES
    # --------------------------------------------------
    if Plt is not None and Plt < 150:
        ev_p = make_evidence("Platelets", raw_map.get("plt"))
        routes.append({
            "name": "Thrombocytopenia route",
            "category": "platelets",
            "priority": "high" if Plt < 50 else "moderate",
            "pattern": "Low platelet count can increase bleeding risk.",
            "evidence": [ev_p] if ev_p else [],
            "suggested_follow_up": [
                "Review for bruising, mucosal bleeding, recent infections or medications.",
                "Check for liver disease, splenomegaly, and consider repeat count or smear.",
                "Follow local guidelines for platelet thresholds before procedures.",
            ],
            "notes": [
                "Sudden or severe thrombocytopenia should prompt urgent review.",
            ],
        })

    if Plt is not None and Plt > 450:
        ev_p = make_evidence("Platelets", raw_map.get("plt"))
        routes.append({
            "name": "Thrombocytosis route",
            "category": "platelets",
            "priority": "moderate",
            "pattern": "Raised platelet count may be reactive (infection, inflammation, iron deficiency) or primary.",
            "evidence": [ev_p] if ev_p else [],
            "suggested_follow_up": [
                "Look for infection, iron deficiency and inflammatory conditions.",
                "If persistently very high or unexplained, consider haematology review.",
            ],
            "notes": [
                "Most thrombocytosis is reactive but persistent extreme values warrant further workup.",
            ],
        })

    # --------------------------------------------------
    # 4) RENAL / DEHYDRATION / ELECTROLYTES
    # --------------------------------------------------
    renal_evidence = []
    if raw_map.get("urea"):
        renal_evidence.append(make_evidence("Urea", raw_map.get("urea")))
    if raw_map.get("creatinine"):
        renal_evidence.append(make_evidence("Creatinine", raw_map.get("creatinine")))

    if Creat is not None and Creat > 110:
        routes.append({
            "name": "Renal impairment route",
            "category": "renal",
            "priority": "high" if Creat > 300 else "moderate",
            "pattern": "Raised creatinine and/or urea may indicate impaired kidney function or reduced perfusion.",
            "evidence": [e for e in renal_evidence if e],
            "suggested_follow_up": [
                "Review baseline renal function and eGFR if available.",
                "Assess volume status, medications (e.g. NSAIDs, ACE inhibitors) and comorbidities.",
                "Consider urine dipstick/urinalysis and repeat labs.",
            ],
            "notes": [
                "Interpret using local reference ranges and trend data.",
            ],
        })

    if Urea is not None and Creat is not None and Urea > 7.5 and (Urea / max(Creat, 1)) > 0.15:
        routes.append({
            "name": "Pre-renal / dehydration pattern route",
            "category": "renal",
            "priority": "moderate",
            "pattern": "Disproportionately raised urea relative to creatinine may be seen in dehydration or reduced renal perfusion.",
            "evidence": [e for e in renal_evidence if e],
            "suggested_follow_up": [
                "Correlate with blood pressure, pulse, mucous membranes and fluid intake.",
                "Review diuretic use, vomiting/diarrhoea and other fluid losses.",
            ],
            "notes": [
                "Definitive assessment of volume status is clinical.",
            ],
        })

    # Sodium / potassium routes
    if Na is not None and Na < 133:
        ev_na = make_evidence("Sodium", raw_map.get("na"))
        routes.append({
            "name": "Hyponatraemia route",
            "category": "electrolytes",
            "priority": "high" if Na < 120 else "moderate",
            "pattern": "Low sodium may be associated with fluid imbalance, medications or endocrine disorders.",
            "evidence": [ev_na] if ev_na else [],
            "suggested_follow_up": [
                "Correlate with symptoms (confusion, seizures, headache) and duration.",
                "Review medications (e.g. diuretics, SSRIs), fluid intake and comorbidities.",
                "Follow local hyponatraemia management guidelines.",
            ],
            "notes": [
                "Rapid shifts in sodium can be dangerous; management must follow protocol.",
            ],
        })

    if Na is not None and Na > 145:
        ev_na = make_evidence("Sodium", raw_map.get("na"))
        routes.append({
            "name": "Hypernatraemia route",
            "category": "electrolytes",
            "priority": "high" if Na > 155 else "moderate",
            "pattern": "High sodium is often related to dehydration or water loss.",
            "evidence": [ev_na] if ev_na else [],
            "suggested_follow_up": [
                "Assess fluid losses, thirst, access to water and level of consciousness.",
                "Review serum glucose and other osmotically active substances if indicated.",
            ],
            "notes": [
                "Correction should be controlled according to local protocols.",
            ],
        })

    if K is not None and K < 3.3:
        ev_k = make_evidence("Potassium", raw_map.get("k"))
        routes.append({
            "name": "Hypokalaemia route",
            "category": "electrolytes",
            "priority": "high" if K < 2.8 else "moderate",
            "pattern": "Low potassium may increase arrhythmia risk.",
            "evidence": [ev_k] if ev_k else [],
            "suggested_follow_up": [
                "Review diuretics, vomiting/diarrhoea and dietary intake.",
                "Consider ECG assessment if significantly abnormal.",
            ],
            "notes": [
                "Replacement strategies must follow local guidelines.",
            ],
        })

    if K is not None and K > 5.1:
        ev_k = make_evidence("Potassium", raw_map.get("k"))
        routes.append({
            "name": "Hyperkalaemia route",
            "category": "electrolytes",
            "priority": "high" if K > 6.0 else "moderate",
            "pattern": "High potassium may be life-threatening due to arrhythmia risk.",
            "evidence": [ev_k] if ev_k else [],
            "suggested_follow_up": [
                "Urgently correlate with ECG changes and symptoms.",
                "Exclude pseudohyperkalaemia (e.g. haemolysed sample).",
                "Follow local emergency hyperkalaemia treatment protocol if confirmed.",
            ],
            "notes": [
                "This route only flags the pattern; actual management is protocol-driven.",
            ],
        })

    # --------------------------------------------------
    # 5) LIVER PATTERN ROUTE
    # --------------------------------------------------
    liver_evidence = []
    for key, label in [("ast", "AST"), ("alt", "ALT"), ("alp", "ALP"), ("ggt", "GGT"), ("bilirubin", "Bilirubin")]:
        if raw_map.get(key):
            liver_evidence.append(make_evidence(label, raw_map.get(key)))

    if any([AST and AST > 40, ALT and ALT > 40, ALP and ALP > 130, GGT and GGT > 60, Bili and Bili > 21]):
        routes.append({
            "name": "Liver function abnormal pattern route",
            "category": "liver",
            "priority": "moderate",
            "pattern": "Abnormal liver enzymes and/or bilirubin suggest hepatic or biliary involvement.",
            "evidence": [e for e in liver_evidence if e],
            "suggested_follow_up": [
                "Correlate with history (alcohol, medications, viral hepatitis risk, metabolic disease).",
                "Consider repeat LFTs, ultrasound or specialist referral depending on pattern and severity.",
            ],
            "notes": [
                "Different patterns (hepatocellular vs cholestatic) require different workup; this route is a high-level prompt.",
            ],
        })

    # --------------------------------------------------
    # 6) SAFETY – if no routes at all, still return a neutral one
    # --------------------------------------------------
    if not routes:
        routes.append({
            "name": "No strong pattern detected",
            "category": "info",
            "priority": "low",
            "pattern": "No major deviations detected by simple rules engine.",
            "evidence": [],
            "suggested_follow_up": [
                "Interpret results against local reference ranges and full clinical history.",
            ],
            "notes": [
                "Normal or near-normal results can still be clinically significant depending on context.",
            ],
        })

    return routes


# ======================================================
#   REPORT PROCESSOR
# ======================================================

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

        # Download original PDF
        pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
        if hasattr(pdf_bytes, "data"):
            pdf_bytes = pdf_bytes.data

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)

        merged_text = None
        ocr_cbc_list = []

        if scanned:
            print(f"📄 Report {report_id}: SCANNED PDF detected → OCR pipeline")
            images = convert_from_bytes(pdf_bytes)
            combined_cbc = []

            for img in images:
                img_bytes_io = io.BytesIO()
                img.save(img_bytes_io, format="PNG")
                img_bytes = img_bytes_io.getvalue()

                try:
                    ocr_json = extract_cbc_from_image(img_bytes)
                    if isinstance(ocr_json, dict) and "cbc" in ocr_json:
                        combined_cbc.extend(ocr_json["cbc"])
                except Exception as e:
                    print("Vision OCR error:", e)

            ocr_cbc_list = combined_cbc
            if combined_cbc:
                merged_text = json.dumps({"cbc": combined_cbc}, ensure_ascii=False)
            else:
                print("⚠️ OCR produced no usable CBC values; falling back to raw text if any.")
                merged_text = text or l_text

        else:
            print(f"📝 Report {report_id}: Digital PDF detected → text pipeline")
            merged_text = text or l_text

        if not (merged_text and merged_text.strip()):
            raise ValueError("No usable content extracted for AI processing")

        # Main interpretation step
        ai_json = call_ai_on_report(merged_text)

        # If OCR list exists but model didn't populate cbc, inject it
        if ocr_cbc_list and not ai_json.get("cbc"):
            ai_json["cbc"] = ocr_cbc_list

        # ROUTE ENGINE – the key AMI differentiator
        routes = generate_clinical_routes(ai_json, job)
        ai_json["routes"] = routes

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
    print("AMI Worker with Vision OCR + Route Engine started… watching for jobs.")

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
