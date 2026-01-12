#!/usr/bin/env python3
"""
AMI Worker v4+ (Pattern → Route → Next Steps)
Features:
- Scanned PDF OCR (OpenAI) -> structured CBC + chemistry extraction
- Text PDF parsing
- AI interpretation (GPT-4o-mini style)
- Route engine (Patterns -> Route -> Next Steps)
- Urgency flags & Severity grading
- Differential diagnosis trees (safe, non-prescriptive)
- Trend comparison with prior completed reports (by patient name)
- Robust error handling and defensive parsing
"""

print(">>> AMI Worker v4 starting — Pattern → Route → Next Steps (with urgency, severity, differential, trends)")

import os
import time
import json
import io
import traceback
import base64
import re
from datetime import datetime

# Third-party
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes

# ---------------------------
# Environment / clients
# ---------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set — OpenAI requests will likely fail (but code will still run).")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

# ---------------------------
# Utilities
# ---------------------------
def safe_get_parsed_from_choice(choice):
    """Helper: get parsed JSON from various SDK response shapes."""
    try:
        # Preferred: parsed property (SDK may populate .message.parsed)
        msg = getattr(choice, "message", None)
        if msg:
            parsed = getattr(msg, "parsed", None)
            if parsed is not None:
                return parsed
            # fallback to content
            content = getattr(msg, "content", None)
            if isinstance(content, dict):
                return content
            if isinstance(content, list):
                # search for dict/json in list
                for item in content:
                    if isinstance(item, dict) and "json" in item:
                        return item.get("json")
                # fallback join strings
                txt = "".join(part.get("text", "") for part in content if isinstance(part, dict))
                try:
                    return json.loads(txt)
                except:
                    return txt
            if isinstance(content, str):
                try:
                    return json.loads(content)
                except:
                    return content

        # older fallback: choice.get("text")
        if hasattr(choice, "text"):
            txt = getattr(choice, "text")
            try:
                return json.loads(txt)
            except:
                return txt

    except Exception:
        pass
    return None

def clean_number(val):
    """Convert things like '88.0%', '11,6 g/dL', '4.2*' -> float or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    s = s.replace(",", ".")
    # remove percent sign, trailing stars and words, then regex first number
    nums = re.findall(r"-?\d+\.?\d*", s)
    if not nums:
        return None
    try:
        return float(nums[0])
    except:
        return None
        
def extract_patient_demographics(text: str) -> dict:
    """
    Extract patient name, age, sex from SA lab reports:
    Lancet, Ampath, PathCare, Path24.
    Conservative: only fills fields when confident.
    """
    if not text:
        return {"name": None, "age": None, "sex": "Unknown"}

    t = text.lower()

    name = None
    age = None
    sex = "Unknown"

    # ==================================================
    # NAME (Lancet / Ampath / PathCare)
    # ==================================================
    name_patterns = [
        r"patient\s*name\s*[:\-]\s*([a-z ,.'-]{3,60})",
        r"patient\s*[:\-]\s*([a-z ,.'-]{3,60})",
        r"\bname\s*[:\-]\s*([a-z ,.'-]{3,60})",
    ]

    for p in name_patterns:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            cand = m.group(1).strip().title()
            if 1 <= len(cand.split()) <= 4:
                name = cand
                break

    # ==================================================
    # AGE / SEX combined (very common in SA labs)
    # ==================================================
    m = re.search(
        r"(\d{1,3})\s*(y|yrs|years)?\s*/\s*(male|female|m|f)",
        t,
        re.IGNORECASE
    )
    if m:
        try:
            a = int(m.group(1))
            if 0 < a < 120:
                age = a
            sex = "Male" if m.group(3).lower() in ("m", "male") else "Female"
        except:
            pass

    # ==================================================
    # AGE only
    # ==================================================
    if age is None:
        m = re.search(r"\bage\s*[:\-]\s*(\d{1,3})\b", t)
        if m:
            try:
                a = int(m.group(1))
                if 0 < a < 120:
                    age = a
            except:
                pass

    # ==================================================
    # SEX only
    # ==================================================
    if sex == "Unknown":
        m = re.search(r"\b(sex|gender)\s*[:\-]\s*(male|female|m|f)\b", t)
        if m:
            sex = "Male" if m.group(2).lower() in ("m", "male") else "Female"

    # ==================================================
    # DOB → AGE (fallback)
    # ==================================================
    if age is None:
        m = re.search(
            r"(dob|date of birth)\s*[:\-]\s*(\d{1,4}[\/\-]\d{1,2}[\/\-]\d{1,4})",
            t
        )
        if m:
            for fmt in ("%Y/%m/%d", "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
                try:
                    dob = datetime.strptime(m.group(2), fmt)
                    today = datetime.today()
                    a = today.year - dob.year - (
                        (today.month, today.day) < (dob.month, dob.day)
                    )
                    if 0 < a < 120:
                        age = a
                        break
                except:
                    continue

    return {
        "name": name,
        "age": age,
        "sex": sex
    }




def iso_now():
    return datetime.utcnow().isoformat() + "Z"
    
def overall_clinical_status(cdict: dict) -> str:
    """
    High-level clinical wording.
    Never uses the word 'NORMAL' if any abnormality exists.
    """
    for row in cdict.values():
        if not isinstance(row, dict):
            continue
        flag = (row.get("flag") or "").lower()
        if flag in ("high", "low"):
            return "No acute pathology detected. Mild abnormalities noted."

    return "No acute abnormalities detected."


# ---------------------------
# PDF helpers
# ---------------------------
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from selectable PDFs using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n\n".join(pages).strip()
    except Exception as e:
        print("PDF parse error:", e)
        return ""

def is_scanned_pdf(text: str) -> bool:
    """Very simple heuristic: nearly no text means scanned image PDF."""
    if not text:
        return True
    return len(text.strip()) < 30

# ---------------------------
# OCR: Image -> structured CBC (OpenAI Vision via chat completions)
# ---------------------------
def extract_cbc_from_image(image_bytes: bytes) -> dict:
    """
    Send an image to OpenAI (vision-capable chat completions) to extract CBC/chemistry.
    Returns: dict with {"cbc": [{analyte, value, units, reference_low, reference_high}, ...]}
    Defensive: returns {"cbc": []} on failure.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        system_prompt = (
            "You are an OCR assistant extracting laboratory values from a scanned blood report. "
            "Return ONLY strict JSON with this structure:\n"
            "{ 'cbc': [ { 'analyte':'', 'value':'', 'units':'', 'reference_low':'', 'reference_high':'' } ] }\n"
            "Include CBC analytes, differential, platelets, electrolytes, urea, creatinine, LFTs, CK, CRP if present. "
            "Do not add explanatory text."
        )

        # Use chat completions create with image_url object payload (SDK format)
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
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        # defensive parsing
        choice = resp.choices[0]
        parsed = safe_get_parsed_from_choice(choice)
        if isinstance(parsed, dict) and "cbc" in parsed:
            return parsed
        # if we got a string containing JSON, try loads
        if isinstance(parsed, str):
            try:
                loaded = json.loads(parsed)
                if isinstance(loaded, dict) and "cbc" in loaded:
                    return loaded
            except:
                pass

        print("OCR: unexpected structure, returning empty CBC. Raw parsed:", parsed)
        return {"cbc": []}

    except Exception as e:
        print("OCR error:", e)
        traceback.print_exc()
        return {"cbc": []}

# ---------------------------
# AI interpretation (text) -> structured ai_json
# ---------------------------
def call_ai_on_report(text: str) -> dict:
    """
    Ask model to interpret a JSON-like CBC content or plain text and return a structured JSON with:
    patient, cbc[], summary (impression, suggested_follow_up).
    Defensive parsing similar to OCR.
    """
    try:
        system_prompt = (
            "You are AMI, a medical lab interpreter for clinicians. "
            "You MUST NOT give a formal diagnosis or prescribe. Return STRICT JSON with at least:\n"
            "{\n"
            '  "patient": {"name": null, "age": null, "sex": "Unknown"},\n'
            '  "cbc": [ { "analyte":"", "value":"", "units":"", "reference_low":"", "reference_high":"", "flag":"normal|low|high|unknown" } ],\n'
            '  "summary": { "impression":"", "suggested_follow_up":"" }\n'
            "}\n"
            "If input is already JSON with cbc rows, parse those and add concise clinical impression and follow-up."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.05,
        )

        choice = resp.choices[0]
        parsed = safe_get_parsed_from_choice(choice)
        if isinstance(parsed, dict):
            return parsed
        # fallback: if it's a string that looks like JSON
        if isinstance(parsed, str):
            try:
                return json.loads(parsed)
            except:
                pass

        # Last fallback: return minimal wrapper
        return {
            "patient": {"name": None, "age": None, "sex": "Unknown"},
            "cbc": [],
            "summary": {"impression": "No structured interpretation produced.", "suggested_follow_up": ""}
        }

    except Exception as e:
        print("AI interpretation error:", e)
        traceback.print_exc()
        return {
            "patient": {"name": None, "age": None, "sex": "Unknown"},
            "cbc": [],
            "summary": {"impression": f"AI error: {e}", "suggested_follow_up": ""}
        }

# ---------------------------
# Build canonical CBC dict for the route engine
# ---------------------------
def build_cbc_value_dict(ai_json: dict) -> dict:
    """
    Convert ai_json['cbc'] (and chemistry-style rows) into a canonical dictionary
    keyed by normalized analyte names (CBC + Chemistry + Lipids).
    """
    out = {}
    rows = ai_json.get("cbc") or []

    for r in rows:
        if not isinstance(r, dict):
            continue

        raw_name = (r.get("analyte") or r.get("test") or "").strip()
        if not raw_name:
            continue

        name = raw_name.lower()

        def put(key):
            # First-write wins to avoid duplicates
            if key not in out:
                out[key] = r

        # -----------------
        # CBC
        # -----------------
        if "haemo" in name or name == "hb" or "hemoglobin" in name:
            put("Hb")

        elif name.startswith("mcv"):
            put("MCV")

        elif name.startswith("mch"):
            put("MCH")

        elif "red cell" in name or name == "rbc":
            put("RBC")

        elif "white" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            put("WBC")

        elif "neut" in name:
            put("Neutrophils")

        elif "lymph" in name:
            put("Lymphocytes")

        elif "platelet" in name or name == "plt":
            put("Platelets")

        # -----------------
        # Renal / Electrolytes
        # -----------------
        elif "creatinine" in name:
            put("Creatinine")

        elif name.startswith("urea"):
            put("Urea")

        elif "sodium" in name or name == "na":
            put("Sodium")

        elif "potassium" in name or name == "k":
            put("Potassium")

        elif "calcium" in name:
            put("Calcium")
            
        elif "anion gap" in name:
            put("Anion Gap")
        
        elif "bicarbonate" in name or name in ("co2", "hco3"):
            put("Bicarbonate")


        # -----------------
        # Liver / Enzymes
        # -----------------
        elif name == "alt" or ("alt" in name and "alanine" in name):
            put("ALT")

        elif "ast" in name:
            put("AST")

        elif "alp" in name or "alkaline phosphatase" in name:
            put("ALP")

        elif "ggt" in name or "gamma gt" in name or "gamma-glutamyl" in name:
            put("GGT")

        elif "bilirubin" in name:
            put("Bilirubin")

        elif "creatine kinase" in name or name == "ck":
            put("CK")

        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")

        # -----------------
        # Inflammation
        # -----------------
        elif name == "crp" or "c-reactive" in name:
            put("CRP")

        # -----------------
        # LIPIDS (THIS WAS MISSING 🔥)
        # -----------------
        elif "triglyceride" in name:
            put("Triglycerides")

        elif "cholesterol total" in name or name == "cholesterol":
            put("Cholesterol Total")

        elif "hdl" in name:
            put("HDL")

        elif "ldl" in name:
            put("LDL")

        elif "non-hdl" in name or "non hdl" in name:
            put("Non-HDL")

    return out

# ---------------------------
# Severity grading & urgency flags
# ---------------------------
def evaluate_severity_and_urgency(cdict: dict) -> dict:
    """
    Returns:
    {
      "severity": "low|moderate|high|critical",
      "urgency_flags": [ { "reason": "...", "level": "urgent|immediate|monitor" } ],
      "key_issues": [ "...brief strings..." ]
    }
    """
    sev_score = 0
    flags = []
    key_issues = []

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    Hb = v("Hb")
    Plt = v("Platelets")
    WBC = v("WBC")
    Neut = v("Neutrophils")
    K = v("Potassium")
    Cr = v("Creatinine")
    CK = v("CK")
    CRP = v("CRP")

    # ---- Hb ----
    if Hb is not None:
        if Hb < 6.5:
            sev_score += 5
            flags.append({"reason": f"Hb {Hb} g/dL — severe anaemia", "level": "immediate"})
            key_issues.append(f"Hb {Hb}")
        elif Hb < 8:
            sev_score += 3
            flags.append({"reason": f"Hb {Hb} g/dL — significant anaemia", "level": "urgent"})
            key_issues.append(f"Hb {Hb}")
        elif Hb < 11:
            sev_score += 1
            key_issues.append(f"Hb {Hb} (mild)")

    # ---- Platelets ----
    if Plt is not None:
        if Plt < 20:
            sev_score += 5
            flags.append({"reason": f"Platelets {Plt} — critical bleeding risk", "level": "immediate"})
            key_issues.append(f"Platelets {Plt}")
        elif Plt < 50:
            sev_score += 3
            flags.append({"reason": f"Platelets {Plt} — bleeding risk", "level": "urgent"})
            key_issues.append(f"Platelets {Plt}")

    # ---- Potassium ----
    if K is not None:
        if K < 3.0 or K > 6.0:
            sev_score += 5
            flags.append({"reason": f"Potassium {K} mmol/L — arrhythmia risk", "level": "immediate"})
            key_issues.append(f"K {K}")
        elif K < 3.3 or K > 5.5:
            sev_score += 3
            flags.append({"reason": f"Potassium {K} mmol/L — significant abnormality", "level": "urgent"})
            key_issues.append(f"K {K}")

    # ---- Creatinine ----
    if Cr is not None:
        if Cr > 250:
            sev_score += 4
            flags.append({"reason": f"Creatinine {Cr} µmol/L — possible AKI", "level": "urgent"})
            key_issues.append(f"Cr {Cr}")
        elif Cr > 120:
            sev_score += 2
            key_issues.append(f"Cr {Cr}")

    # ---- CK ----
    if CK is not None and CK > 5000:
        sev_score += 4
        flags.append({"reason": f"CK {CK} U/L — rhabdomyolysis physiology", "level": "urgent"})
        key_issues.append(f"CK {CK}")

    # ---- WBC / Infection ----
    if WBC is not None:
        if WBC > 25 or WBC < 1:
            sev_score += 4
            flags.append({"reason": f"WBC {WBC} — severe leukocyte abnormality", "level": "urgent"})
            key_issues.append(f"WBC {WBC}")
        elif WBC >= 18:
            sev_score += 2
            flags.append({"reason": f"WBC {WBC} — marked leukocytosis", "level": "urgent"})
            key_issues.append(f"WBC {WBC}")

    if Neut is not None and Neut >= 15:
        sev_score += 2
        flags.append({"reason": f"Neutrophils {Neut} — neutrophil-predominant response", "level": "urgent"})
        key_issues.append(f"Neut {Neut}")

    if CRP is not None and CRP >= 30:
        sev_score += 3
        flags.append({"reason": f"CRP {CRP} mg/L — significant inflammation", "level": "urgent"})
        key_issues.append(f"CRP {CRP}")

    # ---- Severity mapping ----
    if sev_score >= 8:
        severity = "critical"
    elif sev_score >= 4:
        severity = "high"
    elif sev_score >= 2:
        severity = "moderate"
    else:
        severity = "low"

    return {
        "severity": severity,
        "urgency_flags": flags,
        "key_issues": key_issues
    }

# ---------------------------
# Differential diagnosis trees (safe)
# ---------------------------
def generate_differential_trees(cdict: dict) -> dict:
    """
    Return a safe, non-prescriptive differential suggestion tree keyed by patterns.
    Example:
    {
      "Microcytic anaemia": ["Iron deficiency", "Chronic inflammation", "Thalassemia trait (less likely)"],
      "Macrocytic anaemia": [...]
    }
    """
    out = {}
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    Hb = v("Hb"); MCV = v("MCV"); WBC = v("WBC"); Plt = v("Platelets"); Cr = v("Creatinine"); CRP = v("CRP")

    # Anaemia differentials
    if Hb is not None and Hb < 13:
        if MCV is not None and MCV < 80:
            out["Microcytic anaemia"] = [
                "Iron deficiency (common) — check ferritin & iron studies",
                "Anaemia of chronic disease/inflammation",
                "Thalassaemia trait (consider if ferritin normal and family history)"
            ]
        elif MCV is not None and MCV > 100:
            out["Macrocytic anaemia"] = [
                "Vitamin B12 or folate deficiency",
                "Drug or alcohol effect",
                "Bone marrow disorder (less common)"
            ]
        else:
            out["Normocytic anaemia"] = [
                "Early iron deficiency",
                "Renal disease / reduced EPO",
                "Acute blood loss or chronic disease"
            ]

    # Thrombocytopenia
    if Plt is not None and Plt < 150:
        out.setdefault("Thrombocytopenia", []).extend([
            "Immune-mediated thrombocytopenia (ITP)",
            "Drug-induced",
            "Bone marrow suppression / infiltration",
            "Sepsis or consumptive processes"
        ])

    # Leukocytosis
    if WBC is not None and WBC > 12:
        if CRP and CRP > 10:
            out.setdefault("Raised WBC (with inflammation)", []).extend([
                "Bacterial infection (common)",
                "Severe inflammatory response (e.g., pancreatitis, large tissue injury)"
            ])
        else:
            out.setdefault("Raised WBC", []).extend([
                "Inflammation or stress leucocytosis",
                "Haematologic process (if very high or blasts present) — specialist review"
            ])

    # Renal / metabolic
    if Cr is not None and Cr > 120:
        out.setdefault("Renal impairment", []).extend([
            "AKI (pre-renal dehydration, sepsis, nephrotoxic drugs)",
            "Chronic kidney disease (look at past creatinine/eGFR)"
        ])

    if not out:
        out["No specific differential patterns triggered"] = ["Correlate clinically and review prior results"]

    return out
    
def build_chemistry_context_and_steps(cdict: dict) -> dict:
    """
    Conservative, doctor-facing chemistry interpretation support.
    No diagnoses. No admission language.
    """

    steps = []
    context = []

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    CRP = v("CRP")
    Bil = v("Bilirubin")
    ALT = v("ALT")
    AST = v("AST")
    ALP = v("ALP")
    GGT = v("GGT")

    Chol = v("Cholesterol")
    LDL = v("LDL")
    TG = v("Triglycerides")

    # ---- Suggested next steps (optional, conservative) ----
    if Chol is not None or LDL is not None or TG is not None:
        steps.append("Repeat fasting lipid profile in 3–6 months if clinically appropriate")
        steps.append("Consider fasting status and recent alcohol intake when interpreting triglycerides")

    if Bil is not None:
        steps.append("If bilirubin remains elevated, consider repeat fractionation if clinically indicated")

    # ---- Clinical context considerations (reassurance framing) ----
    if Bil is not None and (ALT is not None and AST is not None and ALP is not None and GGT is not None):
        if ALT <= 50 and AST <= 50 and ALP <= 130 and GGT <= 60:
            context.append(
                "Unconjugated hyperbilirubinaemia with normal ALT, AST, ALP, and GGT is commonly benign "
                "(e.g. Gilbert syndrome), particularly if intermittent"
            )

    if CRP is not None and CRP < 5:
        context.append(
            "Absence of inflammatory marker elevation reduces likelihood of acute inflammatory or infectious pathology"
        )

    age = cdict.get("_patient_age")

    if (Chol is not None or LDL is not None) and age is not None and age < 40:
        context.append(
            "At this age, absolute short-term cardiovascular risk is low; "
            "lifestyle optimisation is appropriate as first-line management."
        )
    elif Chol is not None or LDL is not None:
        context.append(
            "Lipid abnormalities suggest increased long-term cardiovascular risk rather than acute illness."
        )





    return {
        "next_steps": steps,
        "clinical_context": context
    }

# ---------------------------
# Trend comparison (search prior completed reports by patient name)
# ---------------------------
def trend_comparison(patient_name: str, current_cdict: dict) -> dict:
    """
    Fetch the last N completed reports and compare main analytes to produce a short trend summary.
    If patient_name is None or no previous reports, returns informative message.
    """
    if not patient_name:
        return {"note": "No patient name available; trend comparison skipped.", "trends": []}

    try:
        # Fetch recent completed reports (limit 20) and filter in Python by patient name match
        res = supabase.table("reports").select("*").eq("ai_status", "completed").order("created_at", desc=True).limit(20).execute()
        prior = res.data or []
        matches = []
        
        for r in prior:
            try:
                ai = r.get("ai_results") or r.get("ai_results_raw") or {}
                # ai could be a string or dict
                if isinstance(ai, str):
                    try:
                        ai = json.loads(ai)
                    except:
                        ai = {}
        
                pname = None
                if isinstance(ai, dict):
                    pname = (ai.get("patient") or {}).get("name")
        
                if pname and patient_name and pname.strip().lower() == patient_name.strip().lower():
                    matches.append({"created_at": r.get("created_at"), "ai": ai})
            except Exception:
                continue


        if not matches:
            return {"note": "No prior completed reports found for this patient.", "trends": []}

        # For each analyte of interest, create tiny trend strings comparing most recent prior value to current
        analytes = ["Hb", "WBC", "Platelets", "Neutrophils", "MCV", "Creatinine", "CRP", "Potassium", "Sodium"]
        trends = []
        # take the most recent prior (first in matches)
        for m in matches[:3]:
            pass  # we will compute pairwise below

        # Build a mapping of analyte -> list of (date, value)
        series = {}
        for m in matches:
            ai = m["ai"] or {}
            cbc_list = ai.get("cbc") or []
            row_map = {}
            for row in cbc_list:
                if not isinstance(row, dict): continue
                analyte = (row.get("analyte") or "").strip()
                if analyte:
                    row_map[analyte.lower()] = row
            created = m.get("created_at") or m.get("createdAt") or ""
            for a in analytes:
                # try find by canonical keys too
                val = None
                # check canonical keys in ai (some reports may store canonical)
                if isinstance(ai, dict) and ai.get("cbc"):
                    for row in ai.get("cbc"):
                        name = (row.get("analyte") or "").lower()
                        if a.lower() in name or name == a.lower():
                            val = clean_number(row.get("value"))
                            break
                # append if found
                if val is not None:
                    series.setdefault(a, []).append({"date": created, "value": val})

        # Compare most recent prior value to current for each analyte with data
        for a, points in series.items():
            # sort by date descending if possible
            points_sorted = [p for p in points]
            # if current_cdict contains a key, compare
            cur_val = clean_number(current_cdict.get(a, {}).get("value"))
            if cur_val is None:
                continue
            # find most recent prior (first point)
            prior_val = points_sorted[0]["value"] if points_sorted else None
            if prior_val is None:
                continue
            if cur_val > prior_val:
                trends.append(f"{a}: increased from {prior_val} → {cur_val}")
            elif cur_val < prior_val:
                trends.append(f"{a}: decreased from {prior_val} → {cur_val}")
            else:
                trends.append(f"{a}: stable at {cur_val}")

        note = f"Compared to {len(matches)} prior completed report(s)."
        return {"note": note, "trends": trends}

    except Exception as e:
        print("Trend comparison error:", e)
        traceback.print_exc()
        return {"note": "Trend comparison failed: " + str(e), "trends": []}
       
def detect_simple_clinical_patterns(cdict: dict) -> list:
    notes = []

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    bilirubin = v("Bilirubin")
    alt = v("ALT")
    ast = v("AST")
    alp = v("ALP")
    ggt = v("GGT")

    tg = v("Triglycerides")
    ldl = v("LDL")
    hdl = v("HDL")

    # ---- Isolated bilirubin pattern ----
    if bilirubin is not None and bilirubin > 21:
        def ref_high(k):
            return clean_number(cdict.get(k, {}).get("reference_high"))

        if all(
            x is not None and
            (ref_high(k) is None or x <= ref_high(k))
            for k, x in [("ALT", alt), ("AST", ast), ("ALP", alp), ("GGT", ggt)]
        ):
            notes.append(
                "Pattern: Isolated unconjugated hyperbilirubinaemia with normal liver enzymes — "
                "commonly benign (e.g. Gilbert syndrome), particularly if intermittent."
            )

    # ---- Lipid pattern ----
    if tg is not None or ldl is not None:
        notes.append(
            "Pattern: Mild mixed dyslipidaemia — lipid abnormalities at this age "
            "are more suggestive of long-term cardiovascular risk rather than acute illness."
        )

    return notes


# ---------------------------
# Route helper: priority-aware insertion
# ---------------------------
def add_route(routes, priority, pattern, route, next_steps):
    """
    priority: 'primary' | 'secondary' | 'contextual'
    Primary routes are inserted at the top.
    """
    entry = {
        "priority": priority,
        "pattern": pattern,
        "route": route,
        "next_steps": next_steps
    }

    if priority == "primary":
        routes.insert(0, entry)
    else:
        routes.append(entry)

def severity_from_routes(routes: list) -> str:
    """
    Determines minimum severity based on dominant clinical routes.
    Routes OVERRIDE numeric scoring (doctor logic).
    """
    if not routes:
        return "low"

    # Any PRIMARY route = at least HIGH severity
    for r in routes:
        if r.get("priority") == "primary":
            return "high"

    # Any SECONDARY route = at least MODERATE severity
    for r in routes:
        if r.get("priority") == "secondary":
            return "moderate"

    return "low"

def build_follow_up_block(cdict: dict, routes: list, severity: str) -> list:
    """
    Returns a short, clean follow-up list using EXISTING findings only.
    No diagnoses. No treatment. No new logic.
    """
    follow_up = []

    # Severity-driven framing
    if severity in ("high", "critical"):
        follow_up.append(
            "Urgent clinical reassessment is advised based on the severity of abnormalities detected."
        )
    elif severity == "moderate":
        follow_up.append(
            "Timely clinical review is recommended to reassess abnormal findings and trends."
        )
    else:
        follow_up.append(
            "Findings may be monitored in appropriate clinical context."
        )

    # Route-driven reinforcement (no new ideas)
    for r in routes:
        if r.get("priority") == "primary":
            follow_up.append(
                "Primary abnormal laboratory patterns should be prioritised during clinical assessment."
            )
            break

    # Trend awareness
    follow_up.append(
        "Correlation with symptoms, vital signs, and previous laboratory results is important."
    )

    return follow_up



# ---------------------------
# Route engine aggregator: Patterns -> Route -> Next Steps
# also includes severity/urgency/differential/trends
# ---------------------------
def build_full_clinical_report(ai_json: dict) -> dict:
    """
    Given ai_json (from call_ai_on_report), add:
    - canonical cdict
    - routes
    - severity & urgency
    - differential trees
    - trend comparison
    """

    # ---------------------------
    # Canonical dict
    # ---------------------------
    cdict = build_cbc_value_dict(ai_json)

    # -----------------------------------
    # Merge chemistry rows into canonical dict (SAFE)
    # -----------------------------------
    for r in (ai_json.get("chemistry") or []):
        if not isinstance(r, dict):
            continue

        raw = (r.get("analyte") or r.get("test") or "").lower().strip()
        if not raw:
            continue

        def put(k):
            if k not in cdict:
                cdict[k] = r

        if "bilirubin" in raw:
            put("Bilirubin")
        elif raw == "alt" or "alanine" in raw:
            put("ALT")
        elif raw == "ast" or "aspartate" in raw:
            put("AST")
        elif "alkaline phosphatase" in raw or raw == "alp":
            put("ALP")
        elif "gamma" in raw or "ggt" in raw:
            put("GGT")
        elif "triglyceride" in raw:
            put("Triglycerides")
        elif "ldl" in raw:
            put("LDL")
        elif "hdl" in raw:
            put("HDL")
        elif "cholesterol" in raw and "non" in raw:
            put("Non-HDL")
        elif "cholesterol" in raw:
            put("Cholesterol")
        elif "crp" in raw:
            put("CRP")
        elif "creatinine" in raw:
            put("Creatinine")

    # ---------------------------
    # Patient context
    # ---------------------------
    if isinstance(ai_json.get("patient"), dict):
        cdict["_patient_age"] = ai_json["patient"].get("age")

    overall_status = overall_clinical_status(cdict)

    # Pattern-first notes
    patterns = detect_simple_clinical_patterns(cdict)

    # ---------------------------
    # Routes
    # ---------------------------
    routes = []
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    # -------- Extract values FIRST (REQUIRED) --------
    Hb = v("Hb")
    MCV = v("MCV")
    WBC = v("WBC")
    Neut = v("Neutrophils")
    Lymph = v("Lymphocytes")
    Plt = v("Platelets")
    Cr = v("Creatinine")
    CRP = v("CRP")
    CK = v("CK")
    Na = v("Sodium")
    K = v("Potassium")
    # =====================================================
    # PASS 1 — PRIMARY LIFE-THREATENING CBC ROUTES
    # =====================================================
    
    # ---- Severe anaemia (immediate risk) ----
    if Hb is not None and Hb < 7:
        add_route(
            routes,
            priority="primary",
            pattern="Severe anaemia",
            route=f"Haemoglobin {Hb} g/dL at a level associated with reduced oxygen delivery",
            next_steps=[
                "Urgent clinical assessment",
                "Assess haemodynamic stability and bleeding",
                "Repeat haemoglobin to confirm"
            ]
        )
    
    # ---- Critical thrombocytopenia ----
    if Plt is not None and Plt < 50:
        add_route(
            routes,
            priority="primary",
            pattern="Critical thrombocytopenia",
            route=f"Platelet count {Plt} ×10⁹/L with increased bleeding risk",
            next_steps=[
                "Assess for bleeding or bruising",
                "Review medications and recent infections",
                "Repeat platelet count and peripheral smear"
            ]
        )
    
    # ---- Pancytopenia (bone marrow danger pattern) ----
    cytopenias = sum([
        1 if Hb is not None and Hb < 10 else 0,
        1 if WBC is not None and WBC < 3 else 0,
        1 if Plt is not None and Plt < 100 else 0
    ])
    
    if cytopenias >= 2:
        add_route(
            routes,
            priority="primary",
            pattern="Pancytopenia physiology",
            route="Multiple concurrent cytopenias raise concern for bone marrow pathology or severe systemic illness",
            next_steps=[
                "Urgent peripheral blood smear",
                "Review recent drugs, infections, and systemic symptoms",
                "Specialist review if persistent or worsening"
            ]
        )
    
    # ---- Neutropenic infection physiology ----
    if WBC is not None and WBC < 1.5 and CRP is not None and CRP > 20:
        add_route(
            routes,
            priority="primary",
            pattern="Neutropenic inflammatory response",
            route="Low white cell count with inflammatory marker elevation raises concern for high-risk infection",
            next_steps=[
                "Urgent clinical assessment",
                "Careful infection source evaluation",
                "Repeat CBC to assess trend"
            ]
        )
    
    # ---- Extreme leukocytosis ----
    if WBC is not None and WBC > 30:
        add_route(
            routes,
            priority="primary",
            pattern="Extreme leukocytosis",
            route=f"WBC {WBC} ×10⁹/L may reflect severe infection or haematologic pathology",
            next_steps=[
                "Assess for sepsis or systemic illness",
                "Peripheral smear review",
                "Specialist input if unexplained"
            ]
        )
    
    
    # =====================================================
    # COMPOSITE CBC PATTERNS (doctor-style reasoning)
    # =====================================================

    # ---- Composite: Acute inflammatory / infective response (PRIMARY) ----
    if (
        WBC is not None and WBC > 12
        and Neut is not None and Neut > 75
        and CRP is not None and CRP > 20
    ):
        add_route(
            routes,
            priority="primary",
            pattern="Acute inflammatory / infective response",
            route=(
                "Neutrophil-predominant leukocytosis with elevated CRP, "
                "suggesting active infection or significant inflammatory stress"
            ),
            next_steps=[
                "Urgent clinical assessment to identify possible source of infection",
                "Correlate with symptoms, vitals, and imaging where appropriate",
                "Repeat CBC and CRP to assess trend if clinically indicated"
            ]
        )

    # ---- Microcytic anaemia (age-aware) ----
    if Hb is not None and Hb < 12 and MCV is not None and MCV < 80:
        age = cdict.get("_patient_age")
        route_text = "Microcytic hypochromic anaemia — iron deficiency most likely"

        if age is not None and age < 25:
            route_text += " (menstrual iron loss common in this age group)"

        add_route(
            routes,
            priority="secondary",
            pattern="Microcytic anaemia",
            route=route_text,
            next_steps=[
                "Order ferritin and iron studies",
                "Review dietary intake and menstrual history where appropriate",
                "Repeat haemoglobin after correction of any acute illness"
            ]
        )

    # ---- Anaemia with concurrent inflammatory illness (SECONDARY) ----
    if Hb is not None and Hb < 12 and WBC is not None and WBC > 12:
        add_route(
            routes,
            priority="contextual",
            pattern="Anaemia with concurrent inflammatory illness",
            route=(
                "Anaemia may be exacerbated or masked by acute inflammatory state"
            ),
            next_steps=[
                "Reassess haemoglobin once acute illness has resolved",
                "Interpret iron studies cautiously while CRP remains elevated"
            ]
        )

    # =====================================================
    # ELECTROLYTE DANGER COMPOSITES (ER PRIORITY)
    # =====================================================

    if K is not None:
        if K < 3.0 or K > 6.0:
            add_route(
                routes,
                priority="primary",
                pattern="Critical potassium abnormality",
                route=f"Potassium {K} mmol/L associated with high risk of cardiac arrhythmia",
                next_steps=[
                    "Urgent clinical assessment and ECG correlation",
                    "Review renal function and medications",
                    "Repeat potassium urgently to confirm"
                ]
            )
        elif K < 3.3 or K > 5.5:
            add_route(
                routes,
                priority="secondary",
                pattern="Significant potassium abnormality",
                route=f"Potassium {K} mmol/L outside safe physiological range",
                next_steps=[
                    "Assess for symptoms and contributing factors",
                    "Review medications and renal function",
                    "Repeat electrolytes to monitor trend"
                ]
            )

    if Na is not None:
        if Na < 125 or Na > 155:
            add_route(
                routes,
                priority="primary",
                pattern="Critical sodium abnormality",
                route=f"Sodium {Na} mmol/L associated with neurological complications",
                next_steps=[
                    "Urgent clinical assessment including mental status",
                    "Review volume status and recent fluid intake",
                    "Repeat sodium and osmolality if clinically indicated"
                ]
            )
        elif Na < 130 or Na > 150:
            add_route(
                routes,
                priority="secondary",
                pattern="Significant sodium abnormality",
                route=f"Sodium {Na} mmol/L outside normal physiological range",
                next_steps=[
                    "Assess hydration status and contributing causes",
                    "Review medications (e.g. diuretics)",
                    "Monitor sodium trend with repeat testing"
                ]
            )



    # Anaemia
    if Hb is not None and Hb < 13:
        if MCV is not None:
            if MCV < 80:
                routes.append({
                    "pattern": "Microcytic anaemia",
                    "route": "Likely iron deficiency / chronic disease pattern",
                    "next_steps": [
                        "Order ferritin & iron studies",
                        "Reticulocyte count",
                        "Consider inflammation markers (CRP)"
                    ]
                })
            elif 80 <= MCV <= 100:
                routes.append({
                    "pattern": "Normocytic anaemia",
                    "route": "Possible chronic disease, renal, or early iron deficiency",
                    "next_steps": [
                        "Check creatinine & eGFR",
                        "Reticulocyte count",
                        "Clinical correlation"
                    ]
                })
            else:
                routes.append({
                    "pattern": "Macrocytic anaemia",
                    "route": "Possible B12/folate deficiency or hepatic/drug effect",
                    "next_steps": [
                        "Order B12 & folate",
                        "Review liver enzymes",
                        "Medication review"
                    ]
                })
        else:
            routes.append({
                "pattern": "Anaemia (MCV unknown)",
                "route": "Low haemoglobin — further classification needed",
                "next_steps": [
                    "Obtain MCV/MCH",
                    "Order ferritin & reticulocytes"
                ]
            })

    # WBC
    if WBC is not None:
        if WBC > 12:
            detail = "Inflammatory/infective physiology"
            nexts = []

            if Neut and Neut > 70:
                detail = "Neutrophil-predominant — bacterial pattern more likely"
                nexts.append(
                    "Correlate with fever, localising signs; treat per clinical context"
                )

            if Lymph and Lymph > 45:
                nexts.append("Consider viral causes; review symptom timeline")

            if not nexts:
                nexts.append("Correlate clinically; consider CRP and cultures if indicated")

            routes.append({
                "pattern": "Leucocytosis",
                "route": detail,
                "next_steps": nexts
            })

        elif WBC < 4:
            routes.append({
                "pattern": "Leukopenia",
                "route": "Viral suppression, marrow effect, or drugs",
                "next_steps": [
                    "Medication review",
                    "Repeat CBC",
                    "Consider specialist review if persistent"
                ]
            })

    # Platelets
    if Plt is not None:
        if Plt < 150:
            routes.append({
                "pattern": "Thrombocytopenia",
                "route": "Bleeding risk assessment",
                "next_steps": [
                    "Assess bleeding symptoms",
                    "Review drugs and prior CBCs",
                    "Haematology review if <50"
                ]
            })
        elif Plt > 450:
            routes.append({
                "pattern": "Thrombocytosis",
                "route": "Reactive vs primary thrombocytosis",
                "next_steps": [
                    "Check CRP",
                    "Repeat CBC",
                    "Consider iron studies"
                ]
            })

        # Kidney / CK
    if Cr is not None and Cr > 120:
        routes.append({
            "pattern": "Renal impairment physiology",
            "route": "Assess for AKI or CKD",
            "next_steps": [
                "Repeat U&E",
                "Review medications & hydration",
                "Consider eGFR"
            ]
        })

    if CK is not None and CK > 1000:
        routes.append({
            "pattern": "High CK",
            "route": "Muscle injury / rhabdomyolysis physiology",
            "next_steps": [
                "Check creatinine",
                "Assess muscle pain / trauma",
                "Urgent review if creatinine rising"
            ]
        })
    # =====================================================
    # PASS 1 — COMBINED HIGH-RISK PHYSIOLOGY
    # =====================================================

    # ---- Infection + thrombocytopenia (bleeding + sepsis risk) ----
    if (
        WBC is not None and WBC > 12
        and CRP is not None and CRP > 20
        and Plt is not None and Plt < 100
    ):
        add_route(
            routes,
            priority="primary",
            pattern="Infection with thrombocytopenia",
            route="Concurrent inflammatory response and thrombocytopenia increase bleeding and sepsis risk",
            next_steps=[
                "Urgent clinical assessment",
                "Assess for bleeding and sepsis physiology",
                "Repeat CBC and CRP to assess trend"
            ]
        )

    # ---- Infection with significant anaemia ----
    if (
        WBC is not None and WBC > 12
        and CRP is not None and CRP > 20
        and Hb is not None and Hb < 10
    ):
        add_route(
            routes,
            priority="primary",
            pattern="Infection with significant anaemia",
            route="Anaemia may impair oxygen delivery during acute inflammatory illness",
            next_steps=[
                "Urgent clinical assessment",
                "Assess haemodynamic stability",
                "Repeat haemoglobin after acute phase"
            ]
        )

    # ---- Bone marrow failure physiology ----
    if (
        Hb is not None and Hb < 10
        and WBC is not None and WBC < 3
        and Plt is not None and Plt < 100
    ):
        add_route(
            routes,
            priority="primary",
            pattern="Bone marrow failure physiology",
            route="Global suppression of blood cell lines suggests marrow failure or infiltration",
            next_steps=[
                "Urgent peripheral blood smear",
                "Review medications, toxins, and systemic symptoms",
                "Specialist review is indicated"
            ]
        )

    # ---- Multiple simultaneous danger signals ----
    danger_count = sum([
        1 if Hb is not None and Hb < 7 else 0,
        1 if Plt is not None and Plt < 50 else 0,
        1 if WBC is not None and (WBC < 1 or WBC > 30) else 0,
        1 if K is not None and (K < 3.0 or K > 6.0) else 0,
        1 if Na is not None and (Na < 125 or Na > 155) else 0
    ])

    if danger_count >= 2:
        add_route(
            routes,
            priority="primary",
            pattern="Multiple concurrent critical abnormalities",
            route="More than one life-threatening laboratory abnormality detected",
            next_steps=[
                "Immediate senior clinical review",
                "Prioritise stabilisation and monitoring",
                "Repeat critical parameters urgently"
            ]
        )

    # =====================================================
    # FAIL-SAFE — MUST BE LAST
    # =====================================================
    if not routes:
        routes.append({
            "pattern": "Laboratory abnormalities detected",
            "route": "Abnormal findings require clinical correlation",
            "next_steps": [
                "Review results in full clinical context",
                "Consider repeat testing if results are unexpected"
            ]
        })

    # =====================================================
    # PASS 2 — ROUTE DOMINANCE & CLINICAL PRIORITISATION
    # =====================================================

    def route_priority_score(r):
        """
        Lower score = higher clinical priority
        """
        if r.get("priority") == "primary":
            return 0
        if r.get("priority") == "secondary":
            return 1
        return 2

    # ---- Sort routes by clinical priority ----
    routes = sorted(routes, key=route_priority_score)

    # ---- If any PRIMARY routes exist, suppress weak contextual noise ----
    has_primary = any(r.get("priority") == "primary" for r in routes)

    if has_primary:
        filtered = []
        for r in routes:
            # Always keep primary routes
            if r.get("priority") == "primary":
                filtered.append(r)
                continue

            # Keep secondary routes ONLY if clinically reinforcing
            if r.get("priority") == "secondary":
                filtered.append(r)

        routes = filtered

    # ---- Hard cap to avoid cognitive overload ----
    MAX_ROUTES = 5
    routes = routes[:MAX_ROUTES]

    # =====================================================
    # PASS 3 — CLINICAL CONFIDENCE & TEMPORAL FRAMING
    # =====================================================

    def classify_timeframe(route):
        """
        Assigns a clinical time-sensitivity label.
        """
        pattern = (route.get("pattern") or "").lower()

        if any(x in pattern for x in [
            "critical",
            "bone marrow",
            "multiple concurrent",
            "severe",
            "life-threatening"
        ]):
            return "Immediate"

        if any(x in pattern for x in [
            "infection",
            "electrolyte",
            "renal impairment",
            "anaemia with"
        ]):
            return "Urgent"

        return "Routine / Monitor"

    def confidence_language(route):
        """
        Adds senior-clinician-style certainty without diagnosis.
        """
        timeframe = route.get("_timeframe")

        if timeframe == "Immediate":
            return "This pattern is clinically concerning and requires immediate senior assessment."

        if timeframe == "Urgent":
            return "This finding warrants timely clinical review and correlation."

        return "This pattern may be monitored in appropriate clinical context."

    # ---- Apply timeframe + confidence to routes ----
    for r in routes:
        tf = classify_timeframe(r)
        r["_timeframe"] = tf
        r["_confidence"] = confidence_language(r)

    # ---- Escalation summary (one-liner doctors scan) ----
    if routes:
        top = routes[0]
        augmented_summary = f"{top.get('pattern')} — {top.get('_timeframe')} priority."
    else:
        augmented_summary = "No dominant clinical priority identified."

    # ---------------------------
    # Severity / differentials / trends
    # ---------------------------
    # ---------------------------
    # Severity resolution (ROUTE-DOMINANT)
    # ---------------------------
    numeric_sev = evaluate_severity_and_urgency(cdict)
    route_sev = severity_from_routes(routes)
    
    # Escalation order (never downgrade)
    severity_rank = {
        "low": 0,
        "moderate": 1,
        "high": 2,
        "critical": 3
    }
    
    final_severity = route_sev
    if severity_rank.get(numeric_sev["severity"], 0) > severity_rank.get(route_sev, 0):
        final_severity = numeric_sev["severity"]
    
    # ---------------------------
    # CHEMISTRY SEVERITY DOMINANCE (DO NOT DOWNGRADE)
    # ---------------------------
    anion_gap = clean_number(cdict.get("Anion Gap", {}).get("value"))
    bicarb = clean_number(cdict.get("Bicarbonate", {}).get("value"))
    potassium = clean_number(cdict.get("Potassium", {}).get("value"))
    creatinine = clean_number(cdict.get("Creatinine", {}).get("value"))

    print("🧪 CHEM CHECK:",
      "AnionGap=", anion_gap,
      "Bicarb=", bicarb,
      "K=", potassium,
      "Cr=", creatinine)

    
    chemistry_dominant = False
    
    # High–anion–gap metabolic acidosis physiology
    if (
        anion_gap is not None and anion_gap >= 16
        and bicarb is not None and bicarb <= 20
    ):
        chemistry_dominant = True
    
    # Acidosis with electrolyte or renal stress escalates risk
    if chemistry_dominant and (
        (potassium is not None and potassium >= 5.2) or
        (creatinine is not None and creatinine >= 105)
    ):
        # Enforce minimum MODERATE severity
        if severity_rank.get(final_severity, 0) < severity_rank["moderate"]:
            final_severity = "moderate"

    # ---------------------------
    # CHEMISTRY DOMINANT PRIMARY ROUTE
    # ---------------------------
    if chemistry_dominant:
        add_route(
            routes,
            priority="primary",
            pattern="High–anion–gap metabolic acidosis physiology",
            route=(
                "Elevated anion gap with reduced bicarbonate indicates a dominant "
                "metabolic acidosis process with associated electrolyte and renal stress."
            ),
            next_steps=[
                "Urgent clinical assessment",
                "Evaluate causes of high–anion–gap metabolic acidosis",
                "Monitor electrolytes and renal function closely"
            ]
        )


    sev = dict(numeric_sev)
    sev["severity"] = final_severity

    diffs = generate_differential_trees(cdict)

    patient_name = None
    try:
        patient_name = (ai_json.get("patient") or {}).get("name")
    except Exception:
        pass

    trends = trend_comparison(patient_name, cdict)
    # ---------------------------
    # Clean follow-up block (summary-level, non-invasive)
    # ---------------------------
    follow_up = build_follow_up_block(
        cdict=cdict,
        routes=routes,
        severity=sev.get("severity")
    )


    # ---------------------------
    # Chemistry context & next steps
    # ---------------------------
    chemistry_context = None
    chemistry_next_steps = None

    if ai_json.get("_chemistry_status") in ("present", "assumed_from_text"):
        chemistry_context = []
        chemistry_next_steps = []

        bilirubin = v("Bilirubin")
        alt = v("ALT")
        ast = v("AST")
        alp = v("ALP")
        ggt = v("GGT")
        crp = v("CRP")
        triglycerides = v("Triglycerides")

        ldl = (
            v("LDL")
            or v("LDL Chol")
            or v("LDL Chol (direct)")
        )

        def ref_high(k):
            return clean_number(cdict.get(k, {}).get("reference_high"))

        if bilirubin is not None and bilirubin > 21:
            if all(
                x is not None and
                (ref_high(k) is None or x <= ref_high(k))
                for k, x in [("ALT", alt), ("AST", ast), ("ALP", alp), ("GGT", ggt)]
            ):
                chemistry_context.append(
                    "Unconjugated hyperbilirubinaemia with normal liver enzymes is commonly benign "
                    "(e.g. Gilbert syndrome), particularly if intermittent."
                )

        if crp is not None and crp < 5:
            chemistry_context.append(
                "Normal CRP reduces the likelihood of acute inflammatory or infectious pathology."
            )

        age = cdict.get("_patient_age")

        if triglycerides is not None or ldl is not None:
            if age is not None and age < 40:
                chemistry_context.append(
                    "At this age, absolute short-term cardiovascular risk is generally low; "
                    "lifestyle optimisation is appropriate as first-line management."
                )
            else:
                chemistry_context.append(
                    "Lipid abnormalities suggest increased long-term cardiovascular risk rather than acute illness."
                )

            chemistry_next_steps.append(
                "Repeat fasting lipid profile in 3–6 months if clinically appropriate."
            )
            chemistry_next_steps.append(
                "Consider fasting status and recent alcohol intake when interpreting triglyceride levels."
            )

        if bilirubin is not None and bilirubin > 21:
            chemistry_next_steps.append(
                "If bilirubin remains elevated, consider repeat fractionation ± reticulocyte count if clinically indicated."
            )
    # ---------------------------
    # CHEMISTRY PRIORITY FRAMING (SUMMARY AUGMENT)
    # ---------------------------
    if chemistry_dominant:
        dominant_line = (
            "The dominant abnormality is high–anion–gap metabolic acidosis "
            "with associated electrolyte and renal stress, which warrants "
            "urgent clinical assessment due to risk of deterioration."
        )
    
        if isinstance(ai_json.get("summary"), dict):
            existing = ai_json["summary"].get("impression", "")
            if dominant_line not in existing:
                ai_json["summary"]["impression"] = (
                    dominant_line if not existing
                    else f"{existing} {dominant_line}"
                )


    # ---------------------------
    # Final assembly
    # ---------------------------
    augmented = dict(ai_json)
    augmented["_canonical_cbc"] = cdict
    augmented["_routes"] = routes
    augmented["_severity"] = sev
    augmented["_differential_trees"] = diffs
    augmented["_trend_comparison"] = trends
    augmented["_clinical_context"] = chemistry_context
    augmented["_suggested_next_steps"] = chemistry_next_steps
    augmented["_clinical_patterns"] = patterns
    augmented["_generated_at"] = iso_now()
    augmented["_overall_status"] = overall_status
    augmented["_follow_up"] = follow_up

    return augmented









# ---------------------------
# Main report processing
# ---------------------------
def process_report(job: dict) -> dict:
    report_id = job.get("id")
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""

    print(f"Processing report {report_id} ...")

    if not file_path:
        err = f"Missing file_path for report {report_id}"
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": err
        }).eq("id", report_id).execute()
        return {"error": err}

    try:
        # --------------------
        # Download PDF
        # --------------------
        pdf_res = supabase.storage.from_(BUCKET).download(file_path)
        pdf_bytes = pdf_res.data if hasattr(pdf_res, "data") else pdf_res

        text = extract_text_from_pdf(pdf_bytes)
        scanned = is_scanned_pdf(text)
        print(f"Report {report_id}: scanned={scanned}, text_len={len(text)}")

        extracted_rows = []
        ocr_text_chunks = []
        merged_text_for_ai = ""

        # --------------------
        # OCR or text parsing
        # --------------------
        if scanned:
            print("SCANNED PDF → OCR")
            ocr_text_chunks = []
            pages = convert_from_bytes(pdf_bytes)
        
            for i, page_img in enumerate(pages, start=1):
                buf = io.BytesIO()
                page_img.save(buf, format="PNG")
                ocr_out = extract_cbc_from_image(buf.getvalue())
                extracted_rows.extend(ocr_out.get("cbc", []))
                ocr_text_chunks.append(ocr_out.get("raw_text", ""))
        
            if not extracted_rows:
                raise ValueError("No CBC extracted from scanned PDF")
        
            merged_text_for_ai = json.dumps(
                {"cbc": extracted_rows},
                ensure_ascii=False
            )
        
            ocr_identity_text = "\n".join(ocr_text_chunks)
        
        else:
            merged_text_for_ai = text or l_text
            if not merged_text_for_ai.strip():
                raise ValueError("No usable text extracted from digital PDF")
        
        # --------------------
        # AI interpretation
        # --------------------
        print("Calling AI interpretation...")
        ai_json = call_ai_on_report(merged_text_for_ai)
        
        # --------------------
        # Patient demographics extraction (THIS IS THE KEY)
        # --------------------
        raw_text_for_patient = text if not scanned else ocr_identity_text
        
        patient = extract_patient_demographics(raw_text_for_patient)
        
        print("🧾 Extracted patient demographics:", patient)
        
        if isinstance(ai_json.get("patient"), dict):
            for k, v in patient.items():
                if ai_json["patient"].get(k) in (None, "Unknown"):
                    ai_json["patient"][k] = v
        else:
            ai_json["patient"] = patient

        # --------------------
        # Patient demographics extraction (from raw PDF / OCR text)
        # --------------------
        raw_text_for_patient = text if not scanned else merged_text_for_ai
        patient = extract_patient_demographics(raw_text_for_patient)

        if isinstance(ai_json.get("patient"), dict):
            for k, v in patient.items():
                if ai_json["patient"].get(k) in (None, "Unknown"):
                    ai_json["patient"][k] = v
        else:
            ai_json["patient"] = patient

        # --------------------
        # HARD FAILSAFE: force CBC extraction if missing
        # --------------------
        if not ai_json.get("cbc"):
            print("⚠️ No structured CBC detected — forcing extraction")
        
            if extracted_rows:
                ai_json["cbc"] = extracted_rows
                ai_json["_cbc_status"] = "forced_from_ocr"
            else:
                raise ValueError("CBC missing after AI interpretation — cannot proceed safely")


        # ---- CBC sanity check (doctor-grade) ----
        cbc_rows = ai_json.get("cbc") or []

        cbc_present = any(
            any(
                key in (r.get("analyte") or r.get("test") or r.get("name") or "").lower()
                for key in (
                    "hb", "hemoglobin", "haemoglobin",
                    "wbc", "white", "leuko",
                    "platelet", "plt"
                )
            )
            for r in cbc_rows
            if isinstance(r, dict)
        )

        # ---- Chemistry detection (doctor-grade) ----
        chemistry_keys = (
            "crp", "creatinine", "egfr",
            "bilirubin", "alt", "ast", "alp", "ggt",
            "cholesterol", "ldl", "hdl", "triglyceride",
            "albumin", "total protein", "globulin"
        )

        chemistry_present = any(
            any(
                key in (r.get("analyte") or r.get("test") or r.get("name") or "").lower()
                for key in chemistry_keys
            )
            for r in cbc_rows
            if isinstance(r, dict)
        )

        # ---- Allow chemistry-only interpretation for DIGITAL PDFs ----
        if not cbc_present and not chemistry_present:
            if scanned:
                raise ValueError(
                    "No interpretable laboratory data extracted — interpretation blocked"
                )
            else:
                # Chemistry likely exists in text (digital PDF)
                ai_json["_cbc_status"] = "missing"
                ai_json["_chemistry_status"] = "assumed_from_text"
        else:
            ai_json["_cbc_status"] = "present" if cbc_present else "missing"
            ai_json["_chemistry_status"] = "present" if chemistry_present else "missing"

        # --------------------
        # Clinical augmentation
        # --------------------
        print("Building clinical augmentation...")
        augmented = build_full_clinical_report(ai_json)



        

        # --------------------
        # Store results + persist patient demographics
        # --------------------
        patient = augmented.get("patient") or {}
        
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": augmented,
            "ai_error": None,
        
            # 🔒 Persist patient fields explicitly
            "name": patient.get("name"),
            "age": patient.get("age"),
            "sex": patient.get("sex"),
        }).eq("id", report_id).execute()

        print(f"✅ Report {report_id} completed")
        return {"success": True, "data": augmented}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        traceback.print_exc()
        supabase.table("reports").update({
            "ai_status": "failed",
            "ai_error": err
        }).eq("id", report_id).execute()
        return {"error": err}


# ---------------------------
# Main worker loop (fixed .model handling)
# ---------------------------
def main():
    print("Entering main loop...")

    # ---- startup sanity check ----
    chk = supabase.table("reports").select("id").eq("ai_status", "pending").limit(5).execute()
    print("🔍 Pending jobs at startup:", chk.data)

    while True:
        try:
            # ALWAYS read .data (supabase-py v2)
            res = supabase.table("reports") \
                .select("*") \
                .eq("ai_status", "pending") \
                .limit(1) \
                .execute()

            jobs = res.data if hasattr(res, "data") else []

            if not jobs:
                print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            job_id = job.get("id")
            print(f"🔎 Found job {job_id}")

            # ---- atomic claim (prevents race + stuck jobs) ----
            claim = supabase.table("reports") \
                .update({"ai_status": "processing"}) \
                .eq("id", job_id) \
                .eq("ai_status", "pending") \
                .execute()

            if not claim.data:
                print("⚠️ Job already claimed by another worker, skipping")
                continue

            process_report(job)

        except Exception as e:
            print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(2)

if __name__ == "__main__":
    main()
