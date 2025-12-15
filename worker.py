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
    Convert ai_json['cbc'] list into canonical dictionary keyed by common names (Hb, MCV, MCH, WBC, Neutrophils, etc.)
    """
    out = {}
    rows = ai_json.get("cbc") or []
    for r in rows:
        if not isinstance(r, dict):
            continue
        name = (r.get("analyte") or r.get("test") or "").strip().lower()
        if not name:
            continue

        def put(k):
            if k not in out:
                out[k] = r

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
        elif "platelet" in name or "plt" in name:
            put("Platelets")
        elif "creatinine" in name:
            put("Creatinine")
        elif name.startswith("urea"):
            put("Urea")
        elif "alt" in name and "alanine" in name:
            put("ALT")
        elif name == "alt":
            put("ALT")
        elif "ast" in name:
            put("AST")
        elif "alp" in name or "alkaline phosphatase" in name:
            put("ALP")
        elif "ggt" in name or "gamma" in name:
            put("GGT")
        elif "biliru" in name:
            put("Bilirubin")
        elif "creatine kinase" in name or name == "ck":
            put("CK")
        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")
        elif "sodium" in name or name == "na":
            put("Sodium")
        elif "potassium" in name or name == "k":
            put("Potassium")
        elif "calcium" in name:
            put("Calcium")
        elif "crp" in name:
            put("CRP")

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
        res = supabase.table("reports").select("*").eq("ai_status", "completed").order("created_at", {"ascending": False}).limit(20).execute()
        prior = res.model or []
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

# ---------------------------
# Route engine aggregator: Patterns -> Route -> Next Steps
# also includes severity/urgency/differential/trends
# ---------------------------
def build_full_clinical_report(ai_json: dict) -> dict:
    """
    Given ai_json (from call_ai_on_report), add:
    - canonical cdict
    - routes (list of pattern dicts)
    - severity & urgency
    - differential_trees
    - trend_comparison (if patient name found)
    """
    # canonical dict
    cdict = build_cbc_value_dict(ai_json)
    overall_status = overall_clinical_status(cdict)


    # routes (reuse earlier generate_routes logic but return rich objects)
    routes = []
    # simple mapping to reuse code above (we'll implement inline)
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    Hb = v("Hb"); MCV = v("MCV"); WBC = v("WBC"); Neut = v("Neutrophils"); Lymph = v("Lymphocytes")
    Plt = v("Platelets"); Cr = v("Creatinine"); CRP = v("CRP"); CK = v("CK"); Na = v("Sodium"); K = v("Potassium")

    # Anaemia
    if Hb is not None and Hb < 13:
        if MCV is not None:
            if MCV < 80:
                routes.append({
                    "pattern": "Microcytic anaemia",
                    "route": "Likely iron deficiency / chronic disease pattern",
                    "next_steps": ["Order ferritin & iron studies", "Reticulocyte count", "Consider inflammation markers (CRP)"]
                })
            elif 80 <= MCV <= 100:
                routes.append({
                    "pattern": "Normocytic anaemia",
                    "route": "Possible chronic disease, renal, or early iron deficiency",
                    "next_steps": ["Check creatinine & eGFR", "Reticulocyte count", "Clinical correlation"]
                })
            else:
                routes.append({
                    "pattern": "Macrocytic anaemia",
                    "route": "Possible B12/folate deficiency or hepatic/drug effect",
                    "next_steps": ["Order B12 & folate", "Review liver enzymes", "Medication review"]
                })
        else:
            routes.append({
                "pattern": "Anaemia (MCV unknown)",
                "route": "Low haemoglobin — further classification needed",
                "next_steps": ["Obtain MCV/MCH", "Order ferritin & reticulocytes"]
            })

    # WBC
    if WBC is not None:
        if WBC > 12:
            detail = "Inflammatory/infective physiology"
            nexts = []
            if Neut and Neut > 70:
                detail = "Neutrophil-predominant — bacterial pattern more likely"
                nexts.append("Correlate with fever, localising signs; consider antibiotics per clinical context")
            if Lymph and Lymph > 45:
                nexts.append("Consider viral causes; review symptom timeline")
            if not nexts:
                nexts.append("Correlate clinically; consider CRP and cultures if indicated")
            routes.append({"pattern": "Leucocytosis", "route": detail, "next_steps": nexts})

        elif WBC < 4:
            routes.append({"pattern": "Leukopenia", "route": "Viral suppression, bone marrow effect, or drugs", "next_steps": ["Medication review", "Repeat CBC", "Consider specialist review if persistent"]})

    # Platelets
    if Plt is not None:
        if Plt < 150:
            routes.append({"pattern": "Thrombocytopenia", "route": "Bleeding risk assessment", "next_steps": ["Assess bleeding symptoms", "Review drugs and previous CBCs", "Consider haematology review if <50"]})
        elif Plt > 450:
            routes.append({"pattern": "Thrombocytosis", "route": "Reactive vs primary thrombocytosis", "next_steps": ["Check CRP/INFLAMMATION", "Repeat CBC", "Consider iron studies"]})

    # Kidney / LFT / CK
    if Cr is not None and Cr > 120:
        routes.append({"pattern": "Renal impairment physiology", "route": "Assess for AKI or CKD", "next_steps": ["Repeat U&E", "Review meds & hydration", "Consider eGFR"]})

    if CK is not None and CK > 1000:
        routes.append({"pattern": "High CK", "route": "Muscle injury / rhabdomyolysis physiology", "next_steps": ["Check creatinine", "Assess for muscle pain/trauma", "Urgent review if creatinine rising"]})

    # Electrolytes urgency included separately in severity function

    if not routes:
        routes.append({"pattern": "No major patterns detected", "route": "Results within expected ranges", "next_steps": ["Correlate clinically and review prior results"]})

    # severity & urgency
    sev = evaluate_severity_and_urgency(cdict)

    # differential trees
    diffs = generate_differential_trees(cdict)

    # trend comparison
    patient_name = None
    try:
        patient_name = (ai_json.get("patient") or {}).get("name")
    except:
        patient_name = None
    trends = trend_comparison(patient_name, cdict)

    # Compose final augmented report
    augmented = dict(ai_json)  # shallow copy
    augmented["_canonical_cbc"] = cdict
    augmented["_routes"] = routes
    augmented["_severity"] = sev
    augmented["_differential_trees"] = diffs
    augmented["_trend_comparison"] = trends
    augmented["_generated_at"] = iso_now()
    augmented["_overall_status"] = overall_status

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
        merged_text_for_ai = ""

        # --------------------
        # OCR or text parsing
        # --------------------
        if scanned:
            print("SCANNED PDF → OCR")
            pages = convert_from_bytes(pdf_bytes)
            for i, page_img in enumerate(pages, start=1):
                buf = io.BytesIO()
                page_img.save(buf, format="PNG")
                ocr_out = extract_cbc_from_image(buf.getvalue())
                extracted_rows.extend(ocr_out.get("cbc", []))

            if not extracted_rows:
                raise ValueError("No CBC extracted from scanned PDF")

            merged_text_for_ai = json.dumps(
                {"cbc": extracted_rows},
                ensure_ascii=False
            )

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
        # CBC sanity check (doctor-grade)
        # --------------------
        cbc_rows = ai_json.get("cbc") or []

        cbc_present = any(
            any(
                key in (r.get("analyte") or "").lower()
                for key in (
                    "hb", "hemoglobin", "haemoglobin",
                    "wbc", "white", "leuko",
                    "platelet", "plt"
                )
            )
            for r in cbc_rows
            if isinstance(r, dict)
        )

        if not cbc_present:
            raise ValueError(
                "CBC expected but not extracted — interpretation blocked"
            )

        # --------------------
        # Clinical augmentation
        # --------------------
        print("Building clinical augmentation...")
        augmented = build_full_clinical_report(ai_json)

        # --------------------
        # Store results
        # --------------------
        supabase.table("reports").update({
            "ai_status": "completed",
            "ai_results": augmented,
            "ai_error": None
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
