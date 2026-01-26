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

def tag_value_source(row: dict, source: str) -> dict:
    """
    Annotate a lab row with its source domain.
    Source examples: 'cbc', 'chemistry', 'abg', 'coox'
    """
    if not isinstance(row, dict):
        return row

    r = dict(row)
    r["_source"] = source
    return r


def assess_data_integrity(cdict: dict) -> dict:
    """
    Read-only data integrity check.
    Determines whether the available laboratory data are internally consistent
    and physiologically interpretable.

    This function:
    - Does NOT interpret results
    - Does NOT change severity, routes, or patterns
    - Does NOT assume missing data
    - Only reports confidence and limitations

    Returns:
    {
        "status": "consistent | limited | discordant",
        "notes": [ "...", "..." ]
    }
    """

    notes = []
    status = "consistent"

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    # Core values (only if present)
    Hb = v("Hb")
    Na = v("Sodium")
    K = v("Potassium")
    Cr = v("Creatinine")
    HCO3 = v("Bicarbonate")
    AG = v("Anion Gap")
    WBC = v("WBC")
    Neut = v("Neutrophils")

    # ----------------------------
    # Unit / scale sanity checks
    # ----------------------------
    if Hb is not None and Hb > 30:
        notes.append(
            "Haemoglobin value appears unusually high for g/dL; unit or scale inconsistency possible."
        )
        status = "limited"

    if K is not None and K > 10:
        notes.append(
            "Potassium value exceeds physiologically plausible range; unit or transcription error possible."
        )
        status = "discordant"

    # ----------------------------
    # Internal consistency checks
    # ----------------------------
    if AG is not None and HCO3 is not None:
        if AG >= 16 and HCO3 >= 26:
            notes.append(
                "Elevated anion gap with normal bicarbonate is physiologically discordant."
            )
            status = "limited"

    if WBC is not None and Neut is not None:
        if Neut > WBC:
            notes.append(
                "Neutrophil count exceeds total white cell count; absolute vs percentage mismatch possible."
            )
            status = "limited"

    # ----------------------------
    # Missing anchor awareness
    # ----------------------------
    if K is not None and HCO3 is None:
        notes.append(
            "Potassium interpretation is limited without accompanying acid–base data."
        )
        status = "limited"

    if Cr is not None and cdict.get("_patient_age") is None:
        notes.append(
            "Renal function interpretation is limited without age or baseline creatinine."
        )
        status = "limited"

    if not notes:
        notes.append(
            "Available laboratory data are internally consistent and physiologically interpretable."
        )

    return {
        "status": status,
        "notes": notes
    }

def assess_pattern_strength(routes: list, cdict: dict) -> list:
    """
    Read-only pattern strength calibration.

    This function:
    - Reads EXISTING routes
    - Does NOT modify routes
    - Does NOT affect severity or urgency
    - Adds descriptive confidence only

    Returns a list aligned 1:1 with routes.
    """

    results = []

    for r in routes:
        pattern = (r.get("pattern") or "").lower()
        strength = "Suggested pattern, limited by available data."

        # Strong, multi-parameter or danger patterns
        if any(x in pattern for x in [
            "acute inflammatory",
            "severe anaemia",
            "critical",
            "bone marrow",
            "high–anion–gap",
            "multiple concurrent"
        ]):
            strength = "Well-supported laboratory pattern."

        # Common but context-dependent patterns
        elif any(x in pattern for x in [
            "anaemia",
            "leucocytosis",
            "renal impairment",
            "electrolyte"
        ]):
            strength = "Moderately supported laboratory pattern."

        # Data-poor situations
        if not cdict or len(cdict.keys()) < 3:
            strength = "Suggested pattern, limited by available data."

        results.append({
            "pattern": r.get("pattern"),
            "strength": strength
        })

def assess_interpretation_boundaries(cdict: dict, routes: list) -> list:
    """
    Read-only interpretation boundary detection.

    Identifies missing contextual anchors that limit how confidently
    current laboratory patterns can be interpreted.

    Does NOT:
    - change routes
    - affect severity
    - recommend actions
    - assume normality

    Returns a list of limitation statements.
    """

    limits = []
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    # ----------------------------
    # Baseline / trend limitations
    # ----------------------------
    if cdict.get("_patient_age") is None:
        limits.append(
            "Interpretation is limited by absence of patient age."
        )

    # No historical trend info available
    limits.append(
        "Interpretation is based on a single time-point; trend data are not available."
    )

    # ----------------------------
    # Renal interpretation limits
    # ----------------------------
    Cr = v("Creatinine")
    if Cr is not None:
        limits.append(
            "Renal interpretation is limited without a known baseline creatinine."
        )

    # ----------------------------
    # Electrolyte context limits
    # ----------------------------
    K = v("Potassium")
    HCO3 = v("Bicarbonate")

    if K is not None and HCO3 is None:
        limits.append(
            "Electrolyte interpretation is limited without accompanying acid–base data."
        )

    # ----------------------------
    # Differential count limits
    # ----------------------------
    Neut = v("Neutrophils")
    WBC = v("WBC")

    if Neut is not None and WBC is not None:
        limits.append(
            "Differential interpretation may be limited without explicit absolute counts."
        )

    # ----------------------------
    # Pattern-driven missing context
    # ----------------------------
    for r in routes:
        p = (r.get("pattern") or "").lower()

        if "potassium" in p:
            limits.append(
                "Potassium abnormalities are best interpreted with ECG correlation when available."
            )

        if "acid" in p or "anion" in p:
            limits.append(
                "Acid–base patterns are best interpreted with arterial or venous blood gas data when available."
            )

    # ----------------------------
    # De-duplicate while preserving order
    # ----------------------------
    seen = set()
    deduped = []
    for l in limits:
        if l not in seen:
            deduped.append(l)
            seen.add(l)
            
    return deduped

def assess_severity_stability(cdict: dict, routes: list, severity: dict) -> dict:
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    active_domains = 0

    # CBC domain
    if any(v(k) is not None for k in ("Hb", "WBC", "Platelets")):
        active_domains += 1

    # Chemistry domain
    if any(v(k) is not None for k in ("Creatinine", "CRP", "Sodium", "Potassium")):
        active_domains += 1

    # ABG domain
    if any(v(k) is not None for k in ("pH", "Bicarbonate", "Anion Gap")):
        active_domains += 1

    if severity.get("severity") == "high" and active_domains == 1:
        severity["severity"] = "moderate"
        severity["note"] = "Severity capped due to single-domain abnormality."

    return severity

def assess_abg_coherence(cdict: dict, abg: dict | None) -> list:
    """
    Read-only ABG coherence assessment.

    Purpose:
    - Reinforce or limit interpretation of chemistry-derived acid–base patterns
    - Never generate routes
    - Never alter severity
    """

    notes = []

    if not abg or not isinstance(abg, dict):
        return notes

    def v(key):
        return clean_number(abg.get(key))

    pH = v("pH")
    pCO2 = v("pCO2")
    HCO3_abg = v("HCO3")

    HCO3_chem = clean_number(cdict.get("Bicarbonate", {}).get("value"))
    AG = clean_number(cdict.get("Anion Gap", {}).get("value"))

    # --- Acidaemia coherence ---
    if pH is not None and HCO3_chem is not None:
        if pH < 7.35 and HCO3_chem < 22:
            notes.append(
                "ABG acidaemia is physiologically consistent with reduced serum bicarbonate."
            )

    # --- High anion gap reinforcement ---
    if pH is not None and AG is not None:
        if pH < 7.35 and AG > 16:
            notes.append(
                "ABG acidaemia aligns with a high–anion-gap metabolic acidosis pattern."
            )

    # --- Limiting statements ---
    if pH is not None and pH >= 7.35 and HCO3_chem is not None and HCO3_chem < 22:
        notes.append(
            "Normal pH limits confidence in the severity of metabolic acidosis without compensation assessment."
        )

    if pH is None:
        notes.append(
            "Acid–base interpretation is limited in the absence of arterial or venous blood gas pH."
        )

    return notes

def assess_ecg_coherence(cdict: dict, ecg: dict | None) -> list:
    """
    Read-only ECG coherence assessment.

    Purpose:
    - Correlate ECG flags with electrolyte-related risk
    - Never generate routes
    - Never alter severity
    """

    notes = []

    if not ecg or not isinstance(ecg, dict):
        return notes

    flags = ecg.get("flags") or []
    if not isinstance(flags, list):
        return notes

    K = clean_number(cdict.get("Potassium", {}).get("value"))
    Na = clean_number(cdict.get("Sodium", {}).get("value"))

    # --- Potassium risk coherence ---
    if K is not None and (K < 3.3 or K > 5.5):
        if flags:
            notes.append(
                "ECG flags present alongside potassium abnormality, supporting increased arrhythmic risk."
            )
        else:
            notes.append(
                "Potassium abnormality without ECG abnormalities limits arrhythmia risk assessment."
            )

    # --- Sodium-related neurological risk ---
    if Na is not None and (Na < 125 or Na > 155):
        if flags:
            notes.append(
                "ECG abnormalities present in the setting of severe sodium disturbance, increasing overall clinical risk."
            )
            
def assess_abg_physiology(abg: dict | None) -> list:
    """
    Read-only ABG physiology assessment.

    Evaluates:
    - Primary acid–base disturbance
    - Expected respiratory compensation (Winter’s formula)
    - Possible mixed disorders

    Does NOT:
    - Diagnose causes
    - Recommend treatment
    - Alter severity or routes
    """

    notes = []

    if not abg or not isinstance(abg, dict):
        return notes

    pH = clean_number(abg.get("pH"))
    HCO3 = clean_number(abg.get("HCO3"))
    pCO2 = clean_number(abg.get("pCO2"))

    if pH is None or HCO3 is None:
        notes.append(
            "ABG interpretation is limited by missing pH or bicarbonate values."
        )
        return notes

def assess_offsetting_contexts(cdict: dict, abg: dict | None, ecg: dict | None) -> list:
    """
    Detects physiological offsetting or buffering patterns.

    Purpose:
    - Highlight when an abnormal value is contextually moderated
    - Reduce false alarm bias without suppressing findings

    Does NOT:
    - Change severity
    - Remove routes
    - Override alerts
    """

    notes = []
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    # ----------------------------
    # Potassium vs ECG buffering
    # ----------------------------
    K = v("Potassium")
    if K is not None and K > 5.5:
        if not ecg:
            notes.append(
                "Potassium is elevated; ECG correlation is not available to assess electrophysiological impact."
            )
        elif ecg.get("abnormalities") in (None, [], "normal"):
            notes.append(
                "Potassium is elevated, but no ECG abnormalities are noted, which may indicate limited immediate electrophysiological effect."
            )

    # ----------------------------
    # Metabolic acidosis with compensation
    # ----------------------------
    HCO3 = v("Bicarbonate")
    anion_gap = v("Anion Gap")
    if HCO3 is not None and HCO3 < 22 and anion_gap is not None and anion_gap >= 16:
        if abg and abg.get("_compensation") == "appropriate":
            notes.append(
                "Metabolic acidosis is present with appropriate respiratory compensation, which may partially buffer physiological impact."
            )

    # ----------------------------
    # Renal impairment buffering
    # ----------------------------
    Cr = v("Creatinine")
    if Cr is not None and Cr > 120:
        if K is not None and 3.5 <= K <= 5.1 and HCO3 is not None and HCO3 >= 22:
            notes.append(
                "Creatinine is elevated, but potassium and bicarbonate remain stable, suggesting preserved metabolic buffering."
            )

    # ----------------------------
    # Inflammatory marker offset
    # ----------------------------
    CRP = v("CRP")
    WBC = v("WBC")
    if CRP is not None and CRP > 30:
        if WBC is not None and WBC < 12:
            notes.append(
                "CRP is elevated without accompanying leukocytosis, which may reflect a non-bacterial or evolving inflammatory process."
            )

    return notes


    # ---------------------------
    # Primary disorder
    # ---------------------------
    if pH < 7.35 and HCO3 < 22:
        notes.append(
            "Primary metabolic acidosis physiology is present."
        )

        if pCO2 is not None:
            expected_pCO2_low = (1.5 * HCO3) + 6
            expected_pCO2_high = (1.5 * HCO3) + 10

            if pCO2 < expected_pCO2_low:
                notes.append(
                    "pCO₂ is lower than expected for compensation, suggesting a concurrent respiratory alkalosis."
                )
            elif pCO2 > expected_pCO2_high:
                notes.append(
                    "pCO₂ is higher than expected for compensation, suggesting a concurrent respiratory acidosis."
                )
            else:
                notes.append(
                    "Respiratory compensation appears appropriate for metabolic acidosis."
                )

    elif pH > 7.45 and HCO3 > 26:
        notes.append(
            "Primary metabolic alkalosis physiology is present."
        )

        if pCO2 is not None:
            notes.append(
                "Respiratory compensation may be present; correlation with clinical context is required."
            )

    else:
        notes.append(
            "No dominant metabolic acid–base disturbance detected on ABG."
        )

    return notes


def build_explainability_floor(
    cdict: dict,
    routes: list,
    severity: dict,
    interpretation_boundaries: list
) -> list:
    explanations = []

    # Severity rationale
    explanations.append(
        f"Overall severity assessed as '{severity.get('severity')}' based on combined laboratory signals and route dominance."
    )

    # Route rationale
    if routes:
        explanations.append(
            "Clinical routes were generated from detected laboratory patterns without diagnostic assumptions."
        )
    else:
        explanations.append(
            "No dominant clinical routes identified based on available data."
        )

    # Boundary rationale
    for b in interpretation_boundaries:
        explanations.append(b)

    return explanations




def derive_dominant_driver(routes: list) -> dict:
    """
    Read-only dominant driver identification.

    Purpose:
    - Provide rapid clinical orientation only
    - Does NOT override, suppress, or reorder routes
    - Does NOT change severity or urgency

    Returns:
    {
        "driver": str | None,
        "confidence": "clear | competing | none"
    }
    """

    if not routes:
        return {
            "driver": None,
            "confidence": "none"
        }

    # Look only at EXISTING primary routes
    primary_routes = [
        r for r in routes
        if r.get("priority") == "primary"
    ]

    if len(primary_routes) == 1:
        return {
            "driver": primary_routes[0].get("pattern"),
            "confidence": "clear"
        }

    if len(primary_routes) > 1:
        return {
            "driver": primary_routes[0].get("pattern"),
            "confidence": "competing"
        }

    # No primary routes → no dominant driver
    return {
        "driver": None,
        "confidence": "none"
    }

def assess_acid_base_coherence(cdict: dict) -> dict | None:
    """
    Read-only acid–base coherence assessment.

    Purpose:
    - Identify whether electrolyte / renal findings are physiologically coherent
    - Highlight offsets (e.g. potassium vs pH, bicarbonate vs anion gap)
    - NEVER diagnose
    - NEVER override severity or routes

    Returns None if insufficient data.
    """

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    pH = v("pH")
    pCO2 = v("pCO2")
    HCO3 = v("Bicarbonate")
    AG = v("Anion Gap")
    K = v("Potassium")
    Cr = v("Creatinine")

    # Require at least bicarbonate or pH to proceed
    if pH is None and HCO3 is None:
        return None

    notes = []

    # ----------------------------
    # Metabolic acidosis coherence
    # ----------------------------
    if HCO3 is not None and HCO3 < 22:
        if AG is not None and AG >= 16:
            notes.append(
                "Low bicarbonate with elevated anion gap is physiologically coherent with a high–anion–gap metabolic acidosis."
            )
        else:
            notes.append(
                "Low bicarbonate without an elevated anion gap suggests a non–anion–gap metabolic acidosis physiology."
            )

    # ----------------------------
    # Potassium offset awareness
    # ----------------------------
    if K is not None and HCO3 is not None:
        if HCO3 < 22 and K <= 5.0:
            notes.append(
                "Potassium is not elevated despite metabolic acidosis, which may reflect early disease, renal compensation, or intracellular shifts."
            )

        if HCO3 >= 22 and K > 5.0:
            notes.append(
                "Elevated potassium without metabolic acidosis suggests a non–acidotic driver (e.g. renal or medication-related)."
            )

    # ----------------------------
    # Renal–acid base interaction
    # ----------------------------
    if Cr is not None and Cr > 110 and HCO3 is not None and HCO3 < 22:
        notes.append(
            "Renal impairment with reduced bicarbonate suggests impaired acid clearance contributing to acid–base disturbance."
        )

    if not notes:
        return None

    return {
        "acid_base_coherence": notes
    }
    
def assess_ecg_coherence(cdict: dict, ecg_data: dict | None) -> dict | None:
    """
    Read-only ECG coherence assessment.

    Purpose:
    - Relate electrolyte abnormalities to potential electrical risk
    - Orientation only
    - NEVER interpret rhythm
    - NEVER diagnose
    - NEVER override severity or routes

    Returns None if ECG data or relevant context is absent.
    """

    if not ecg_data or not isinstance(ecg_data, dict):
        return None

    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    potassium = v("Potassium")
    calcium = v("Calcium")
    magnesium = v("Magnesium")

    qtc = clean_number(ecg_data.get("QTc"))
    heart_rate = clean_number(ecg_data.get("heart_rate"))

    notes = []

    # ----------------------------
    # Potassium ↔ electrical risk
    # ----------------------------
    if potassium is not None:
        if potassium >= 5.5:
            notes.append(
                "Elevated potassium is associated with increased risk of conduction abnormalities; ECG correlation is important."
            )
        elif potassium <= 3.0:
            notes.append(
                "Low potassium may increase susceptibility to ventricular arrhythmias, particularly in the presence of QT prolongation."
            )

    # ----------------------------
    # QT context (if available)
    # ----------------------------
    if qtc is not None and qtc >= 480:
        notes.append(
            "Prolonged QT interval increases arrhythmic risk, especially in the presence of electrolyte disturbances."
        )

    # ----------------------------
    # Calcium / magnesium context
    # ----------------------------
    if calcium is not None and calcium < 2.0:
        notes.append(
            "Low calcium may contribute to QT prolongation and electrical instability."
        )

    if magnesium is not None and magnesium < 0.7:
        notes.append(
            "Low magnesium may predispose to ventricular arrhythmias."
        )

    # ----------------------------
    # Heart rate amplification
    # ----------------------------
    if heart_rate is not None and heart_rate > 120:
        notes.append(
            "Tachycardia may amplify the electrical effects of electrolyte abnormalities."
        )

    if not notes:
        return None

    return {
        "ecg_coherence": notes
    }

def assess_notes_coherence(notes_text: str | None, routes: list) -> dict | None:
    """
    Read-only clinical notes coherence assessment.

    Purpose:
    - Detect alignment or mismatch between notes and lab/ECG-derived physiology
    - Highlight missing contextual anchors
    - NEVER diagnose
    - NEVER recommend treatment
    - NEVER override other modalities
    """

    if not notes_text or not isinstance(notes_text, str):
        return None

    t = notes_text.lower()
    notes = []

    # ----------------------------
    # Infective / inflammatory context
    # ----------------------------
    if any(x in t for x in ["fever", "rigors", "sepsis", "infection"]):
        if any("infection" in (r.get("pattern") or "").lower() for r in routes):
            notes.append(
                "Clinical notes describe infective symptoms that align with inflammatory laboratory patterns."
            )
        else:
            notes.append(
                "Clinical notes describe infective symptoms, but laboratory markers do not show a strong inflammatory response."
            )

    # ----------------------------
    # Cardiorespiratory symptoms
    # ----------------------------
    if any(x in t for x in ["chest pain", "palpitations", "syncope", "shortness of breath"]):
        notes.append(
            "Cardiorespiratory symptoms are described; correlation with ECG and electrolyte findings is important."
        )

    # ----------------------------
    # Gastrointestinal losses
    # ----------------------------
    if any(x in t for x in ["vomiting", "diarrhoea"]):
        notes.append(
            "Gastrointestinal losses described in notes may contribute to electrolyte or acid–base disturbances."
        )

    # ----------------------------
    # Volume / renal context
    # ----------------------------
    if any(x in t for x in ["dehydration", "poor intake", "reduced urine", "oliguria"]):
        notes.append(
            "Clinical notes suggest possible volume depletion, which may influence renal function and electrolyte findings."
        )

    # ----------------------------
    # Medication / toxin awareness
    # ----------------------------
    if any(x in t for x in ["diuretics", "ace inhibitor", "arb", "nsaid", "lithium"]):
        notes.append(
            "Medication history in notes may influence renal function or electrolyte balance."
        )

    if not notes:
        return None

    return {
        "notes_coherence": notes
    }

def build_cross_domain_coherence(
    routes: list,
    dominant_driver: dict,
    acid_base: dict | None,
    ecg: dict | None,
    notes: dict | None
) -> dict:
    """
    Cross-domain coherence synthesis.

    Purpose:
    - Reduce the big picture into a short, clinically useful orientation
    - Highlight agreement, tension, or uncertainty between domains
    - NEVER diagnose
    - NEVER override severity or routes
    """

    summary = []
    confidence = "limited"

    # ----------------------------
    # Dominant driver framing
    # ----------------------------
    if dominant_driver and dominant_driver.get("driver"):
        if dominant_driver.get("confidence") == "clear":
            summary.append(
                f"Primary physiological driver appears to be '{dominant_driver['driver']}'."
            )
            confidence = "moderate"
        elif dominant_driver.get("confidence") == "competing":
            summary.append(
                "Multiple competing physiological drivers are present; avoid anchoring on a single cause."
            )
            confidence = "limited"

    # ----------------------------
    # Acid–base reinforcement
    # ----------------------------
    if acid_base:
        summary.append(
            "Acid–base findings provide additional physiological context that supports laboratory interpretation."
        )
        confidence = "moderate"

    # ----------------------------
    # ECG reinforcement
    # ----------------------------
    if ecg:
        summary.append(
            "ECG context highlights potential electrical risk related to electrolyte abnormalities."
        )
        confidence = "moderate"

    # ----------------------------
    # Notes reinforcement
    # ----------------------------
    if notes:
        summary.append(
            "Clinical notes provide contextual information that should be correlated with physiological findings."
        )
        confidence = "moderate"

    if not summary:
        summary.append(
            "Interpretation is limited to available laboratory data; no additional cross-domain context is present."
        )

    return {
        "cross_domain_summary": summary,
        "confidence": confidence
    }





    return deduped


        
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

def extract_abg_from_image(image_bytes: bytes) -> dict:
    """
    Extract arterial blood gas values from scanned ABG reports.
    Returns a flat dict of ABG parameters.
    Never guesses. Silent on failure.
    """
    try:
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        system_prompt = (
            "You are extracting arterial blood gas values from a scanned ABG report. "
            "Return STRICT JSON ONLY in this format:\n"
            "{\n"
            '  "pH": "", "pCO2": "", "pO2": "", "HCO3": "", '
            '"BaseExcess": "", "Lactate": "", "Sodium": "", '
            '"Potassium": "", "Chloride": "", "Glucose": ""\n'
            "}\n"
            "If a value is not visible, use null. Do not infer."
        )

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

        parsed = safe_get_parsed_from_choice(resp.choices[0])
        return parsed if isinstance(parsed, dict) else {}

    except Exception:
        return {}


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
        r = tag_value_source(r, "cbc")
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

def elevate_severity(
    current: str | None,
    minimum: str,
) -> str:
    """
    Elevate severity to at least `minimum`, never downgrade.

    Severity scale (ascending):
        normal → mild → moderate → severe → critical

    If current is None or invalid, minimum is returned.
    """

    SEVERITY_ORDER = [
        "normal",
        "mild",
        "moderate",
        "severe",
        "critical",
    ]

    if not current:
        return minimum

    current = current.lower()
    minimum = minimum.lower()

    if current not in SEVERITY_ORDER:
        return minimum

    if minimum not in SEVERITY_ORDER:
        return current

    if SEVERITY_ORDER.index(current) < SEVERITY_ORDER.index(minimum):
        return minimum

    return current


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
    
def is_authoritative_electrolyte(cdict: dict, analyte: str) -> bool:
    """
    Only serum / chemistry electrolytes may independently drive severity.
    ABG or co-oximetry values may inform context but not dominate.
    """
    entry = cdict.get(analyte)
    if not isinstance(entry, dict):
        return False

    source = entry.get("_source")
    return source in ("chemistry", "serum")

def assess_interpretation_boundaries(cdict: dict, routes: list) -> list:
    boundaries = []
    v = lambda k: clean_number(cdict.get(k, {}).get("value"))

    K = v("Potassium")
    pH = v("pH")
    HCO3 = v("Bicarbonate")
    AG = v("Anion Gap")
    CRP = v("CRP")
    WBC = v("WBC")
    Hb = v("Hb")

    if K is not None and pH is not None and pH < 7.30:
        boundaries.append(
            "Potassium interpretation is constrained by acidemia; transcellular shifts may elevate measured potassium."
        )

    if HCO3 is not None and AG is not None and AG >= 16 and HCO3 <= 20:
        boundaries.append(
            "Low bicarbonate with elevated anion gap reflects metabolic acidosis physiology; electrolyte interpretation should be contextualised."
        )

    if WBC is not None and CRP is not None and WBC >= 12 and CRP < 5:
        boundaries.append(
            "Leukocytosis with low CRP may reflect stress physiology rather than active infection."
        )

    if Hb is not None and pH is not None and pH < 7.30:
        boundaries.append(
            "Haemoglobin interpretation during acute acid–base disturbance may not reflect baseline status."
        )

    return boundaries

def build_primary_physiology_summary(
    routes: list,
    severity: str,
    dominant_driver: dict,
    chemistry_dominant: bool,
    acid_base_coherence: dict | None,
    ecg_coherence: dict | None,
    offset_notes: list
) -> str | None:
    """
    Primary Physiology Orchestrator (read-only).

    Purpose:
    - Compress the full report into 1–3 sentences
    - Highlight dominant physiology
    - Reflect buffering or reinforcement
    - Never diagnose
    - Never mention absent domains
    """

    if not routes:
        return None

    sentences = []

    # -------------------------------------------------
    # Sentence 1 — Dominant physiology
    # -------------------------------------------------
    primary_route = routes[0]
    pattern = primary_route.get("pattern", "").lower()

    if chemistry_dominant:
        sentence = (
            "Dominant physiology is high–anion–gap metabolic acidosis "
            "with associated electrolyte or renal stress"
        )
    elif "electrical" in pattern or "potassium" in pattern:
        sentence = (
            "Dominant physiology relates to increased electrical instability"
        )
    elif dominant_driver and dominant_driver.get("driver"):
        sentence = (
            f"Dominant physiology is driven by {dominant_driver['driver'].lower()}"
        )
    else:
        sentence = "Dominant physiological pattern is present"

    # Buffering language (ONLY if already detected)
    if offset_notes:
        if any("buffer" in n.lower() or "compens" in n.lower() for n in offset_notes):
            sentence += ", partially buffered at this stage"

    sentence += "."
    sentences.append(sentence)

    # -------------------------------------------------
    # Sentence 2 — Cross-domain reinforcement (optional)
    # -------------------------------------------------
    if ecg_coherence:
        sentences.append(
            "ECG findings reinforce the clinical significance of this physiology."
        )
    elif acid_base_coherence:
        sentences.append(
            "Acid–base findings provide important physiological context."
        )

    # -------------------------------------------------
    # Sentence 3 — Low-severity compression fallback
    # -------------------------------------------------
    if severity in ("low", "moderate") and not chemistry_dominant:
        if len(routes) == 1:
            sentences = [
                "Findings suggest mild laboratory abnormalities without a single dominant acute pathology."
            ]

    return " ".join(sentences[:3])





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
    # Default severity (must exist)
    # ---------------------------
    sev = {
        "severity": "low",
        "key_issues": [],
        "urgency_flags": []
    }

    # ---------------------------
    # Canonical dict
    # ---------------------------
    cdict = build_cbc_value_dict(ai_json)
    has_cbc = bool(ai_json.get("cbc"))
    has_chem = bool(ai_json.get("chemistry"))
    has_abg = bool(ai_json.get("abg"))
    has_ecg = bool(ai_json.get("ecg"))
    
    supported_domains = sum([has_cbc, has_chem, has_abg, has_ecg])


    # ---------------------------
    # STEP 1: Data integrity check (read-only)
    # ---------------------------
    data_integrity = assess_data_integrity(cdict)


    # -----------------------------------
    # Merge chemistry rows into canonical dict (SAFE)
    # -----------------------------------
    for r in (ai_json.get("chemistry") or []):
        r = tag_value_source(r, "chemistry")

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
    # ABG DOMINANCE FLAG (PHYSIOLOGY ONLY)
    # =====================================================
    abg = ai_json.get("abg") or {}
    
    pH_abg = clean_number(abg.get("pH"))
    pCO2_abg = clean_number(abg.get("pCO2"))
    lactate_abg = clean_number(abg.get("Lactate"))
    
    abg_dominant = False
    
    # Primary ventilatory failure physiology
    if (
        pH_abg is not None and pCO2_abg is not None
        and pH_abg < 7.30
        and pCO2_abg > 45
    ):
        abg_dominant = True
    
    # Additional metabolic stress signal
    if lactate_abg is not None and lactate_abg >= 3.0:
        abg_dominant = True

    # =====================================================
    # ABG PRIMARY PHYSIOLOGY ROUTE (VENTILATORY RISK)
    # =====================================================
    abg = ai_json.get("abg") or {}
    
    pH_abg = clean_number(abg.get("pH"))
    pCO2_abg = clean_number(abg.get("pCO2"))
    lactate_abg = clean_number(abg.get("Lactate"))
    
    # ---- Primary respiratory acidosis physiology ----
    if (
        pH_abg is not None and pCO2_abg is not None
        and pH_abg < 7.30
        and pCO2_abg > 45
    ):
        add_route(
            routes,
            priority="primary",
            pattern="Respiratory acidosis physiology",
            route=(
                "Low pH with elevated pCO₂ indicates ventilatory failure physiology "
                "with risk of clinical deterioration"
            ),
            next_steps=[
                "Prompt clinical assessment of ventilation and respiratory status",
                "Correlate with respiratory rate, oxygenation, and mental status",
                "Repeat blood gas if clinically indicated"
            ]
        )
    
    # ---- Lactate escalation (metabolic stress signal) ----
    if lactate_abg is not None and lactate_abg >= 3.0:
        add_route(
            routes,
            priority="secondary",
            pattern="Elevated lactate (metabolic stress)",
            route=(
                "Raised lactate indicates systemic metabolic stress and increases overall clinical risk"
            ),
            next_steps=[
                "Assess for hypoxia, hypoperfusion, or increased work of breathing",
                "Repeat lactate to assess trend if clinically indicated"
            ]
        )

    
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
        if (
            (K < 3.0 or K > 6.0)
            and is_authoritative_electrolyte(cdict, "Potassium")
        ):

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
    # ---------------------------
    # STEP 9.3 — Interpretation boundaries (read-only)
    # ---------------------------
    interpretation_boundaries = assess_interpretation_boundaries(cdict, routes)
    # ---------------------------
    # STEP 2: Pattern strength calibration (read-only)
    # ---------------------------
    pattern_strength = assess_pattern_strength(routes, cdict)
    # ---------------------------
    # STEP 3: Interpretation boundaries (read-only)
    # ---------------------------
    interpretation_boundaries = assess_interpretation_boundaries(cdict, routes)
    # ---------------------------
    # STEP 4: Dominant driver (orientation only)
    # ---------------------------
    dominant_driver = derive_dominant_driver(routes)
    # ---------------------------
    # STEP 6: ECG coherence (read-only)
    # ---------------------------
    ecg_data = ai_json.get("ecg") if isinstance(ai_json, dict) else None
    ecg_coherence = assess_ecg_coherence(cdict, ecg_data)
    # ---------------------------
    # STEP 7: Notes coherence (read-only)
    # ---------------------------
    notes_text = ai_json.get("clinical_notes") or ai_json.get("notes")
    notes_coherence = assess_notes_coherence(notes_text, routes)
    acid_base_coherence = assess_acid_base_coherence(cdict)
    # ---------------------------
    # STEP 8: Cross-domain coherence (synthesis only)
    # ---------------------------
    cross_domain_coherence = build_cross_domain_coherence(
        routes=routes,
        dominant_driver=dominant_driver,
        acid_base=acid_base_coherence,
        ecg=ecg_coherence,
        notes=notes_coherence
    )







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
    # ABG coherence (read-only)
    # ---------------------------
    abg_notes = assess_abg_coherence(cdict, ai_json.get("abg"))
    # ---------------------------
    # ECG coherence (read-only)
    # ---------------------------
    ecg_notes = assess_ecg_coherence(cdict, ai_json.get("ecg"))


    # ---------------------------
    # Interpretation boundaries (read-only)
    # ---------------------------
    interpretation_boundaries = assess_interpretation_boundaries(cdict, routes)
    abg_notes = assess_abg_coherence(cdict, ai_json.get("abg"))
    ecg_notes = assess_ecg_coherence(cdict, ai_json.get("ecg"))
    abg_physiology_notes = assess_abg_physiology(ai_json.get("abg"))
    offset_notes = assess_offsetting_contexts(
    cdict,
    ai_json.get("abg"),
    ai_json.get("ecg"),
)

    # ---------------------------
    # STEP 10.4 — Contextual interpretation notes (read-only)
    # ---------------------------
    context_notes = []
    
    for group in (
        interpretation_boundaries,
        abg_notes,
        ecg_notes,
        abg_physiology_notes,
        offset_notes,
    ):

        if isinstance(group, list):
            for note in group:
                if isinstance(note, str):
                    context_notes.append(note)
    
    # De-duplicate while preserving order
    seen = set()
    context_notes = [
        n for n in context_notes
        if not (n in seen or seen.add(n))
    ]


    
    # ---------------------------
    # Explainability floor (non-decisional)
    # ---------------------------
    explainability = build_explainability_floor(
        cdict,
        routes,
        sev,
        interpretation_boundaries
    )
    
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

    diffs = generate_differential_trees(cdict)

    patient_name = None
    try:
        patient_name = (ai_json.get("patient") or {}).get("name")
    except Exception:
        pass

    trends = trend_comparison(patient_name, cdict)

    # =====================================================
    # SEVERITY RESOLUTION (ROUTE-DOMINANT, SAFE ORDER)
    # =====================================================
    
    # 1️⃣ Numeric severity from labs (CBC / chemistry / ABG-safe)
    numeric_sev = evaluate_severity_and_urgency(cdict)
    
    # 2️⃣ Route-based severity (clinical dominance)
    route_sev = severity_from_routes(routes)
    
    # 3️⃣ Escalation order (never downgrade)
    severity_rank = {
        "low": 0,
        "moderate": 1,
        "high": 2,
        "critical": 3
    }
    
    # 4️⃣ Choose highest severity
    final_severity = route_sev
    if severity_rank.get(numeric_sev["severity"], 0) > severity_rank.get(route_sev, 0):
        final_severity = numeric_sev["severity"]

    


    # -------------------------------------------------
    # SAFETY GUARD: LOW must be justified by data
    # -------------------------------------------------
    
    # 5️⃣ Build final severity object
    sev = dict(numeric_sev)
    sev["severity"] = final_severity
    
    # ABG-only data can NEVER be LOW
    if supported_domains == 1 and has_abg:
        sev["severity"] = max(
            sev["severity"],
            "moderate",
            key=lambda x: severity_rank[x]
        )
    
    # 6️⃣ Stability guard (single-domain cap, etc.)
    # Do NOT allow ABG-only cases to be downgraded
    if not (supported_domains == 1 and ai_json.get("abg")):
        sev = assess_severity_stability(cdict, routes, sev)
    
    # 7️⃣ Follow-up block (depends ONLY on final severity)
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
    augmented["_data_integrity"] = data_integrity
    augmented["_pattern_strength"] = pattern_strength
    augmented["_interpretation_boundaries"] = interpretation_boundaries
    augmented["_dominant_driver"] = dominant_driver
    augmented["_ecg_coherence"] = ecg_coherence
    augmented["_notes_coherence"] = notes_coherence
    augmented["_cross_domain_coherence"] = cross_domain_coherence
    augmented["_interpretation_boundaries"] = interpretation_boundaries
    augmented["_explainability"] = explainability
    augmented["_interpretation_context"] = context_notes
    augmented["context_notes"] = context_notes
    # ---------------------------
    # PRIMARY PHYSIOLOGY SUMMARY (TOP-LEVEL ORCHESTRATION)
    # ---------------------------
    primary_physiology_summary = build_primary_physiology_summary(
    routes=routes,
    severity=sev.get("severity"),
    dominant_driver=dominant_driver,
    chemistry_dominant=chemistry_dominant,
    acid_base_coherence=acid_base_coherence,
    ecg_coherence=ecg_coherence,
    offset_notes=offset_notes,
)

    # ---------------------------
    # Primary physiology summary fallback (MUST EXIST)
    # ---------------------------
    if not primary_physiology_summary:
        primary_physiology_summary = (
            "Interpretation is limited to available data without a single dominant acute physiology."
        )
    
    augmented["primary_physiology_summary"] = primary_physiology_summary
    return augmented



    return augmented

def has_supported_domain(ai_json: dict) -> bool:
    """
    At least ONE supported data domain must be present.
    CBC is NOT mandatory.
    """
    return any([
        bool(ai_json.get("cbc")),
        bool(ai_json.get("chemistry")),
        bool(ai_json.get("abg")),
        bool(ai_json.get("ecg")),
    ])




# ---------------------------
# Main report processing
# ---------------------------
def process_report(job: dict) -> dict:
    report_id = job.get("id")
    extracted_abg = {}
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
        
                # ---- ABG OCR (optional, non-blocking) ----
                abg_out = extract_abg_from_image(buf.getvalue())
                if isinstance(abg_out, dict) and any(v is not None for v in abg_out.values()):
                    extracted_abg.update(abg_out)
        
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
        # ---- Merge ABG OCR results (if present) ----
        if extracted_abg:
            ai_json["abg"] = extracted_abg

        
        # --------------------
        # Patient demographics extraction (THIS IS THE KEY)
        # --------------------
        raw_text_for_patient = text if not scanned else ocr_identity_text
        
        patient = extract_patient_demographics(raw_text_for_patient)
        print("🧪 RAW IDENTITY TEXT (first 500 chars):\n", raw_text_for_patient[:500])
        
        print("🧾 Extracted patient demographics:", patient)
        
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
                pass




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
