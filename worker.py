#!/usr/bin/env python3
"""
AMI Worker v4 - Pattern → Route → Next Steps

NOTES:
- This attempts OpenAI Vision via the 'responses' endpoint if available.
  If your OpenAI client doesn't have `client.responses`, set OPENAI_SDK_MODERN=False
  below and the worker will fall back to the chat completions shape used previously.
- For scanned PDFs we convert to images (pdf2image) and then OCR via OpenAI Vision.
- If OCR fails, we return a graceful error and mark the job failed (so you can inspect).
- Route Engine V4 produces: patterns[], routes[], next_steps[] and an overall priority score.
- Make sure your Dockerfile installs poppler-utils (already done) and requirements include:
  openai, pypdf, pdf2image, pillow, supabase, requests
"""

import os
import time
import json
import io
import traceback
import base64
import re
from typing import List, Dict, Any

# Third-party
from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image

# ---------- CONFIG ----------
OPENAI_SDK_MODERN = True
# If your OpenAI client (installed package) doesn't provide client.responses, set above to False.

# Environment & clients
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY not set; OpenAI calls will fail (running in test mode).")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = "reports"

print("AMI Worker v4 starting — Pattern → Route → Next Steps")

# ---------- Helpers ----------
def safe_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract selectable text from PDF with pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            pages.append(p.extract_text() or "")
        return "\n\n".join(pages).strip()
    except Exception as e:
        safe_print("PDF parse error:", e)
        return ""

def is_scanned_pdf(pdf_text: str) -> bool:
    if not pdf_text:
        return True
    if len(pdf_text.strip()) < 40:
        return True
    return False

def clean_number(val) -> float | None:
    """Robust numeric extractor: handles '88.0%', '11,6 g/dL', '<5.0', 'ND' etc."""
    if val is None: return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val)
    s = s.replace(",", ".")
    # remove leading < or > but keep number
    s = re.sub(r"[<>~\s]+", "", s)
    m = re.search(r"-?\d+\.?\d*", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except:
        return None

# ---------- Vision OCR wrapper (tries modern responses API first, fallback to chat completions) ----------
def ocr_image_with_openai(image_bytes: bytes) -> dict:
    """
    Returns a dict: {"cbc": [ {analyte, value, units, reference_low, reference_high} ] }
    If failure, raises Exception.
    """

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    system_prompt = (
        "You are an OCR/data extraction assistant for laboratory PDF scans. "
        "Extract ALL analytes you can see (CBC, differential, platelets, electrolytes, U&E, renal markers, LFTs, CRP, CK etc.). "
        "Return STRICT JSON: { 'cbc': [ { 'analyte': '', 'value': '', 'units': '', 'reference_low': '', 'reference_high': '' } ] } "
        "Return ONLY JSON."
    )

    # Try modern responses API if available
    if OPENAI_SDK_MODERN:
        try:
            resp = client.responses.create(
                model="gpt-4o",   # vision-capable
                input=[
                    {"role":"system", "content": system_prompt},
                    {
                        "role":"user",
                        "content":[
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            }
                        ]
                    }
                ],
                response_format={"type":"json_object"},
                temperature=0.0,
            )
            # The responses API with response_format=json_object returns python object
            # Try to extract it safely:
            out = None
            try:
                # Many SDKs put parsed JSON under resp.output[...] structure
                out = resp.output[0].content[0].json
            except Exception:
                # fallback: resp.data or resp.output_text
                out = getattr(resp, "output_text", None) or getattr(resp, "text", None)
                if isinstance(out, str):
                    out = json.loads(out)
            if not isinstance(out, dict):
                raise ValueError("Unexpected OCR response content (not dict)")
            return out

        except Exception as e:
            safe_print("Vision OCR (responses API) error:", e)
            # let below fallback attempt proceed

    # Fallback: older chat completions style (may fail for images depending on SDK)
    try:
        # Use chat completions but pass the image as data URL object if supported
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system", "content": system_prompt},
                {"role":"user", "content": [
                    {"type":"image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                ]}
            ],
            response_format={"type":"json_object"},
            temperature=0.0,
        )
        # Some SDKs place parsed JSON at resp.choices[0].message.content or .parsed
        try:
            parsed = resp.choices[0].message.content
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
        except Exception:
            parsed = getattr(resp.choices[0].message, "parsed", None) or getattr(resp.choices[0].message, "content", None)
        if isinstance(parsed, str):
            parsed = json.loads(parsed)
        if not isinstance(parsed, dict):
            raise ValueError("Unexpected OCR chat response")
        return parsed

    except Exception as e:
        safe_print("Vision OCR (chat fallback) error:", e)
        raise

# ---------- Interpretation model (text) ----------
def call_ai_on_report(text: str) -> dict:
    """Ask the model to produce a structured interpretation JSON (patient, cbc list, summary)."""
    system_prompt = (
        "You are a clinical assistant analysing lab results. Output STRICT JSON with fields:\n"
        "{ 'patient': { 'name': null, 'age': null, 'sex': 'Unknown' },\n"
        "  'cbc': [ { 'analyte':'', 'value':'', 'units':'', 'reference_low':'', 'reference_high':'', 'flag':'' } ],\n"
        "  'summary': { 'impression':'', 'suggested_follow_up':'' }\n"
        "}\n"
        "Do not invent numbers. If you cannot find a value, omit it. Return only JSON."
    )
    try:
        # Try modern responses API first
        if OPENAI_SDK_MODERN:
            resp = client.responses.create(
                model="gpt-4o-mini",
                input=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": text}
                ],
                response_format={"type":"json_object"},
                temperature=0.0,
            )
            try:
                ai_json = resp.output[0].content[0].json
            except Exception:
                ai_json = json.loads(getattr(resp, "output_text", "{}"))
            if not isinstance(ai_json, dict):
                raise ValueError("Interpretation is not a dict")
            return ai_json
        else:
            # Chat completions fallback
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system", "content": system_prompt},
                    {"role":"user", "content": text}
                ],
                response_format={"type":"json_object"},
                temperature=0.0
            )
            parsed = resp.choices[0].message.content
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            return parsed
    except Exception as e:
        safe_print("Interpretation model error:", e)
        raise

# ---------- Build canonical cbc dict ----------
def build_canonical_dict(ai_json: dict) -> Dict[str, dict]:
    """
    Turn ai_json["cbc"] (list) into canonical names for route engine.
    Returns dict keyed by canonical short names.
    """
    out = {}
    rows = ai_json.get("cbc") or []
    for r in rows:
        if not isinstance(r, dict): continue
        name = (r.get("analyte") or r.get("test") or "").lower()
        if not name: continue
        def put(key):
            if key not in out:
                out[key] = r

        if "haemoglobin" in name or "hemoglobin" in name or name == "hb":
            put("Hb")
        elif "mcv" in name:
            put("MCV")
        elif "mch" in name:
            put("MCH")
        elif "mchc" in name:
            put("MCHC")
        elif "rbc" in name or "red cell" in name:
            put("RBC")
        elif "white" in name or "wbc" in name or "leucocyte" in name or "leukocyte" in name:
            put("WBC")
        elif "neutrophil" in name:
            put("Neutrophils")
        elif "lymphocyte" in name or "lymph" in name:
            put("Lymphocytes")
        elif "monocyte" in name:
            put("Monocytes")
        elif "platelet" in name or "plt" in name:
            put("Platelets")
        elif "creatinine" in name:
            put("Creatinine")
        elif "urea" in name:
            put("Urea")
        elif "crp" in name:
            put("CRP")
        elif "sodium" in name or name == "na":
            put("Sodium")
        elif "potassium" in name or name == "k":
            put("Potassium")
        elif "bicarbonate" in name or "co2" in name:
            put("Bicarbonate")
        elif "alt" in name:
            put("ALT")
        elif "ast" in name:
            put("AST")
        elif "alp" in name:
            put("ALP")
        elif "bilirubin" in name:
            put("Bilirubin")
        elif "ck-mb" in name or "ck mb" in name:
            put("CK-MB")
        elif "ck" in name:
            put("CK")
        # else: ignore for now
    return out

# ---------- Route Engine V4 (Patterns -> Routes -> Next steps) ----------
def generate_routes_v4(canonical: Dict[str, dict]) -> Dict[str, Any]:
    """
    Returns structure:
    {
      "patterns": [ { "pattern":"...", "evidence":[...], "priority":int } ],
      "routes": [ "..." ],
      "next_steps": [ "..." ],
      "priority_score": int  # 0-100
    }
    """
    def v(k):
        return clean_number(canonical.get(k, {}).get("value")) if canonical.get(k) else None

    Hb = v("Hb")
    MCV = v("MCV")
    MCH = v("MCH")
    WBC = v("WBC")
    Neut = v("Neutrophils")
    Lymph = v("Lymphocytes")
    Plt = v("Platelets")
    Cr = v("Creatinine")
    Urea = v("Urea")
    CRP = v("CRP")
    Na = v("Sodium")
    K = v("Potassium")
    Bic = v("Bicarbonate")

    patterns = []
    routes = []
    next_steps = []
    score = 0

    # --- Anemia patterns
    if Hb is not None and Hb < 12.5:
        evidence = []
        if Hb is not None: evidence.append(f"Hb {Hb}")
        if MCV is not None: evidence.append(f"MCV {MCV}")
        if MCH is not None: evidence.append(f"MCH {MCH}")

        if MCV is not None and MCV < 80:
            patterns.append({"pattern":"Microcytic anaemia", "evidence": evidence, "priority": 75})
            routes.append("Microcytic anaemia → iron deficiency most likely; consider blood loss.")
            next_steps.append("Order ferritin, iron studies (serum iron, TIBC), and reticulocyte count.")
            score += 20
        elif MCV is not None and MCV > 100:
            patterns.append({"pattern":"Macrocytic anaemia", "evidence": evidence, "priority": 75})
            routes.append("Macrocytic anaemia → consider B12/folate deficiency, liver disease, meds.")
            next_steps.append("Order B12, folate, liver enzymes; review medications (e.g., methotrexate).")
            score += 20
        else:
            patterns.append({"pattern":"Normocytic anaemia", "evidence": evidence, "priority": 60})
            routes.append("Normocytic anaemia → consider anaemia of chronic disease, early iron deficiency or renal disease.")
            next_steps.append("Assess renal function, review chronic disease history; consider ferritin and reticulocyte count.")
            score += 15

    # --- Infection / inflammation patterns
    if (WBC is not None and WBC > 11) or (CRP is not None and CRP > 10):
        evidence = []
        if WBC is not None: evidence.append(f"WBC {WBC}")
        if Neut is not None: evidence.append(f"Neutrophils {Neut}")
        if CRP is not None: evidence.append(f"CRP {CRP}")

        patterns.append({"pattern":"Inflammation / Infection", "evidence": evidence, "priority": 80})
        if Neut is not None and Neut > 65:
            routes.append("Neutrophil-predominant leukocytosis → bacterial infection more likely.")
            next_steps.append("Assess for likely focal source (chest, urine, abdomen). Consider starting empirical antibiotics if clinically indicated.")
            score += 25
        elif Lymph is not None and Lymph > 45:
            routes.append("Lymphocytosis → viral or post-viral pattern.")
            next_steps.append("Consider viral testing based on presentation; supportive treatment unless bacterial features present.")
            score += 15
        else:
            routes.append("Inflammatory response pattern (CRP/WBC) → correlate clinically.")
            next_steps.append("Repeat CBC + CRP in 24–72 hours to trend; consider cultures if febrile.")
            score += 10

    # --- Platelets
    if Plt is not None:
        if Plt < 150:
            patterns.append({"pattern":"Thrombocytopenia", "evidence":[f"Platelets {Plt}"], "priority": 70})
            routes.append("Thrombocytopenia → assess bleeding risk, drugs, marrow causes.")
            next_steps.append("Check peripheral smear, repeat platelet count; review medications and bleeding history.")
            score += 10
        elif Plt > 450:
            patterns.append({"pattern":"Thrombocytosis", "evidence":[f"Platelets {Plt}"], "priority": 60})
            routes.append("Thrombocytosis → likely reactive (infection/inflammation/iron deficiency) vs myeloproliferative.")
            next_steps.append("Correlate with inflammatory markers; if persistent, consider referral for hematology evaluation.")
            score += 8

    # --- Renal / electrolytes / acid-base
    if Cr is not None and Cr > 120:
        patterns.append({"pattern":"Impaired renal function", "evidence":[f"Creatinine {Cr}"], "priority": 70})
        routes.append("Possible acute or chronic kidney impairment.")
        next_steps.append("Repeat U&E, assess hydration and medication nephrotoxins; calculate eGFR if possible.")
        score += 12

    if K is not None and (K < 3.3 or K > 5.5):
        patterns.append({"pattern":"Potassium abnormality", "evidence":[f"K {K}"], "priority": 90})
        routes.append("Potassium derangement → arrhythmia risk.")
        next_steps.append("Check ECG if symptomatic; correct potassium per local protocols.")
        score += 20

    if Bic is not None and Bic < 21:
        patterns.append({"pattern":"Low bicarbonate (metabolic acidosis)", "evidence":[f"Bicarbonate {Bic}"], "priority": 60})
        routes.append("Metabolic acidosis — consider sepsis, renal impairment, ketoacidosis, diarrhoea.")
        next_steps.append("Correlate clinically; consider blood gas if severe or symptomatic.")
        score += 8

    # --- Muscle injury
    CK = clean_number(canonical.get("CK", {}).get("value")) if canonical.get("CK") else None
    if CK is not None and CK > 300:
        patterns.append({"pattern":"Raised CK", "evidence":[f"CK {CK}"], "priority": 60})
        routes.append("Muscle injury / rhabdomyolysis risk.")
        next_steps.append("Check renal function and urine myoglobin; hydrate; consider urgent nephrology if CK very high.")
        score += 8

    # If nothing strongly abnormal:
    if not patterns:
        patterns.append({"pattern":"No high-priority abnormalities", "evidence":[], "priority":10})
        routes.append("No major route-level abnormalities found from available results. Correlate with clinical picture and prior labs.")
        next_steps.append("If symptoms persist or high clinical suspicion, consider additional tests or repeat labs in 24–72 hours.")
        score += 1

    # Normalize priority_score 0-100
    priority_score = min(100, max(0, score))

    return {
        "patterns": patterns,
        "routes": routes,
        "next_steps": next_steps,
        "priority_score": priority_score
    }

# ---------- Main report processing ----------
def process_report(job: dict) -> dict:
    report_id = job["id"]
    file_path = job.get("file_path")
    l_text = job.get("l_text") or ""
    safe_print(f"Processing report {report_id} (file_path={file_path})")

    if not file_path:
        err = "Missing file_path"
        safe_print(err)
        supabase.table("reports").update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
        return {"error": err}

    pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_bytes, "data"):
        pdf_bytes = pdf_bytes.data

    # 1) try text extraction
    text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(text)
    safe_print(f"scanned={scanned}, text_length={len(text)}")

    try:
        if scanned:
            safe_print("SCANNED PDF -> converting pages to images...")
            images = convert_from_bytes(pdf_bytes)
            combined_rows = []
            # For each page, OCR with OpenAI Vision
            for i, img in enumerate(images, start=1):
                safe_print(f"OCR page {i}/{len(images)}")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                page_bytes = buf.getvalue()
                try:
                    ocr_result = ocr_image_with_openai(page_bytes)
                    if isinstance(ocr_result, dict) and "cbc" in ocr_result:
                        # extend rows
                        for r in ocr_result["cbc"]:
                            combined_rows.append(r)
                        safe_print(f" -> page {i} extracted {len(ocr_result.get('cbc', []))} rows")
                except Exception as e:
                    safe_print(f" -> OCR page {i} error:", e)
            if not combined_rows:
                raise ValueError("No CBC extracted from scanned PDF via OCR.")
            merged_text = json.dumps({"cbc": combined_rows}, ensure_ascii=False)
        else:
            merged_text = text or l_text
            if not merged_text.strip():
                raise ValueError("No usable content (empty digital PDF).")

        # 2) interpretation model -> structured ai_json
        ai_json = call_ai_on_report(merged_text)
        safe_print("Interpretation model returned JSON keys:", list(ai_json.keys()))

        # 3) build canonical dict + Route Engine V4
        canonical = build_canonical_dict(ai_json)
        safe_print("Canonical keys:", list(canonical.keys()))
        routes_struct = generate_routes_v4(canonical)

        # merge into ai_json for storage
        ai_json["routes_v4"] = routes_struct

        # 4) update supabase record
        supabase.table("reports").update({
            "ai_status":"completed",
            "ai_results": ai_json,
            "ai_error": None
        }).eq("id", report_id).execute()

        safe_print(f"✅ Report {report_id} processed successfully")
        return {"success": True, "data": ai_json}

    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        safe_print("❌ Error processing report:", err)
        traceback.print_exc()
        supabase.table("reports").update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
        return {"error": err}

# ---------- Worker loop ----------
def main():
    safe_print("Entering main loop...")
    while True:
        try:
            res = supabase.table("reports").select("*").eq("ai_status","pending").limit(1).execute()
            jobs = getattr(res, "data", None) or res.get("data") or []
            if not jobs:
                safe_print("No jobs...")
                time.sleep(1)
                continue

            job = jobs[0]
            safe_print("🔎 Found job:", job["id"])
            supabase.table("reports").update({"ai_status":"processing"}).eq("id", job["id"]).execute()
            process_report(job)

        except Exception as e:
            safe_print("LOOP ERROR:", e)
            traceback.print_exc()
            time.sleep(3)

if __name__ == "__main__":
    main()
