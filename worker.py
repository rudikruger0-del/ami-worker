# worker.py
# AMI Worker — Digital & Scanned PDF handling + Dr Riekert Route Engine inserted
# Minimal external behaviour changed from your original: scanned detection, Vision OCR, AI interpretation.
# Added: parsing, canonical mapping, route engine, severity/urgency, ddx, next_steps, trend analysis.

import os
import time
import json
import io
import traceback
import base64
from typing import Dict, Any, List, Optional

from supabase import create_client, Client
from openai import OpenAI
from pypdf import PdfReader
from pdf2image import convert_from_bytes   # convert scanned PDFs → images
from PIL import Image
import re

# -------- ENV & CLIENTS --------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
  raise RuntimeError("SUPABASE_URL or SUPABASE_SERVICE_KEY is missing")

if not OPENAI_API_KEY:
  print("⚠️  OPENAI_API_KEY is not set – OpenAI client will fail.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

BUCKET = os.getenv("SUPABASE_BUCKET", "reports")

# ---------- small helpers ----------

def safe_float(v) -> Optional[float]:
  try:
    return float(v)
  except Exception:
    return None

def first_or_none(arr):
  return arr[0] if arr else None

# ---------- PDF text extraction ----------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
  """Extract text from selectable text PDFs (pypdf)."""
  try:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
      txt = page.extract_text() or ""
      pages.append(txt)
    full = "\n\n".join(pages).strip()
    return full
  except Exception as e:
    print("PDF parse error:", e)
    return ""

def is_scanned_pdf(pdf_text: str) -> bool:
  """
  Heuristic: if extracted text is very short, treat as scanned.
  """
  if not pdf_text:
    return True
  if len(pdf_text.strip()) < 40:
    return True
  return False

# ---------- Vision OCR for scanned PDFs ----------

def extract_cbc_from_image(image_bytes: bytes) -> dict:
  """
  Send a single image (bytes) to OpenAI Vision (gpt-4o) and request strict CBC JSON.
  Returns parsed dict or {'cbc': []} on failure.
  """
  base64_image = base64.b64encode(image_bytes).decode("utf-8")

  system_prompt = (
    "You are an OCR and data extraction assistant for medical laboratory PDFs. "
    "Extract all CBC and chemistry analytes (CBC, WBC differential, CRP, creatinine, electrolytes, liver enzymes, CK, etc.). "
    "Return STRICT JSON only, with structure:\n"
    "{\n"
    "  \"cbc\": [\n"
    "    {\"analyte\": \"Hb\", \"value\": 11.6, \"units\": \"g/dL\", \"reference_low\": 12.4, \"reference_high\": 16.7}\n"
    "  ]\n"
    "}\n"
    "If you cannot identify numeric values, return an empty list for cbc."
  )

  try:
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "system", "content": system_prompt},
        {
          "role": "user",
          "content": [
            {"type": "text", "text": "Extract the lab values from this image and return the JSON described."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}" }}
          ]
        }
      ],
      response_format={"type": "json_object"},
      temperature=0.0,
    )

    raw = response.choices[0].message.content
    # raw could be dict or JSON string
    if isinstance(raw, dict):
      return raw
    if isinstance(raw, str):
      try:
        return json.loads(raw)
      except Exception:
        # fallback: try to extract JSON substring
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
          try:
            return json.loads(m.group(0))
          except Exception:
            pass
    return {"cbc": []}
  except Exception as e:
    print("Vision OCR call failed:", e)
    return {"cbc": []}

# ---------- Parsing - text & OCR JSON merging ----------

# Simple synonyms map to canonical keys
SYNONYMS = {
  "hb": "Hb","haemoglobin":"Hb","hemoglobin":"Hb",
  "rbc":"RBC","erythrocyte":"RBC",
  "hct":"HCT","haematocrit":"HCT","hematocrit":"HCT",
  "mcv":"MCV","mean corpuscular volume":"MCV",
  "mch":"MCH","mean corpuscular hemoglobin":"MCH",
  "mchc":"MCHC",
  "rdw":"RDW",
  "wbc":"WBC","white cell count":"WBC","leukocyte":"WBC","leucocyte":"WBC",
  "neutrophils":"Neutrophils","neutrophil":"Neutrophils",
  "lymphocytes":"Lymphocytes","lymphocyte":"Lymphocytes",
  "monocytes":"Monocytes","eosinophils":"Eosinophils","basophils":"Basophils",
  "platelets":"Platelets","thrombocytes":"Platelets",
  "crp":"CRP","c-reactive protein":"CRP",
  "creatinine":"Creatinine",
  "sodium":"Sodium","na":"Sodium",
  "potassium":"Potassium","k":"Potassium",
  "chloride":"Chloride","cl":"Chloride",
  "urea":"Urea",
  "alt":"ALT","ast":"AST","ck":"CK","creatine kinase":"CK"
}

def normalize_label(label: str) -> Optional[str]:
  if not label:
    return None
  l = re.sub(r'[^a-z0-9 ]', '', label.lower()).strip()
  if l in SYNONYMS:
    return SYNONYMS[l]
  # partial match
  for syn,label_key in SYNONYMS.items():
    if syn in l:
      return label_key
  return None

def parse_values_from_text(text: str) -> Dict[str, Dict[str, Any]]:
  """
  Basic regex-based extraction from digital text.
  Returns mapping canonical -> {'value': float, 'units': str, 'raw': line}
  """
  results: Dict[str, Dict[str, Any]] = {}
  if not text:
    return results

  lines = [ln.strip() for ln in re.split(r'\r|\n', text) if ln.strip()]

  # common numeric patterns: label ... number (unit?) (ref ...)
  for line in lines:
    low = line.lower()
    # try find tokens that look like label
    # e.g. 'Haemoglobin 11.6 g/dL (ref: 12.4-16.7)'
    gen = re.findall(r'([A-Za-z\-\s]{1,30})[:\s]{1,3}(-?\d+\.\d+|-?\d+)(?:\s*([a-zA-Z/%\^\-0-9\(\)\/]*))?', line)
    if gen:
      for g in gen:
        label_raw = g[0].strip()
        val = safe_float(g[1])
        unit = g[2].strip() if g[2] else None
        canon = normalize_label(label_raw)
        if canon and val is not None:
          results.setdefault(canon, {})['value'] = val
          if unit:
            results[canon]['units'] = unit
          results[canon]['raw'] = line
    # percentage-only forms: 'Neutrophils: 88%'
    pct = re.findall(r'([A-Za-z\-\s]{2,30})[:\s]{1,3}(\d{1,3}\.?\d*)\s*%', line)
    if pct:
      for p in pct:
        label_raw = p[0].strip()
        val = safe_float(p[1])
        canon = normalize_label(label_raw)
        if canon and val is not None:
          # percentages for differentials often used as % not absolute
          results.setdefault(canon, {})['value'] = val
          results[canon]['units'] = '%'
          results[canon]['raw'] = line

  # lightweight fallback: search for common labels in whole text
  fallback = re.findall(r'\b(hb|haemoglobin|wbc|crp|creatinine|platelets|mcv|mch|mcvc)\b[^\d]{0,10}(-?\d+\.?\d+|-?\d+)', text, re.I)
  for f in fallback:
    label_raw, val_text = f
    val = safe_float(val_text)
    canon = normalize_label(label_raw)
    if canon and val is not None:
      results.setdefault(canon, {})['value'] = val
      results[canon]['raw'] = label_raw

  return results

def parse_values_from_ocr_json(ocr_json: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
  """
  Expected OCR JSON: {"cbc": [{"analyte":"Hb","value":11.6,"units":"g/dL","reference_low":12.4,"reference_high":16.7}, ...]}
  Normalize analyte names to canonical keys.
  """
  results: Dict[str, Dict[str, Any]] = {}
  if not ocr_json:
    return results
  items = ocr_json.get("cbc") or []
  if not isinstance(items, list):
    return results
  for item in items:
    if not isinstance(item, dict):
      continue
    raw_label = item.get("analyte") or item.get("name") or ""
    canon = normalize_label(raw_label)
    if not canon:
      # try direct capitalized mapping as last resort
      canon = raw_label.strip()
    val = safe_float(item.get("value"))
    if val is not None:
      results.setdefault(canon, {})['value'] = val
    units = item.get("units") or item.get("unit")
    if units:
      results.setdefault(canon, {})['units'] = units
    # attempt extract reference low/high if present
    rl = item.get("reference_low")
    rh = item.get("reference_high")
    if rl is not None or rh is not None:
      try:
        results[canon]['reference_low'] = safe_float(rl)
        results[canon]['reference_high'] = safe_float(rh)
      except:
        pass
    # store raw representation
    results[canon]['raw'] = item
  return results

# ---------- Canonical mapping + derived values ----------

def canonical_map(parsed: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
  """
  Normalize parsed dict to canonical set, compute NLR if able.
  """
  out: Dict[str, Dict[str, Any]] = {}
  for k, v in parsed.items():
    out[k] = {
      "value": safe_float(v.get("value")),
      "units": v.get("units"),
      "raw": v.get("raw"),
      "reference_low": v.get("reference_low"),
      "reference_high": v.get("reference_high")
    }
  # compute NLR if Neutrophils & Lymphocytes present and numeric
  try:
    n = out.get("Neutrophils", {}).get("value")
    l = out.get("Lymphocytes", {}).get("value")
    if n is not None and l is not None and l != 0:
      out["NLR"] = {"value": round(n / l, 2), "units": None, "raw": "computed NLR"}
  except Exception:
    pass
  return out

# ---------- Route Engine (Dr Riekert logic) ----------

COLOR_MAP = {
  5: {"label": "critical", "color": "#b91c1c", "tw":"bg-red-600", "urgency":"high"},
  4: {"label": "severe", "color": "#f97316", "tw":"bg-orange-500", "urgency":"high"},
  3: {"label": "moderate", "color": "#f59e0b", "tw":"bg-yellow-400", "urgency":"medium"},
  2: {"label": "borderline", "color": "#facc15", "tw":"bg-yellow-300", "urgency":"low"},
  1: {"label": "normal", "color": "#10b981", "tw":"bg-green-500", "urgency":"low"}
}

def age_group(age: Optional[float]) -> str:
  if age is None:
    return "adult"
  try:
    a = float(age)
  except:
    return "adult"
  if a < (1/12):
    return "neonate"
  if a < 1:
    return "infant"
  if a < 13:
    return "child"
  if a < 18:
    return "teen"
  if a < 65:
    return "adult"
  return "elderly"

def score_severity(key: str, val: Optional[float], ag: str, sex: str) -> int:
  if val is None:
    return 1
  keyu = key.lower()
  try:
    if keyu == "hb":
      low = 12.0 if sex and str(sex).lower()=="female" else 13.0
      if ag in ("neonate","infant"):
        low = 14.0
      if val < low - 4:
        return 5
      if val < low - 2:
        return 4
      if val < low:
        return 3
      return 1
    if keyu == "wbc":
      if val > 25: return 5
      if val > 15: return 4
      if val > 11: return 3
      return 1
    if keyu == "crp":
      if val > 200: return 5
      if val > 100: return 4
      if val > 50: return 3
      if val > 10: return 2
      return 1
    if keyu == "platelets":
      if val < 20: return 5
      if val < 50: return 4
      if val < 100: return 3
      return 1
    if keyu == "creatinine":
      if val > 354: return 5
      if val > 200: return 4
      if val > 120: return 3
      return 1
    if keyu == "nlr":
      if val > 10: return 4
      if val > 5: return 3
      return 1
    if keyu == "ck":
      if val > 10000: return 5
      if val > 5000: return 4
      if val > 2000: return 3
      return 1
  except:
    return 1
  return 1

def route_engine(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
  ag = age_group(patient_meta.get("age"))
  sex = patient_meta.get("sex") or "unknown"

  patterns: List[Dict[str, str]] = []
  routes: List[str] = []
  next_steps: List[str] = []
  ddx: List[str] = []
  per_key = {}
  severity_scores = []

  # per-analyte scoring and per_key info
  for key, info in canonical.items():
    v = info.get("value")
    score = score_severity(key, v, ag, sex)
    severity_scores.append(score)
    cmap = COLOR_MAP.get(score, COLOR_MAP[1])
    per_key[key] = {
      "value": v,
      "units": info.get("units"),
      "severity": score,
      "urgency": cmap["urgency"],
      "color": cmap["color"],
      "tw": cmap["tw"],
      "raw": info.get("raw")
    }

  # Helper to add pattern
  def add_pat(name, reason, score):
    patterns.append({"pattern": name, "reason": reason})
    # ensure severity considered
    severity_scores.append(score)

  # Extract common values
  hb = canonical.get("Hb", {}).get("value")
  mcv = canonical.get("MCV", {}).get("value")
  mch = canonical.get("MCH", {}).get("value")
  wbc = canonical.get("WBC", {}).get("value")
  neut = canonical.get("Neutrophils", {}).get("value")
  lymph = canonical.get("Lymphocytes", {}).get("value")
  nlr = canonical.get("NLR", {}).get("value")
  crp = canonical.get("CRP", {}).get("value")
  plate = canonical.get("Platelets", {}).get("value")
  creat = canonical.get("Creatinine", {}).get("value")
  ck = canonical.get("CK", {}).get("value")

  # Microcytic anaemia
  if hb is not None and hb < (12.0 if sex.lower()=="female" else 13.0):
    add_pat("anemia", f"Hb {hb}", score_severity("Hb", hb, ag, sex))
    if mcv is not None and mcv < 80:
      add_pat("microcytic anemia", f"MCV {mcv}", max(3, score_severity("Hb", hb, ag, sex)))
      routes.append("Iron Deficiency Route")
      ddx += ["Iron deficiency anemia", "Thalassaemia trait", "Chronic blood loss"]
      if ag == "teen" and sex.lower()=="female":
        next_steps.append("Assess menstrual blood loss; order ferritin and reticulocyte count urgently")
      else:
        next_steps.append("Order ferritin, reticulocyte count; consider stool occult blood if adult")

  # Macrocytic
  if hb is not None and mcv is not None and mcv > 100:
    add_pat("macrocytic anemia", f"MCV {mcv}", max(3, score_severity("Hb", hb, ag, sex)))
    routes.append("Macrocytic Route")
    ddx += ["Vitamin B12 deficiency", "Folate deficiency", "Alcohol related", "Myelodysplasia"]
    next_steps.append("Order B12, folate, peripheral smear; review medications and alcohol history")

  # Normocytic anaemia
  if hb is not None and hb < (12.0 if sex.lower()=="female" else 13.0) and (mcv is None or (80 <= (mcv or 80) <= 100)):
    add_pat("normocytic anemia", "MCV normal or missing", max(2, score_severity("Hb", hb, ag, sex)))
    routes.append("Normocytic Route")
    ddx += ["Acute blood loss", "Haemolysis", "Anaemia of chronic disease"]
    next_steps.append("Order reticulocyte count, LDH, peripheral smear")

  # Leukocytosis & neutrophilic shift
  if wbc is not None and wbc > 11:
    add_pat("leukocytosis", f"WBC {wbc}", score_severity("WBC", wbc, ag, sex))
    if neut is not None and neut > 70:
      add_pat("neutrophilic predominance", f"Neutrophils {neut}%", max(3, score_severity("WBC", wbc, ag, sex)))
      routes.append("Bacterial Infection Route")
      ddx += ["Bacterial infection", "Sepsis", "Acute localized infection"]
      next_steps.append("Assess for source clinically; consider blood cultures and empiric antibiotics if unstable")

  # CRP inflammation
  if crp is not None and crp > 10:
    add_pat("elevated CRP", f"CRP {crp}", score_severity("CRP", crp, ag, sex))
    if crp > 50:
      routes.append("Severe inflammatory response")
      ddx += ["Bacterial infection", "Severe inflammatory disease"]
      next_steps.append("Consider urgent review, blood cultures if febrile, imaging if indicated")

  # High NLR
  if nlr is not None and nlr > 5:
    add_pat("high NLR", f"NLR {nlr}", 4 if nlr > 10 else 3)
    routes.append("High NLR route")
    next_steps.append("Consider sepsis pathway if clinically unwell")

  # Platelets
  if plate is not None:
    if plate < 150:
      add_pat("thrombocytopenia", f"Platelets {plate}", score_severity("Platelets", plate, ag, sex))
      routes.append("Thrombocytopenia route")
      ddx += ["Immune thrombocytopenia", "DIC", "Bone marrow disorder", "Viral infection"]
      next_steps.append("Check peripheral smear; repeat platelet count; urgent review if bleeding or platelets <50")
    elif plate > 450:
      add_pat("thrombocytosis", f"Platelets {plate}", 2)
      routes.append("Thrombocytosis route")
      next_steps.append("Repeat count; consider reactive causes (inflammation); refer to haematology if persistent")

  # Creatinine / AKI
  if creat is not None:
    if creat > 120:
      add_pat("renal stress", f"Creatinine {creat}", score_severity("Creatinine", creat, ag, sex))
      routes.append("AKI route")
      ddx += ["Acute kidney injury", "Chronic kidney disease"]
      next_steps.append("Repeat creatinine and electrolytes urgently; assess urine output and nephrotoxins")

  # CK / Rhabdomyolysis
  if ck is not None and ck > 2000:
    add_pat("possible rhabdomyolysis physiology", f"CK {ck}", score_severity("CK", ck, ag, sex))
    routes.append("Rhabdomyolysis route")
    ddx += ["Rhabdomyolysis", "Severe muscle injury"]
    next_steps.append("Aggressive IV fluids if clinically indicated; monitor creatinine and electrolytes")

  # Neutropenia route
  neut_abs = canonical.get("Neutrophils", {}).get("value")
  if neut_abs is not None and neut_abs < 1.0:
    add_pat("neutropenia", f"Neutrophils {neut_abs}", 4 if neut_abs < 0.5 else 3)
    routes.append("Neutropenia route")
    ddx += ["Drug-induced neutropenia", "Bone marrow failure", "Severe sepsis"]
    next_steps.append("Urgent clinical review; consider isolation and urgent haematology input if severe")

  # combine ddx dedupe
  ddx = list(dict.fromkeys(ddx))

  overall_severity = max(severity_scores) if severity_scores else 1
  color_entry = COLOR_MAP.get(overall_severity, COLOR_MAP[1])
  urgency = color_entry["urgency"]

  # Build short doctor-friendly summary
  summary_lines = []
  if patterns:
    summary_lines.append("Patterns: " + "; ".join([p["pattern"] for p in patterns]))
  if routes:
    summary_lines.append("Primary routes: " + "; ".join(routes))
  if ddx:
    summary_lines.append("Top differentials: " + ", ".join(ddx[:6]))
  if next_steps:
    summary_lines.append("Immediate suggestions: " + " | ".join(next_steps[:6]))

  age_note = ""
  if ag == "teen" and sex and str(sex).lower()=="female":
    age_note = "Teenage female — consider menstrual blood loss and iron deficiency."

  result = {
    "patterns": patterns,
    "routes": routes,
    "next_steps": next_steps,
    "differential": ddx,
    "per_key": per_key,
    "overall_severity": overall_severity,
    "urgency": urgency,
    "color": color_entry["color"],
    "tw_class": color_entry["tw"],
    "age_group": ag,
    "age_note": age_note,
    "summary": "\n".join(summary_lines) if summary_lines else "No significant abnormalities detected."
  }
  return result

# ---------- Trend analysis (simple) ----------

def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if not previous:
    return {"trend":"no_previous"}
  diffs = {}
  for k, v in current.items():
    prev_val = previous.get(k, {}).get("value") if previous else None
    cur_val = v.get("value")
    if prev_val is None or cur_val is None:
      continue
    try:
      delta = cur_val - prev_val
      pct = (delta / prev_val) * 100 if prev_val != 0 else None
      diffs[k] = {"previous": prev_val, "current": cur_val, "delta": delta, "pct_change": pct}
    except:
      pass
  return {"trend": diffs}

# ---------- AI interpretation wrapper (uses your existing interpreter) ----------

def call_ai_on_report(text: str) -> dict:
  """
  Uses your existing GPT interpreter (gpt-4o-mini).
  Expects either raw text (digital) or JSON text (from OCR) and returns strict JSON.
  """
  MAX_CHARS = 12000
  if len(text) > MAX_CHARS:
    text = text[:MAX_CHARS]

  system_prompt = (
    "You are an assistive clinical tool analysing CBC and chemistry results. "
    "You MUST NOT give a formal diagnosis or prescribe treatment. "
    "Only describe laboratory abnormalities, patterns, and general follow-up recommendations.\n\n"
    "Return STRICT JSON with this structure:\n"
    "{\n"
    "  \"patient\": {\"name\": null, \"age\": null, \"sex\": \"Unknown\"},\n"
    "  \"cbc\": [ {\"analyte\": \"\", \"value\": 0, \"units\": \"\", \"reference_low\": null, \"reference_high\": null, \"flag\": \"\" } ],\n"
    "  \"summary\": {\"impression\": \"\", \"suggested_follow_up\": \"\"}\n"
    "}\n"
    "Return ONLY this JSON, no extra text."
  )

  try:
    response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":text},
      ],
      response_format={"type":"json_object"},
      temperature=0.1,
    )
    raw = response.choices[0].message.content
    if isinstance(raw, dict):
      return raw
    if isinstance(raw, str):
      try:
        return json.loads(raw)
      except Exception:
        # attempt substring JSON
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
          try:
            return json.loads(m.group(0))
          except:
            pass
    return {}
  except Exception as e:
    print("AI interpret error:", e)
    return {}

# ---------- Process a single report (main) ----------

def process_report(job: dict) -> dict:
  report_id = job.get("id")
  file_path = job.get("file_path")
  l_text = job.get("l_text") or ""
  patient_age = job.get("age")
  patient_sex = job.get("sex")

  try:
    if not file_path:
      err = f"Missing file_path for report {report_id}"
      print("⚠️", err)
      supabase.table("reports").update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
      return {"error": err}

    # Download PDF bytes
    pdf_bytes = supabase.storage.from_(BUCKET).download(file_path)
    if hasattr(pdf_bytes, "data"):
      pdf_bytes = pdf_bytes.data

    # Detect scanned vs digital
    pdf_text = extract_text_from_pdf(pdf_bytes)
    scanned = is_scanned_pdf(pdf_text)

    merged_text_for_ai = ""
    parsed = {}

    if scanned:
      print(f"📄 Report {report_id} detected as SCANNED — using Vision OCR")
      images = convert_from_bytes(pdf_bytes)
      combined_items = []
      for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()
        try:
          ocr_json = extract_cbc_from_image(img_bytes)
          if isinstance(ocr_json, dict) and ocr_json.get("cbc"):
            combined_items.extend(ocr_json.get("cbc"))
        except Exception as e:
          print("Vision OCR error on page:", e)

      if not combined_items:
        print(f"⚠️ Vision OCR returned no structured CBC items for {report_id}. Proceeding to text fallback.")
        # fallback to text extraction (attempt)
        parsed = parse_values_from_text(pdf_text)
      else:
        # normalize OCR JSON items
        parsed = parse_values_from_ocr_json({"cbc": combined_items})
        # create merged_text_for_ai as JSON string of extracted CBC so interpreter gets structured input
        merged_text_for_ai = json.dumps({"cbc": combined_items})

    else:
      print(f"📝 Report {report_id} appears to have digital text — using text interpreter")
      # parse numeric values out of text
      parsed = parse_values_from_text(pdf_text)
      if l_text:
        # include l_text (metadata) - append to pdf_text for AI context
        merged_text_for_ai = (l_text + "\n\n" + pdf_text).strip()
      else:
        merged_text_for_ai = pdf_text

    # canonical mapping and compute NLR
    canonical = canonical_map(parsed)

    # fetch previous results for trend analysis
    previous = None
    try:
      if supabase and job.get("patient_id"):
        prev_q = supabase.table("reports").select("ai_results,created_at").eq("patient_id", job.get("patient_id")).order("created_at", desc=True).limit(1).execute()
        rows = prev_q.data if hasattr(prev_q, "data") else prev_q
        if rows:
          previous = rows[0].get("ai_results")
    except Exception:
      previous = None

    trends = trend_analysis(canonical, previous)

    # route engine
    patient_meta = {"age": patient_age, "sex": patient_sex}
    route_info = route_engine(canonical, patient_meta, previous)

    # If AI interpreter expects structured JSON as text, prefer that; otherwise send textual context
    ai_input_text = merged_text_for_ai if merged_text_for_ai else json.dumps({"canonical": canonical})

    ai_json = call_ai_on_report(ai_input_text)

    # build final payload to save
    ai_results_payload = {
      "processed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "scanned": scanned,
      "canonical": canonical,
      "routes": route_info,
      "trends": trends,
      "ai_interpretation": ai_json
    }

    # save to supabase
    supabase.table("reports").update({"ai_status": "completed", "ai_results": ai_results_payload, "ai_error": None}).eq("id", report_id).execute()

    print(f"✅ Report {report_id} processed successfully")
    return {"success": True, "data": ai_results_payload}

  except Exception as e:
    err = f"{type(e).__name__}: {e}"
    print(f"❌ Error processing report {report_id}: {err}")
    traceback.print_exc()
    try:
      supabase.table("reports").update({"ai_status":"failed","ai_error":err}).eq("id", report_id).execute()
    except Exception:
      pass
    return {"error": err}


# ---------- WORKER LOOP ----------

def main():
  print("AMI Worker with Route Engine started… watching for jobs.")

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
        print(f"🔎 Found job: {job_id}")

        supabase.table("reports").update({"ai_status":"processing"}).eq("id", job_id).execute()

        process_report(job)
      else:
        # No jobs: sleep a bit (backoff can be added if desired)
        time.sleep(2)

    except Exception as e:
      print("Worker loop error:", e)
      traceback.print_exc()
      time.sleep(5)


if __name__ == "__main__":
  main()
