#!/usr/bin/env python3
# AMI Health Worker V6 — FULL MEDICAL ENGINE (CLEAN)
# Digital-first + OCR fallback, hybrid parser (table-first), CK-MB, age/sex ranges,
# severity TEXT only, confidence scoring, route engine, decorated outputs, Supabase hooks.

import os, io, re, time, json, logging
from typing import Dict, Any, Optional, List, Tuple
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter

# Optional libs
try:
    import pytesseract
    HAS_OCR = True
except:
    HAS_OCR = False

try:
    from supabase import create_client
    HAS_SUPABASE = True
except:
    create_client = None
    HAS_SUPABASE = False

# Config via env
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "reports")
PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "200"))
TEXT_LEN_THRESHOLD = int(os.getenv("TEXT_LEN_THRESHOLD", "80"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ami-worker-v6")

supabase = None
if HAS_SUPABASE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.warning("Supabase init failed: %s", e)
else:
    logger.info("Supabase not configured or package missing.")

# ---------- utilities ----------
def safe_float(x) -> Optional[float]:
    if x is None: return None
    s = str(x)
    s = s.replace(",", ".")
    s = re.sub(r"[^0-9.\-]", "", s)
    if s == "" or s == "-" or s == ".": return None
    try:
        return float(s)
    except:
        return None

def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", s.strip())

def merge_digits(s: str) -> str:
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    s = re.sub(r"(?<=\.)\s+(?=\d)", "", s)
    return s

def current_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ---------- PDF extraction ----------
def extract_text_digital(pdf_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for p in reader.pages:
            txt = p.extract_text() or ""
            pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        logger.debug("digital extract failed: %s", e)
        return ""

def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = img.filter(ImageFilter.MedianFilter(3))
    w,h = img.size
    maxdim = max(w,h)
    if maxdim < 1600:
        factor = max(1, 1600//maxdim)
        img = img.resize((w*factor, h*factor), Image.LANCZOS)
    return img

def extract_text_ocr(pdf_bytes: bytes, dpi: int = PDF_RENDER_DPI) -> str:
    if not HAS_OCR:
        return ""
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi)
    except Exception as e:
        logger.debug("pdf2image failed: %s", e)
        return ""
    pages = []
    for img in imgs:
        try:
            pi = preprocess_image(img)
            txt = pytesseract.image_to_string(pi, lang="eng", config="--oem 3 --psm 6")
            pages.append(txt)
        except Exception:
            pages.append("")
    return "\n".join(pages)

def is_scanned(pdf_text: str) -> bool:
    if len(pdf_text.strip()) < TEXT_LEN_THRESHOLD:
        return True
    if len(re.findall(r"\d", pdf_text)) < 3:
        return True
    return False

# ---------- label mapping ----------
LABEL_MAP = {
    "hb":"Hb","haemoglobin":"Hb","hemoglobin":"Hb","hgb":"Hb",
    "rbc":"RBC","erythrocyte":"RBC",
    "hct":"HCT","haematocrit":"HCT",
    "mcv":"MCV","mean corpuscular volume":"MCV",
    "mch":"MCH","mchc":"MCHC","rdw":"RDW",
    "wbc":"WBC","white cell":"WBC","white blood":"WBC","leukocyte":"WBC","leucocyte":"WBC",
    "neutrophil":"Neutrophils","neut":"Neutrophils",
    "lymphocyte":"Lymphocytes","lymph":"Lymphocytes",
    "monocyte":"Monocytes","eosinophil":"Eosinophils","basophil":"Basophils",
    "platelet":"Platelets","plt":"Platelets",
    "crp":"CRP","c-reactive":"CRP","c reactive protein":"CRP",
    "creatinine":"Creatinine","creat":"Creatinine",
    "sodium":"Sodium","na ":"Sodium"," s-sodium":"Sodium",
    "potassium":"Potassium","k ":"Potassium",
    "chloride":"Chloride","cl ":"Chloride",
    "urea":"Urea",
    "alt":"ALT","alanine aminotransferase":"ALT",
    "ast":"AST","aspartate aminotransferase":"AST",
    "ck":"CK","creatine kinase":"CK",
    "ckmb":"CKMB","ck-mb":"CKMB",
    "calcium":"Calcium","ca ":"Calcium","ca_adj":"CalciumAdj","calcium adj":"CalciumAdj",
    "albumin":"Albumin",
    "co2":"CO2","bicarb":"CO2",
    "egfr":"eGFR"
}

CANONICAL_KEYS = [
 "Hb","MCV","MCH","MCHC","RDW","WBC","Neutrophils","Lymphocytes","Monocytes","Eosinophils","Basophils",
 "NLR","Platelets","RBC","HCT","Creatinine","CRP","Sodium","Potassium","Chloride","Urea","ALT","AST","CK","CKMB",
 "Calcium","CalciumAdj","Albumin","CO2","eGFR"
]

def find_label_key(lbl: str) -> Optional[str]:
    s = lbl.lower().strip()
    s = re.sub(r"[^a-z0-9\- ]", " ", s)
    for k,v in LABEL_MAP.items():
        if k in s:
            return v
    return None

# ---------- parsing helpers ----------
VALUE_RE = r'(-?\d{1,7}\.\d+|-?\d{1,7})'
PERCENT_RE = r'([0-9]{1,3}\.?\d*)\s*%'

def extract_numbers_from_line(line: str) -> List[str]:
    # capture scientific x10 patterns as a number too (take the mantissa)
    m = re.findall(VALUE_RE, line)
    if m:
        return m
    # fallback to percent style
    p = re.findall(PERCENT_RE, line)
    if p:
        return p
    return []

def split_possible_multi_analyte_line(line: str) -> List[str]:
    # Attempt to split lines that contain multiple analytes by recognized labels
    parts = []
    # simple heuristic: split by double spacing
    if re.search(r'\s{2,}', line):
        chunks = re.split(r'\s{2,}', line)
        for c in chunks:
            if c.strip():
                parts.append(c.strip())
        if len(parts) > 1:
            return parts
    # fallback: split when a known short label followed by number occurs repeatedly
    tokens = re.split(r'(\s[A-Za-z\-]{1,6}\s)', line)
    if len(tokens) > 3:
        # reconstruct plausible small pieces
        cur = ""
        for t in tokens:
            cur += t
            if re.search(r'\d', cur) and any(k in cur.lower() for k in ['hb','wbc','rbc','crp','alt','ast','ck','platelet','plt','neut']):
                parts.append(cur.strip())
                cur = ""
        if cur.strip():
            parts.append(cur.strip())
        if len(parts) > 1:
            return parts
    return [line.strip()]

def parse_table_first(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    lines = [l for l in text.splitlines() if l.strip()]
    for line in lines:
        # merge digit fragments first
        line = merge_digits(line)
        # split possible multi analyte lines
        pieces = split_possible_multi_analyte_line(line)
        for piece in pieces:
            # identify label
            # look for "Label   Value   Range" patterns (double-space)
            parts = re.split(r'\s{2,}', piece)
            if len(parts) >= 2:
                label = parts[0].strip()
                key = find_label_key(label)
                if not key:
                    # try first token as label
                    firsttok = part = piece.split()[0]
                    key = find_label_key(firsttok)
                if key:
                    nums = extract_numbers_from_line(piece)
                    if nums:
                        v = safe_float(nums[0])
                        results[key] = {'value': v, 'raw': piece}
                        continue
            # fallback: search label anywhere in piece
            for lab in LABEL_MAP.keys():
                if lab in piece.lower():
                    key = find_label_key(lab)
                    if key:
                        nums = extract_numbers_from_line(piece)
                        if nums:
                            v = safe_float(nums[0])
                            results[key] = {'value': v, 'raw': piece}
                        break
    return results

def parse_free_text(text: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines:
        linelow = line.lower()
        for lab in LABEL_MAP:
            if lab in linelow:
                key = find_label_key(lab)
                if not key:
                    continue
                nums = extract_numbers_from_line(line)
                if nums:
                    v = safe_float(nums[0])
                    results[key] = {'value': v, 'raw': line}
    return results

def hybrid_parse(text: str) -> Dict[str, Dict[str, Any]]:
    text = merge_digits(text)
    table = parse_table_first(text)
    free = parse_free_text(text)
    out = {}
    for k in CANONICAL_KEYS:
        v = None
        if k in table:
            v = table[k].get('value')
        elif k in free:
            v = free[k].get('value')
        out[k] = {'value': v}
    # compute NLR if available (Neutrophils expressed as % or absolute)
    try:
        neut = out.get('Neutrophils', {}).get('value')
        lymph = out.get('Lymphocytes', {}).get('value')
        if neut is not None and lymph is not None and lymph != 0:
            out['NLR'] = {'value': round(float(neut) / float(lymph), 2)}
    except Exception:
        pass
    return out

# ---------- reference ranges and severity (text-only) ----------
def reference_range_for(key: str, sex: str, age: Optional[float]=None) -> Tuple[Optional[float], Optional[float]]:
    sex = (sex or "unknown").lower()
    if key == 'Hb':
        return (12.0, 15.5) if sex == 'female' else (13.0, 17.5)
    if key == 'WBC': return (4.0,11.0)
    if key == 'Platelets': return (150,450)
    if key == 'CRP': return (0,10)
    if key == 'Creatinine': return (45,120) if sex == 'female' else (60,130)
    if key == 'Sodium': return (135,145)
    if key == 'Potassium': return (3.5,5.1)
    if key == 'ALT': return (0,34 if sex=='female' else 40)
    if key == 'AST': return (0,34 if sex=='female' else 40)
    if key == 'CK': return (0,200 if sex=='female' else 250)
    if key == 'CKMB': return (0,7)
    return (None,None)

def severity_text_for(key: str, value: Optional[float], sex: str) -> str:
    if value is None:
        return "normal"
    low, high = reference_range_for(key, sex)
    if low is None:
        return "normal"
    try:
        v = float(value)
    except:
        return "normal"
    if v < low:
        return "low"
    if v > high:
        # map numerical severity to mild/moderate/severe/critical could be added; we keep 'high'
        return "high"
    return "normal"

# ---------- route engine (patterns, ddx, next steps) ----------
def score_severity_numeric(key: str, value: Optional[float], sex: str) -> int:
    # numeric score 1-5 used internally to choose top severity pattern
    if value is None:
        return 1
    try:
        v = float(value)
    except:
        return 1
    k = key.lower()
    if k == 'hb':
        low,high = reference_range_for('Hb', sex)
        if v < (low - 4): return 5
        if v < (low - 2): return 4
        if v < low: return 3
        return 1
    if k == 'wbc':
        if v > 30: return 5
        if v > 20: return 4
        if v > 12: return 3
        return 1
    if k == 'crp':
        if v > 250: return 5
        if v > 100: return 4
        if v > 50: return 3
        if v > 10: return 2
        return 1
    if k in ('neutrophils','nlr'):
        if v > 12: return 5
        if v > 7: return 4
        if v > 3: return 3
        return 1
    if k == 'ck':
        if v > 10000: return 5
        if v > 2000: return 4
        if v > 1000: return 3
        return 1
    if k == 'ckmb':
        if v > 50: return 5
        if v > 20: return 4
        if v > 7: return 3
        return 1
    return 1

def generate_diagnostic_possibilities(canonical: Dict[str, Dict[str, Any]], age: Optional[int], sex: str) -> List[str]:
    out = []
    Hb = canonical.get('Hb', {}).get('value')
    WBC = canonical.get('WBC', {}).get('value')
    Neut = canonical.get('Neutrophils', {}).get('value')
    CRP = canonical.get('CRP', {}).get('value')
    NLR = canonical.get('NLR', {}).get('value')
    CK = canonical.get('CK', {}).get('value')
    CKMB = canonical.get('CKMB', {}).get('value')
    Creat = canonical.get('Creatinine', {}).get('value')

    if (WBC and WBC > 12) or (Neut and Neut > 80) or (NLR and NLR > 10) or (CRP and CRP > 20):
        reasons = []
        if WBC and WBC > 12: reasons.append(f"WBC {WBC}")
        if Neut and Neut > 80: reasons.append(f"neutrophilia {Neut}%")
        if NLR and NLR > 10: reasons.append(f"NLR {NLR}")
        if CRP and CRP > 20: reasons.append(f"CRP {CRP}")
        out.append("Sepsis / bacterial infection — " + "; ".join(reasons))
    if CK and CK > 1000:
        out.append(f"Rhabdomyolysis signal — CK {CK}")
    if CKMB and CKMB > 7:
        out.append(f"Myocardial injury signal — CK-MB {CKMB}")
    if Creat and Creat > 120:
        out.append(f"Acute kidney injury suspected — creatinine {Creat} umol/L")
    if Hb is not None and Hb < 11:
        out.append(f"Anemia — Hb {Hb} g/dL")
    if CRP and CRP > 10:
        out.append(f"Inflammatory response — CRP {CRP} mg/L")
    return out

def route_engine_v6(canonical: Dict[str, Dict[str, Any]], patient_meta: Dict[str, Any], previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    age = patient_meta.get('age')
    sex = patient_meta.get('sex','unknown')
    patterns = []
    routes = []
    next_steps = []
    ddx = []

    def add_pattern(p): 
        if p not in patterns: patterns.append(p)
    def add_route(r):
        if r not in routes: routes.append(r)
    def add_ddx(d):
        if d not in ddx: ddx.append(d)
    # read values
    Hb = canonical.get('Hb', {}).get('value')
    MCV = canonical.get('MCV', {}).get('value')
    WBC = canonical.get('WBC', {}).get('value')
    CRP = canonical.get('CRP', {}).get('value')
    Neut = canonical.get('Neutrophils', {}).get('value')
    NLR = canonical.get('NLR', {}).get('value')
    CK = canonical.get('CK', {}).get('value')
    CKMB = canonical.get('CKMB', {}).get('value')
    Creat = canonical.get('Creatinine', {}).get('value')
    Plate = canonical.get('Platelets', {}).get('value')

    # anemia
    if Hb is not None and severity_text_for('Hb', Hb, sex) != "normal":
        add_pattern("anemia")
        if MCV and MCV < 80:
            add_pattern("microcytic anemia")
            add_route("Iron deficiency route")
            add_ddx("Iron deficiency anemia")
        elif MCV and MCV > 100:
            add_pattern("macrocytic anemia")
            add_route("Macrocytic route")
            add_ddx("B12/Folate deficiency")
        else:
            add_pattern("normocytic anemia")
            add_route("Normocytic anemia route")
            add_ddx("Anaemia of inflammation")

    # infection/inflammation
    if WBC and WBC > 11:
        add_pattern("leukocytosis")
        add_route("Bacterial infection / Sepsis route")
        add_ddx("Bacterial infection")
        next_steps.append("Clinical assessment for sepsis; consider blood cultures and fluids if unstable.")

    if Neut and Neut >= 70:
        add_pattern("neutrophilic predominance")
    if CRP and CRP > 10:
        add_pattern("elevated CRP")
        if CRP > 50:
            add_route("Significant inflammatory response")
            add_ddx("Severe infection / inflammatory disease")

    if NLR and NLR > 10:
        add_pattern("very high NLR")
        add_route("High NLR route")
        next_steps.append("Urgent clinical review for sepsis if clinically unwell.")

    # rhabdomyolysis
    if CK and CK > 1000:
        add_pattern("rhabdomyolysis signal")
        add_route("Rhabdomyolysis route")
        add_ddx("Rhabdomyolysis")
        next_steps.append("Assess for muscle pain/urine colour; check creatinine and electrolytes; consider fluids.")

    # myocardial
    if CKMB and CKMB > 7:
        add_pattern("myocardial injury signal")
        add_route("Myocardial injury route")
        add_ddx("Possible myocardial injury")
        next_steps.append("Consider ECG and troponin; repeat CK-MB as per local protocol.")

    # platelets
    if Plate is not None and Plate < 150:
        add_pattern("thrombocytopenia")
        add_ddx("Immune thrombocytopenia; DIC; marrow suppression")
        next_steps.append("Repeat platelet count; review bleeding risk.")

    # kidney
    if Creat is not None and Creat > 120:
        add_pattern("elevated creatinine")
        add_route("AKI route")
        add_ddx("Acute kidney injury")
        next_steps.append("Repeat creatinine urgently; review meds and urine output.")

    # assemble ddx priority
    ddx_sorted = ddx

    # severity aggregation
    per_key_scores = []
    for k in CANONICAL_KEYS:
        v = canonical.get(k, {}).get('value')
        per_key_scores.append(score_severity_numeric(k, v, sex))
    combined_score = max(per_key_scores) if per_key_scores else 1
    if combined_score <= 1:
        severity_text = 'normal'
    elif combined_score == 2:
        severity_text = 'mild'
    elif combined_score == 3:
        severity_text = 'moderate'
    elif combined_score == 4:
        severity_text = 'severe'
    else:
        severity_text = 'critical'

    color = '#10b981' if combined_score<=1 else ('#facc15' if combined_score==2 else ('#f59e0b' if combined_score==3 else ('#ef4444' if combined_score==4 else '#b91c1c')))
    urgency = 'low' if combined_score<=2 else ('medium' if combined_score==3 else 'high')

    return {
        'patterns': patterns,
        'routes': routes,
        'next_steps': next_steps,
        'differential': ddx_sorted,
        'severity_text': severity_text,
        'urgency_flag': urgency,
        'color': color
    }

# ---------- trends ----------
def trend_analysis(current: Dict[str, Dict[str, Any]], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not previous:
        return {'trend': 'no_previous'}
    diffs = {}
    for k,v in current.items():
        prev = None
        if isinstance(previous, dict):
            prev = previous.get(k, {}).get('value')
        cur = v.get('value')
        if prev is None or cur is None:
            continue
        try:
            delta = cur - prev
            pct = (delta / prev) * 100 if prev != 0 else None
            diffs[k] = {'previous': prev, 'current': cur, 'delta': delta, 'pct_change': pct}
        except:
            pass
    return {'trend': diffs or 'no_change'}

# ---------- save helper ----------
def save_results_to_supabase(report_id: str, ai_results: Dict[str, Any]) -> None:
    if not supabase:
        logger.info("Supabase not configured; skipping save.")
        return
    try:
        payload = {'ai_status': 'completed', 'ai_results': ai_results}
        supabase.table(SUPABASE_TABLE).update(payload).eq('id', report_id).execute()
        logger.info("Saved ai_results for %s", report_id)
    except Exception as e:
        logger.exception("Failed to save to supabase: %s", e)

# ---------- main processing ----------
def process_report_bytes(pdf_bytes: bytes, patient_meta: Dict[str, Any], fetch_previous_fn=None) -> Dict[str, Any]:
    # digital-first
    digital_text = extract_text_digital(pdf_bytes)
    scanned = is_scanned(digital_text)
    ocr_text = ""
    text_used = 'digital'
    if scanned:
        ocr_text = extract_text_ocr(pdf_bytes)
        # if ocr produced usable text prefer ocr
        text = ocr_text or digital_text
        text_used = 'ocr'
    else:
        text = digital_text
    parsed = hybrid_parse(text)
    # attach raw excerpts
    raw_excerpt = text[:5000]
    # previous results
    previous = None
    try:
        if fetch_previous_fn and isinstance(fetch_previous_fn, callable):
            previous = fetch_previous_fn(patient_meta.get('patient_id'))
    except Exception:
        previous = None
    trends = trend_analysis(parsed, previous)
    patient_age = patient_meta.get('age')
    patient_sex = patient_meta.get('sex','unknown')
    routes = route_engine_v6(parsed, {'age': patient_age, 'sex': patient_sex}, previous)
    decorated = {}
    for k in CANONICAL_KEYS:
        val = parsed.get(k, {}).get('value')
        unit = parsed.get(k, {}).get('unit') if isinstance(parsed.get(k, {}).get('unit'), str) else None
        flag, flag_color = 'normal', '#ffffff'
        # use severity mapping for flag
        st = severity_text_for(k, val, patient_sex)
        if st == 'low': flag, flag_color = 'low', '#f59e0b'
        elif st == 'high': flag, flag_color = 'high', '#b91c1c'
        decorated[k] = {
            'value': val,
            'unit': unit,
            'flag': flag,
            'color': flag_color,
            'severity_text': st
        }
    conf = confidence = {
        'score': 90 if text_used=='digital' else 70,
        'quality': 'high' if text_used=='digital' else 'medium',
        'reasons': ['digital' if text_used=='digital' else 'ocr']
    }
    ai_results = {
        'canonical': parsed,
        'decorated': decorated,
        'routes': routes,
        'trends': trends,
        'raw_text_excerpt': raw_excerpt,
        'scanned': scanned,
        'confidence': conf,
        'processed_at': current_ts()
    }
    return ai_results

# CLI/test
if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", help="path to pdf")
    parser.add_argument("--sex", default="female")
    parser.add_argument("--age", type=int, default=30)
    args = parser.parse_args()
    if not args.pdf:
        print("Provide --pdf file")
        sys.exit(1)
    with open(args.pdf,"rb") as f:
        b = f.read()
    out = process_report_bytes(b, {'age': args.age, 'sex': args.sex, 'patient_id': 'local-test'})
    print(json.dumps(out, indent=2))
