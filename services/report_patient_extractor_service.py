import json


def _coerce_json_object(value) -> dict:
    if isinstance(value, dict):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {}

    return {}


def extract_patient_fields_from_report(report_row: dict | None) -> dict:
    """
    Extracts patient-facing fields from report.ai_results when available.
    Falls back to None for each field to keep prescription generation optional.
    """
    if not isinstance(report_row, dict):
        return {"patient_name": None, "patient_id": None, "patient_dob": None}

    ai_results = _coerce_json_object(report_row.get("ai_results"))
    patient = ai_results.get("patient") if isinstance(ai_results.get("patient"), dict) else {}

    patient_name = (
        patient.get("name")
        or patient.get("full_name")
        or ai_results.get("patient_name")
    )

    patient_id = (
        patient.get("id")
        or patient.get("patient_id")
        or ai_results.get("patient_id")
    )

    patient_dob = (
        patient.get("dob")
        or patient.get("date_of_birth")
        or ai_results.get("patient_dob")
    )

    return {
        "patient_name": patient_name or None,
        "patient_id": patient_id or None,
        "patient_dob": patient_dob or None,
    }

