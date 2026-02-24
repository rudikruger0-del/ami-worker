from services.report_patient_extractor_service import extract_patient_fields_from_report


def test_extract_patient_fields_from_nested_ai_results_dict():
    report_row = {
        "ai_results": {
            "patient": {
                "name": "Jane Doe",
                "id": "P-100",
                "dob": "1988-09-12",
            }
        }
    }

    actual = extract_patient_fields_from_report(report_row)

    assert actual == {
        "patient_name": "Jane Doe",
        "patient_id": "P-100",
        "patient_dob": "1988-09-12",
    }


def test_extract_patient_fields_from_ai_results_json_string():
    report_row = {
        "ai_results": '{"patient": {"full_name": "John Smith", "patient_id": "X-22", "date_of_birth": "1970-01-01"}}'
    }

    actual = extract_patient_fields_from_report(report_row)

    assert actual == {
        "patient_name": "John Smith",
        "patient_id": "X-22",
        "patient_dob": "1970-01-01",
    }


def test_extract_patient_fields_are_optional_when_absent():
    report_row = {"id": "report-1", "ai_results": {"summary": {"impression": "Normal"}}}

    actual = extract_patient_fields_from_report(report_row)

    assert actual == {
        "patient_name": None,
        "patient_id": None,
        "patient_dob": None,
    }
