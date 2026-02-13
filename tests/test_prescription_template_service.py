import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.prescription_template_service import save_prescription_template


def test_save_template_local_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_TEMPLATE_DIR", str(tmp_path))

    result = save_prescription_template(
        clinician_id="clinician-123",
        file_bytes=b"%PDF-1.7\nmock",
        supabase=None,
    )

    assert result["template_id"]
    assert result["storage_backend"] == "local"
    saved_path = Path(result["storage_path"])
    assert saved_path.exists()
    assert saved_path.read_bytes().startswith(b"%PDF")


def test_save_template_requires_file_bytes():
    try:
        save_prescription_template("clinician-123", b"", None)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Missing file bytes"