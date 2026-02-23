import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.prescription_template_service import save_prescription_template


class _FakeStorageBucket:
    def __init__(self):
        self.upload_calls = []

    def upload(self, storage_path, file_bytes, file_options):
        self.upload_calls.append(
            {
                "storage_path": storage_path,
                "file_bytes": file_bytes,
                "file_options": file_options,
            }
        )


class _FakeStorage:
    def __init__(self):
        self.bucket = _FakeStorageBucket()

    def from_(self, bucket_name):
        assert bucket_name == "prescription-templates"
        return self.bucket


class _FakeInsert:
    def execute(self):
        return {"ok": True}


class _FakeTable:
    def __init__(self):
        self.insert_payload = None

    def insert(self, payload):
        self.insert_payload = payload
        return _FakeInsert()


class _FakeSupabase:
    def __init__(self):
        self.storage = _FakeStorage()
        self.table_name = None
        self.table_ref = _FakeTable()

    def table(self, table_name):
        self.table_name = table_name
        return self.table_ref


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


def test_save_template_supabase_includes_clinician_id():
    supabase = _FakeSupabase()

    result = save_prescription_template(
        clinician_id="clinician-abc",
        file_bytes=b"%PDF-1.7\nmock",
        supabase=supabase,
    )

    assert result["storage_backend"] == "supabase"
    assert supabase.table_name == "prescription_templates"
    assert supabase.table_ref.insert_payload is not None
    assert supabase.table_ref.insert_payload["clinician_id"] == "clinician-abc"


def test_save_template_requires_file_bytes():
    try:
        save_prescription_template("clinician-123", b"", None)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert str(exc) == "Missing file bytes"
