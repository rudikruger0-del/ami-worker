import os
import uuid
import logging
from datetime import datetime
from pathlib import Path

from supabase import Client


logger = logging.getLogger(__name__)


def save_prescription_template(
    clinician_id: str,
    file_bytes: bytes,
    supabase: Client | None,
) -> dict:
    """Persist a clinician template to Supabase Storage, with local fallback."""

    if not clinician_id:
        raise ValueError("Missing clinician_id")
    if not file_bytes:
        raise ValueError("Missing file bytes")

    template_id = str(uuid.uuid4())
    storage_path = f"{clinician_id}/{template_id}.pdf"
    now = datetime.utcnow().isoformat()

    if supabase:
        supabase.storage.from_("prescription-templates").upload(
            storage_path,
            file_bytes,
            file_options={
                "content-type": "application/pdf",
                "upsert": False,
            },
        )
        supabase.table("prescription_templates").insert(
            {
                "id": template_id,
                "clinician_id": clinician_id,
                "storage_path": storage_path,
                "storage_backend": "supabase",
                "created_at": now,
            }
        ).execute()

        return {
            "template_id": template_id,
            "storage_path": storage_path,
            "storage_backend": "supabase",
        }

    base_dir = Path(os.getenv("LOCAL_TEMPLATE_DIR", "storage/prescription-templates"))
    local_path = base_dir / clinician_id / f"{template_id}.pdf"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(file_bytes)
    logger.warning(
        "Supabase client not configured. Stored template locally at %s",
        local_path,
    )

    return {
        "template_id": template_id,
        "storage_path": str(local_path),
        "storage_backend": "local",
    }
