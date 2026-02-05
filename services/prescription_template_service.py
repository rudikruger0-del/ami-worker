# services/prescription_template_service.py

import os
from supabase import create_client


SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

BUCKET_NAME = "prescription_templates"


def upload_prescription_template(
    *,
    clinician_id: str,
    pdf_bytes: bytes
) -> dict:
    """
    Uploads or replaces a clinician's prescription template PDF.
    """

    if not pdf_bytes:
        raise ValueError("Empty PDF upload")

    storage_path = f"doctor_{clinician_id}.pdf"

    # ---- Upload (overwrite allowed) ----
    supabase.storage.from_(BUCKET_NAME).upload(
        path=storage_path,
        file=pdf_bytes,
        file_options={
            "content-type": "application/pdf",
            "upsert": True,
        },
    )

    full_path = f"{BUCKET_NAME}/{storage_path}"

    # ---- Persist reference on clinician record ----
    supabase.table("clinicians").update({
        "prescription_template_path": full_path
    }).eq("id", clinician_id).execute()

    return {
        "prescription_template_path": full_path
    }
