import base64
from supabase import Client
from datetime import datetime

def upload_prescription_template_action(payload: dict, supabase: Client):
    """
    Uploads a clinician's prescription template PDF
    and stores the path on clinicians.prescription_template_path
    """

    clinician_id = payload.get("clinician_id")
    file_base64 = payload.get("file_base64")

    if not clinician_id:
        raise Exception("Missing clinician_id")

    if not file_base64:
        raise Exception("Missing file_base64")

    # Decode PDF
    try:
        pdf_bytes = base64.b64decode(file_base64)
    except Exception:
        raise Exception("Invalid base64 PDF")

    # Storage path
    storage_path = f"clinician_{clinician_id}/prescription_template.pdf"

    # Upload to Supabase Storage
    supabase.storage.from_("prescription-templates").upload(
        storage_path,
        pdf_bytes,
        {
            "content-type": "application/pdf",
            "upsert": True
        }
    )

    # Save path on clinician record
    supabase.table("clinicians").update({
        "prescription_template_path": storage_path,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", clinician_id).execute()

    return {
        "status": "success",
        "storage_path": storage_path
    }
