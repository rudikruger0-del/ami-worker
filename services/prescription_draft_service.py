# services/prescription_draft_service.py

import os
from supabase import create_client
from services.prescription_render_service import render_prescription_pdf

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

DRAFT_BUCKET = "prescription_drafts"
TEMPLATE_BUCKET = "prescription-templates"


def generate_prescription_draft(
    *,
    clinician_id: str,
    report_id: str,
    patient_name: str | None,
    patient_id: str | None,
    patient_dob: str | None,
) -> dict:
    """
    Generates a filled prescription PDF draft.
    Nothing is sent.
    """

    # ---- Fetch latest clinician template path ----
    template_records = (
        supabase.table("prescription_templates")
        .select("storage_path")
        .eq("clinician_id", clinician_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    latest_template = template_records.data[0] if template_records.data else None
    template_path = latest_template.get("storage_path") if latest_template else None
    if not template_path:
        raise ValueError("No prescription template uploaded for this clinician")

    # ---- Download template ----
    template_pdf_bytes = supabase.storage.from_(TEMPLATE_BUCKET).download(template_path)

    if template_pdf_bytes in (None, False):
        raise ValueError("Failed to download prescription template from storage")

    if not isinstance(template_pdf_bytes, bytes):
        raise TypeError(
            "Expected bytes from Supabase storage download, "
            f"got {type(template_pdf_bytes).__name__}"
        )

    # ---- Render filled PDF ----
    rendered_pdf = render_prescription_pdf(
        template_pdf_bytes=template_pdf_bytes,
        patient_name=patient_name,
        patient_id=patient_id,
        patient_dob=patient_dob,
        reference=report_id,
    )

    # ---- Store draft ----
    draft_path = f"drafts/{report_id}_{clinician_id}.pdf"

    supabase.storage.from_(DRAFT_BUCKET).upload(
        path=draft_path,
        file=rendered_pdf,
        file_options={
            "content-type": "application/pdf",
            "upsert": True,
        },
    )

    full_path = f"{DRAFT_BUCKET}/{draft_path}"

    return {
        "prescription_draft_path": full_path
    }
