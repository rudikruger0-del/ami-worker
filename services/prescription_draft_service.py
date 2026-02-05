# services/prescription_draft_service.py

import os
from supabase import create_client
from services.prescription_render_service import render_prescription_pdf

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

DRAFT_BUCKET = "prescription_drafts"


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

    # ---- Fetch clinician template path ----
    clinician = (
        supabase.table("clinicians")
        .select("prescription_template_path")
        .eq("id", clinician_id)
        .single()
        .execute()
    )

    template_path = clinician.data.get("prescription_template_path")
    if not template_path:
        raise ValueError("No prescription template uploaded for this clinician")

    # ---- Download template ----
    bucket, path = template_path.split("/", 1)
    template_bytes = (
        supabase.storage.from_(bucket)
        .download(path)
    )

    template_pdf_bytes = (
        template_bytes.data
        if hasattr(template_bytes, "data")
        else template_bytes
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
