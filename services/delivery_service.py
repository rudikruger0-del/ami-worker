# services/delivery_service.py

from datetime import datetime
from uuid import uuid4

from services.email_service import send_email
from services.supabase_client import supabase
from services.whatsapp_service import send_whatsapp_notification


def deliver_prescription(
    *,
    doctor_id: str,
    patient_id: str | None,
    pdf_path: str,
    patient_email: str | None,
    patient_whatsapp: str | None,
    whatsapp_enabled: bool,
):
    """
    Attempts delivery and records outcome.
    Never retries. Never raises to caller.
    """

    if supabase is None:
        raise RuntimeError("Supabase client is not configured")

    delivery_id = str(uuid4())

    email_status = "not_attempted"
    email_error = None

    whatsapp_status = "not_attempted"
    whatsapp_error = None

    # ---- EMAIL (PRIMARY) ----
    if patient_email:
        try:
            send_email(
                to=patient_email,
                subject="Your prescription",
                body="Please find your prescription attached.",
                attachment_url=pdf_path,
            )
            email_status = "sent"
        except Exception as e:
            email_status = "not_delivered"
            email_error = str(e)

    # ---- WHATSAPP (SECONDARY / OPTIONAL) ----
    if whatsapp_enabled and patient_whatsapp:
        try:
            send_whatsapp_notification(to=patient_whatsapp)
            whatsapp_status = "sent"
        except Exception as e:
            whatsapp_status = "not_delivered"
            whatsapp_error = str(e)

    # ---- OVERALL STATUS ----
    if email_status == "sent":
        overall_status = "sent"
    elif whatsapp_status == "sent":
        overall_status = "partially_delivered"
    else:
        overall_status = "not_delivered"

    # ---- WRITE DELIVERY LOG (ALWAYS) ----
    supabase.table("prescription_deliveries").insert({
        "id": delivery_id,
        "doctor_id": doctor_id,
        "patient_id": patient_id,
        "pdf_path": pdf_path,

        "email_address": patient_email,
        "email_status": email_status,
        "email_error": email_error,

        "whatsapp_number": patient_whatsapp,
        "whatsapp_status": whatsapp_status,
        "whatsapp_error": whatsapp_error,

        "overall_status": overall_status,
        "updated_at": datetime.utcnow().isoformat(),
    }).execute()

    return {
        "delivery_id": delivery_id,
        "email_status": email_status,
        "whatsapp_status": whatsapp_status,
        "overall_status": overall_status,
    }
