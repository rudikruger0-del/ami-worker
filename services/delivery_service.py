# services/delivery_service.py

from datetime import datetime
from uuid import uuid4

from services.email_service import send_email
from services.whatsapp_service import send_whatsapp_notification
from supabase import create_client
import os


# --- Supabase client ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def deliver_prescription(
    *,
    doctor_id: str,
    patient_id: str | None,
    prescription_pdf_url: str,
    patient_email: str | None,
    patient_whatsapp: str | None,
    whatsapp_enabled: bool,
):
    """
    Attempts delivery and records outcome.
    Never retries. Never raises to caller.
    """

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
                attachment_url=prescription_pdf_url,
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
        "prescription_pdf_url": prescription_pdf_url,

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
