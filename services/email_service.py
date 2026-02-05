# services/email_service.py

import smtplib
from email.message import EmailMessage
import os
import requests


def send_email(*, to: str, subject: str, body: str, attachment_url: str):
    """
    Sends an email with a PDF attachment.
    Raises an Exception if anything fails.
    """

    # --- Read SMTP config from environment ---
    SMTP_HOST = os.getenv("SMTP_HOST")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "AMI Health")

    if not SMTP_HOST or not SMTP_USER or not SMTP_PASSWORD:
        raise Exception("SMTP configuration missing")

    if not to:
        raise Exception("Recipient email missing")

    # --- Fetch PDF (from Supabase or private URL) ---
    response = requests.get(attachment_url, timeout=10)
    if response.status_code != 200:
        raise Exception("Could not fetch PDF attachment")

    pdf_bytes = response.content

    # --- Build email ---
    msg = EmailMessage()
    msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_USER}>"
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="prescription.pdf",
    )

    # --- Send via SMTP ---
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
    except Exception as e:
        raise Exception(f"Email send failed: {str(e)}")
