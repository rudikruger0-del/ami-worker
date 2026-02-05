# services/whatsapp_service.py

import os
import requests


WHATSAPP_ENABLED = os.getenv("WHATSAPP_ENABLED", "false").lower() == "true"
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")


def send_whatsapp_notification(*, to: str):
    """
    Sends a NON-CLINICAL WhatsApp notification.
    Raises Exception only if WhatsApp is enabled and sending fails.
    """

    if not WHATSAPP_ENABLED:
        # WhatsApp intentionally disabled — silent exit
        return

    if not to:
        raise Exception("WhatsApp number missing")

    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_ID:
        raise Exception("WhatsApp configuration missing")

    message_text = (
        "A prescription has been sent to you by email.\n"
        "No medical or medication details are included in this message."
    )

    url = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {
            "body": message_text
        }
    }

    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers, timeout=10)

    if response.status_code not in (200, 201):
        raise Exception(f"WhatsApp send failed: {response.text}")
