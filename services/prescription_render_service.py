# services/prescription_render_service.py

import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter


def render_prescription_pdf(
    *,
    template_pdf_bytes: bytes,
    patient_name: str | None,
    patient_id: str | None,
    patient_dob: str | None,
    reference: str | None,
) -> bytes:
    """
    Overlays patient/admin details onto a prescription template PDF.
    Medication area is intentionally untouched.
    """

    # ---- Create overlay PDF in memory ----
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)

    # Coordinates are intentionally conservative
    # These can be tuned per practice later if needed

    if patient_name:
        c.drawString(50, 760, f"Patient: {patient_name}")

    if patient_id:
        c.drawString(50, 740, f"ID: {patient_id}")

    if patient_dob:
        c.drawString(250, 740, f"DOB: {patient_dob}")

    c.drawString(400, 760, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    if reference:
        c.drawString(400, 740, f"Ref: {reference}")

    c.save()
    packet.seek(0)

    overlay_pdf = PdfReader(packet)
    template_pdf = PdfReader(io.BytesIO(template_pdf_bytes))

    writer = PdfWriter()

    # ---- Merge overlay onto first page only ----
    first_page = template_pdf.pages[0]
    first_page.merge_page(overlay_pdf.pages[0])
    writer.add_page(first_page)

    # ---- Append remaining pages unchanged ----
    for page in template_pdf.pages[1:]:
        writer.add_page(page)

    output = io.BytesIO()
    writer.write(output)
    output.seek(0)

    return output.read()
