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
    prescription_text: str | None = None,
) -> bytes:
    """
    Overlays patient/admin details onto a prescription template PDF and
    renders medication instructions into the template's medication body area.
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

    # Medication body area tuned to the standard doctor template layout.
    medication_x = 55
    medication_top_y = 540
    medication_width = 500
    medication_bottom_y = 240
    medication_font_name = "Helvetica"
    medication_font_size = 11
    medication_line_height = 14

    normalized_prescription_text = (prescription_text or "").strip()
    if normalized_prescription_text:
        c.setFont(medication_font_name, medication_font_size)

        text_object = c.beginText(medication_x, medication_top_y)
        text_object.setFont(medication_font_name, medication_font_size)
        text_object.setLeading(medication_line_height)

        wrapped_lines: list[str] = []
        for paragraph in normalized_prescription_text.splitlines():
            if not paragraph.strip():
                wrapped_lines.append("")
                continue

            current_line = ""
            for word in paragraph.split():
                candidate = f"{current_line} {word}".strip()
                if c.stringWidth(candidate, medication_font_name, medication_font_size) <= medication_width:
                    current_line = candidate
                else:
                    if current_line:
                        wrapped_lines.append(current_line)
                    current_line = word

            if current_line:
                wrapped_lines.append(current_line)

        current_y = medication_top_y
        for line in wrapped_lines:
            if current_y < medication_bottom_y:
                break

            text_object.textLine(line)
            current_y -= medication_line_height

        c.drawText(text_object)

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
