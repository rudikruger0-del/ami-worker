# services/prescription_render_service.py

import io
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader, PdfWriter


def _require_layout_dict(layout_json: dict, key: str) -> dict:
    value = layout_json.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"layout_json.{key} is required and must be an object")
    return value


def _require_layout_number(layout_item: dict, key: str, label: str) -> float:
    value = layout_item.get(key)
    if not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} is required and must be numeric")
    return float(value)


def _wrap_paragraph_line(*, paragraph: str, max_width: float, font_name: str, font_size: float, pdf_canvas: canvas.Canvas) -> list[str]:
    if not paragraph.strip():
        return [""]

    words = paragraph.split()
    wrapped: list[str] = []
    current_line = ""

    for word in words:
        candidate_line = f"{current_line} {word}".strip()
        if pdf_canvas.stringWidth(candidate_line, font_name, font_size) <= max_width:
            current_line = candidate_line
            continue

        if current_line:
            wrapped.append(current_line)

        if pdf_canvas.stringWidth(word, font_name, font_size) <= max_width:
            current_line = word
            continue

        chunk = ""
        for char in word:
            candidate_chunk = f"{chunk}{char}"
            if pdf_canvas.stringWidth(candidate_chunk, font_name, font_size) <= max_width:
                chunk = candidate_chunk
            else:
                if chunk:
                    wrapped.append(chunk)
                chunk = char

        current_line = chunk

    if current_line:
        wrapped.append(current_line)

    return wrapped


def render_prescription_pdf(
    *,
    template_pdf_bytes: bytes,
    patient_name: str | None,
    patient_id: str | None,
    patient_dob: str | None,
    reference: str | None,
    prescription_text: str | None = None,
    layout_json: dict | None = None,
) -> bytes:
    """
    Overlays patient/admin details onto a prescription template PDF and
    renders medication instructions into the template's medication body area.
    """

    # ---- Create overlay PDF in memory ----
    packet = io.BytesIO()
    c = canvas.Canvas(packet, pagesize=A4)

    if not isinstance(layout_json, dict):
        raise ValueError("layout_json is required and must be an object")

    patient_name_layout = _require_layout_dict(layout_json, "patient_name")
    patient_id_layout = _require_layout_dict(layout_json, "patient_id")
    date_layout = _require_layout_dict(layout_json, "date")
    body_area_layout = _require_layout_dict(layout_json, "body_area")
    _require_layout_dict(layout_json, "signature")

    if patient_name:
        c.setFont("Helvetica", _require_layout_number(patient_name_layout, "font_size", "layout_json.patient_name"))
        c.drawString(
            _require_layout_number(patient_name_layout, "x", "layout_json.patient_name"),
            _require_layout_number(patient_name_layout, "y", "layout_json.patient_name"),
            patient_name,
        )

    if patient_id:
        c.setFont("Helvetica", _require_layout_number(patient_id_layout, "font_size", "layout_json.patient_id"))
        c.drawString(
            _require_layout_number(patient_id_layout, "x", "layout_json.patient_id"),
            _require_layout_number(patient_id_layout, "y", "layout_json.patient_id"),
            patient_id,
        )

    c.setFont("Helvetica", _require_layout_number(date_layout, "font_size", "layout_json.date"))
    c.drawString(
        _require_layout_number(date_layout, "x", "layout_json.date"),
        _require_layout_number(date_layout, "y", "layout_json.date"),
        datetime.now().strftime('%Y-%m-%d'),
    )

    body_left = _require_layout_number(body_area_layout, "left", "layout_json.body_area")
    body_right = _require_layout_number(body_area_layout, "right", "layout_json.body_area")
    body_top = _require_layout_number(body_area_layout, "top", "layout_json.body_area")
    body_bottom = _require_layout_number(body_area_layout, "bottom", "layout_json.body_area")
    body_font_size = _require_layout_number(body_area_layout, "font_size", "layout_json.body_area")
    line_spacing = _require_layout_number(body_area_layout, "line_spacing", "layout_json.body_area")

    if body_right <= body_left:
        raise ValueError("layout_json.body_area.right must be greater than left")
    if body_top <= body_bottom:
        raise ValueError("layout_json.body_area.top must be greater than bottom")
    if line_spacing <= 0:
        raise ValueError("layout_json.body_area.line_spacing must be greater than 0")

    normalized_prescription_text = (prescription_text or "").strip()
    if normalized_prescription_text:
        medication_font_name = "Helvetica"
        medication_width = body_right - body_left
        c.setFont(medication_font_name, body_font_size)

        text_object = c.beginText(body_left, body_top)
        text_object.setFont(medication_font_name, body_font_size)
        text_object.setLeading(line_spacing)

        wrapped_lines: list[str] = []
        for paragraph in normalized_prescription_text.splitlines():
            wrapped_lines.extend(
                _wrap_paragraph_line(
                    paragraph=paragraph,
                    max_width=medication_width,
                    font_name=medication_font_name,
                    font_size=body_font_size,
                    pdf_canvas=c,
                )
            )

        current_y = body_top
        for line in wrapped_lines:
            if current_y < body_bottom:
                break

            text_object.textLine(line)
            current_y -= line_spacing

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
