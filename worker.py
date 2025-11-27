def call_ami_ai(extracted_text, pdf_base64):

    # 1️⃣ Clean + limit extracted text
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 15000:   # prevent context overflow
            extracted_text = extracted_text[:15000]

    # 2️⃣ Only send PDF base64 when absolutely needed
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 50:
        pdf_chunk = pdf_base64[:50000]  # limit PDF fallback to 50k chars max

    system_message = """
You are AMI — an advanced laboratory interpretation AI.

Output STRICT JSON ONLY.
Never include text, markdown, explanations, or commentary.

JSON format:
{
  "summary": [],
  "trend_summary": [],
  "flagged_results": [],
  "interpretation": [],
  "risk_level": "",
  "recommendations": [],
  "cbc_values": {},
  "disclaimer": "This is not medical advice."
}
"""

    user_message = f"""
Extracted Text (clean & truncated):
{extracted_text}

PDF Fallback (may be empty):
{pdf_chunk}

Return ONLY valid JSON.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o-mini",
            max_output_tokens=1500,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        ai_text = response.output_text

        # JSON parse
        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}
