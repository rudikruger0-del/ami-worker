def call_ami_ai(extracted_text, pdf_base64):

    # 1️⃣ Clean + safely limit extracted text
    if extracted_text:
        extracted_text = extracted_text.strip()
        if len(extracted_text) > 20000:
            extracted_text = extracted_text[:20000]  # avoid context overflow

    # 2️⃣ Only send PDF fallback when text is missing or too short
    pdf_chunk = ""
    if not extracted_text or len(extracted_text) < 100:
        pdf_chunk = pdf_base64[:60000]  # PDF fallback capped at 60k chars

    system_message = """
You are AMI — an advanced laboratory interpretation AI.
You analyse blood tests (CBC, chemistry, markers of infection/inflammation).

CRITICAL RULES:
1. Base your interpretation ONLY on values found in the provided text/PDF.
2. NEVER invent values, NEVER guess diagnoses (e.g. “knee infection”).
3. If the labs are non-specific or incomplete, clearly say so.
4. If infection markers (WBC, neutrophils, CRP, ESR) are missing, state that infection risk cannot be determined from labs alone.
5. Output STRICT JSON ONLY — no markdown, no explanations outside JSON.

Expected JSON structure:
{
  "summary": [],
  "trend_summary": [],
  "flagged_results": [],
  "interpretation": [],
  "risk_level": "",
  "recommendations": [],
  "cbc_values": {},
  "chemistry_values": {},
  "disclaimer": "This is not medical advice."
}
"""

    user_message = f"""
Extracted Lab Text (clean):
{extracted_text}

PDF Fallback (limited):
{pdf_chunk}

Extract ALL CBC and general chemistry values you can find.
ONLY return the JSON object described in the system prompt.
"""

    try:
        response = openai_client.responses.create(
            model="gpt-4o",   # ⬅️ UPGRADED MODEL (supports long PDFs)
            max_output_tokens=1800,
            input=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        )

        ai_text = response.output_text.strip()

        # --- Clean AI output to ensure JSON-only ---
        # Remove backticks if model adds any
        ai_text = ai_text.replace("```json", "").replace("```", "").strip()

        # Ensure output starts with "{" and ends with "}"
        if "{" in ai_text and "}" in ai_text:
            ai_text = ai_text[ai_text.index("{"): ai_text.rindex("}") + 1]

        return json.loads(ai_text)

    except Exception as e:
        print("❌ AI error:", e)
        return {"error": str(e)}
