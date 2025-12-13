FROM python:3.10-slim

# --- System dependencies ---
# poppler-utils → pdf2image
# tesseract-ocr → pytesseract (fallback OCR)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# --- App directory ---
WORKDIR /app

# --- Python dependencies ---
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# --- Copy worker ---
COPY . .

# --- Runtime settings ---
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "worker.py"]
