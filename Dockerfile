FROM python:3.10-slim

# Install Poppler + Tesseract OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "worker.py"]
