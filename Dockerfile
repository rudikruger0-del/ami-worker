FROM python:3.10-slim

# Install Poppler (required for pdf2image) + Tesseract OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source files
COPY . /app

# Python unbuffered for better logs
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "worker.py"]
