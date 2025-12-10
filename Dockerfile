FROM python:3.10-slim

# Install Poppler (required for pdf2image)
RUN apt-get update && apt-get install -y \
    poppler-utils \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source files
COPY . /app

# Make Python output unbuffered so Railway logs see every print
ENV PYTHONUNBUFFERED=1

# Run worker with -u (unbuffered) as extra safety
CMD ["python", "-u", "worker.py"]
