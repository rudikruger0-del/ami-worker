FROM python:3.10-slim

# Install Poppler for pdf2image
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL source files
COPY . /app

# Run worker
CMD ["python", "worker.py"]
