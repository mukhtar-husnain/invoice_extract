FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY invoice_extract/ invoice_extract/
COPY tests/ tests/
COPY eval/ eval/
RUN pip install --no-deps .

RUN useradd --create-home --uid 1000 app
USER app

ENTRYPOINT ["python", "-m", "invoice_extract"]
CMD ["--data", "/data", "--out", "/output", "--extractor", "auto"]
