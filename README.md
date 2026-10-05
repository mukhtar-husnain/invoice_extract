# Invoice Extraction Pipeline

Extract structured fields from invoice PDFs and write JSON with values, bounding boxes, and flags for human review.

Point the tool at a folder of PDFs and it produces one JSON file per invoice plus a summary of how many fields and documents need a person to check.

## Run with Docker

**Prerequisite:** Docker Desktop or Docker Engine with Compose v2. Clone the repo and keep invoice PDFs in `./data`.

### Rules only (no API key, no model)

```bash
docker compose build
docker compose run --rm app --extractor rules
```

### External vision API

```bash
cp .env.example .env   # fill INVOICE_LLM_PROVIDER and that provider's key
docker compose run --rm app --extractor api
```

### Ollama inside Docker (one-time ~6 GB model pull; give Docker ~8 GB RAM)

```bash
docker compose --profile ollama up -d ollama
docker compose --profile ollama run --rm ollama-pull
docker compose --profile ollama run --rm app --extractor ollama
```

### Ollama already installed on the host (faster on Apple Silicon)

```bash
docker compose run --rm -e INVOICE_OLLAMA_URL=http://host.docker.internal:11434 app --extractor ollama
```

On Linux, Compose maps `host.docker.internal` to the host. Host Ollama must listen beyond loopback (`OLLAMA_HOST=0.0.0.0 ollama serve`).

### Results

- `./output/<invoice>.json` and `./output/summary.json`
- Other PDFs: add them under `./data`, or mount another folder with `-v /path/to/pdfs:/data:ro`

### Tests

```bash
docker compose run --rm --entrypoint pytest app
```

### Clean up

```bash
docker compose --profile ollama down
```

Add `-v` to also delete the Ollama model volume (~6 GB).

### Troubleshooting

| Problem | What to try |
|---------|-------------|
| Ollama out of memory | Raise Docker Desktop memory limit, or use host Ollama |
| Missing API key in api mode | Set the key in `.env`; the run fails explicitly (no fallback) |
| "Ollama is not reachable" | Check `docker compose --profile ollama ps` or `INVOICE_OLLAMA_URL` |
| Empty output | Confirm `./data` is mounted and contains PDFs |
| Permission denied on `./output` | `mkdir -p output` before running |

## Quick start

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
brew install tesseract tesseract-lang
```

Run on the bundled sample invoices:

```bash
python -m invoice_extract --data data --out output --extractor rules
```

This writes:

- `output/<documentId>.json` — one file per PDF
- `output/summary.json` — counts of fields and invoices requiring review

## Output format

Each invoice JSON looks like this:

```json
{
  "documentId": "digital_01.pdf",
  "invoice": {
    "invoiceNumber": {
      "value": "INV-001",
      "bbox": {"page": 1, "x1": 100, "y1": 100, "x2": 250, "y2": 130},
      "isHumanReviewRequired": false,
      "reviewReasons": []
    }
  },
  "lineItems": [],
  "isHumanReviewRequired": false,
  "reviewReasons": []
}
```

Every field (header and line item) always includes `value`, `bbox`, `isHumanReviewRequired`, and `reviewReasons`. Missing values use `"value": null` and `"bbox": null`.

Mandatory header fields: `invoiceNumber`, `invoiceDate`, `supplierName`, `buyerName`, `currency`, `totalAmount`.

## Extractor modes

The same pipeline runs on every PDF. You choose how fields are mapped:

| Mode | Command | Best for |
|------|---------|----------|
| **auto** (default) | `--extractor auto` | Tries Ollama first, then an API key, then rules |
| **ollama** | `--extractor ollama` | Local, private runs with a vision model (~8 GB RAM) |
| **api** | `--extractor api` | Highest quality when you have OpenAI, Anthropic, or Gemini access |
| **rules** | `--extractor rules` | No model needed; label matching only (fastest to set up) |

### Ollama (local, offline after setup)

```bash
brew install ollama
ollama serve
ollama pull qwen2.5vl:7b
python -m invoice_extract --extractor ollama
```

Override the model with `INVOICE_OLLAMA_MODEL=my-model`.

### External vision API

```bash
export INVOICE_LLM_PROVIDER=openai   # or anthropic, gemini
export OPENAI_API_KEY=sk-...
python -m invoice_extract --extractor api
```

Supported environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`.

## How it works

Processing follows seven stages for every PDF, whether it is a clean digital file, a scan, handwriting, or a multilingual document.

```
  PDF files
      │
      ▼
  1. Ingest          Walk data/**/*.pdf; documentId = filename
      │
      ▼
  2. Render          Rasterize each page at 200 DPI; read native text-layer words
      │
      ▼
  3. OCR             Tesseract fills in words when a page has little or no text layer
      │
      ▼
  4. Extract         Ollama, vision API, or label rules pick field values + token ids
      │
      ▼
  5. Align           Build bounding boxes from cited tokens; validate value ↔ box match
      │
      ▼
  6. Review          Flag missing fields, low confidence, and amount mismatches
      │
      ▼
  7. Emit            Write per-invoice JSON and summary.json
```

**Ingest.** All PDFs under `--data` are processed the same way. Folder names (`digital/`, `scanned/`, etc.) are corpus labels only — the pipeline does not treat them differently.

**Render & OCR.** PyMuPDF renders pages and extracts word boxes from digital text layers. Image-only pages (scans, handwriting) fall back to Tesseract with English, simplified/traditional Chinese, and Vietnamese language packs.

**Extract.** Three backends share one contract: return a printed text value and the token ids that support it. Ollama runs locally; the API path sends page images to an external provider; rules mode matches labels like "Invoice No" and "Total" without a model.

**Align & review.** Token boxes are merged into field bounding boxes. The review step marks fields that are missing, uncertain, or arithmetically inconsistent, then rolls those up to a document-level `isHumanReviewRequired` flag.

### Project layout

| Path | Purpose |
|------|---------|
| `invoice_extract/cli.py` | Command-line entry point |
| `invoice_extract/pipeline.py` | Orchestrates the full run |
| `invoice_extract/render.py` | PDF rasterization and native word extraction |
| `invoice_extract/ocr.py` | Tesseract fallback for image-only pages |
| `invoice_extract/extract.py` | Ollama and external vision API clients |
| `invoice_extract/local_extract.py` | Label and pattern rules (no model) |
| `invoice_extract/align.py` | Token union and bbox assignment |
| `invoice_extract/review.py` | Review flags and amount checks |
| `invoice_extract/schema.py` | JSON output models |
| `data/` | Sample invoice PDFs (13 files) |
| `output/` | Generated JSON results |
| `tests/` | Unit tests |

## Coordinates

- Pages are rasterized at **200 DPI**.
- `bbox` uses integer pixel corners: `x1`, `y1` (top-left), `x2`, `y2` (bottom-right), origin at top-left, `y` increases downward.
- `page` is 1-based.
- When a value spans multiple tokens, the box is the union of those token boxes.

## Review flags

| Reason | Meaning |
|--------|---------|
| `FIELD_MISSING` | A mandatory header field has no value |
| `LOW_CONFIDENCE` | Weak OCR/model confidence, value not grounded in cited tokens, or value spans pages |
| `AMOUNT_INCONSISTENT` | `quantity × unitPrice ≠ lineAmount`, or header totals do not add up |

Document-level `isHumanReviewRequired` is `true` when any field is flagged or any mandatory field is missing.

## Tests

```bash
pytest -q
```

Tests cover review rules, alignment, and schema round-trips. They do not require Ollama, API keys, or network access.

## FAQ

### What kinds of invoices can this handle?

The sample corpus includes digital PDFs with text layers, scanned images, handwritten invoices, and multilingual documents (Hong Kong, Vietnam, Singapore). The same pipeline runs on all of them. Quality depends on the extractor: rules mode works best on clean digital invoices; scans and handwriting benefit from Tesseract being installed and generally improve further with Ollama or an API vision model.

### Which extractor mode should I use?

Use **rules** for a quick offline demo or when you have no GPU and no API key. Use **Ollama** when you want local, private extraction and have ~8 GB free RAM for the vision model. Use **api** when accuracy matters most and you are fine sending page images to a cloud provider. **auto** picks Ollama if it is running, otherwise an API key if set, otherwise rules.

### Why are so many fields flagged for human review?

Review flags are intentional. The pipeline prefers to flag uncertainty rather than silently accept a bad value. Common causes: rules mode guessing labels on complex layouts (`LOW_CONFIDENCE`), mandatory fields not found on scans without OCR (`FIELD_MISSING`), or line amounts that do not multiply correctly (`AMOUNT_INCONSISTENT`). Better extractors and installed Tesseract language packs reduce false flags.

### Why is there no accuracy or missed-review rate in the summary?

This project outputs extraction results and review counts only. Measuring "incorrect values that were not flagged" requires a hidden gold-standard dataset held by an external evaluator. The summary reports `fieldsRequiringHumanReview` and `invoicesRequiringHumanReview` — not a percentage accuracy score.

### Do I need internet to run this?

**Setup** requires internet once (Python packages, Tesseract). **Rules mode** runs fully offline after that. **Ollama** needs internet once to install the model, then runs offline. **API mode** needs internet on every run. **Auto** follows whichever backend it selects.

### What do the bounding boxes represent?

Each `bbox` marks where the extracted text appears on the rasterized page at 200 DPI. Reviewers or downstream UI can overlay the box on the page image to verify the value. Boxes come from cited OCR or text-layer tokens; if a model cannot point to tokens, the field is flagged `LOW_CONFIDENCE`.
