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

### Score against the answer key

```bash
docker compose run --rm --entrypoint python app -m invoice_extract.evaluate --pred /output
```

### Everything in one go

`scripts/docker_verify.sh` builds the image, runs the tests, extracts all PDFs (with Tesseract OCR), scores them, and prints the image size (about 360 MB). Pass `ollama` or `api` to use a model instead of rules.

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

Without Tesseract, the pipeline still runs, but image-only pages (scans, handwriting) produce empty, flagged fields. Docker includes Tesseract, so use it if installing Tesseract locally is a problem.

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

Header fields (always all 13 keys): `invoiceNumber`, `invoiceDate`, `dueDate`, `supplierName`, `supplierAddress`, `buyerName`, `buyerTaxId`, `currency`, `taxAmount`, `totalAmount`, `amountDue`, `paymentTerms`, `purchaseOrderNumber`.

Line-item fields (always all 6 keys per row): `description`, `quantity`, `unitPrice`, `taxRate`, `taxAmount`, `lineAmount`.

Mandatory header fields: `invoiceNumber`, `invoiceDate`, `supplierName`, `buyerName`, `currency`, `totalAmount`. If any is missing, the invoice is flagged with `FIELD_MISSING`.

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

Override the model with `INVOICE_OLLAMA_MODEL=my-model` (for example `qwen2.5vl:3b` on machines with less RAM).

**Speed warning.** Vision models need a GPU or Apple Silicon to be practical. On an Intel Mac with 16 GB RAM and no GPU, `qwen2.5vl:3b` had not finished a single one-page invoice after 26 minutes. On such machines use `rules` or `api`.

### External vision API

```bash
export INVOICE_LLM_PROVIDER=openai   # or anthropic, gemini
export OPENAI_API_KEY=sk-...
python -m invoice_extract --extractor api
```

### Environment variables

All are optional; blank values (as in `.env.example`) mean "use the default".

| Variable | Default | Purpose |
|----------|---------|---------|
| `INVOICE_LLM_PROVIDER` | `openai` | `openai`, `anthropic`, or `gemini` for api mode |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | none | Key for the chosen provider |
| `INVOICE_OPENAI_MODEL` / `INVOICE_ANTHROPIC_MODEL` / `INVOICE_GEMINI_MODEL` | `gpt-4o` / `claude-3-5-sonnet-20241022` / `gemini-2.0-flash` | Provider model |
| `INVOICE_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama server |
| `INVOICE_OLLAMA_MODEL` | `qwen2.5vl:7b` | Ollama vision model |
| `INVOICE_OLLAMA_NUM_CTX` | `16384` | Ollama context window (prompt includes every word box) |
| `INVOICE_OLLAMA_TIMEOUT_S` | `1800` | Per-invoice Ollama timeout |
| `INVOICE_IMAGE_MAX_SIDE` | `1280` | Longest side, in pixels, of page images sent to a model |

## Evaluation and results

`eval/gold.json` is a hand-checked answer key for the 13 sample invoices. A field's expected value is a string, `null` (not printed on the invoice), or a list of acceptable values when the invoice is ambiguous (for example, receipts printing both "Total Sales" and "Payment Due").

```bash
python -m invoice_extract.evaluate --gold eval/gold.json --pred output --report eval/report.json
```

Values are compared after normalization: amounts as numbers, dates as calendar days (day-first), currencies as ISO codes, identifiers by letters and digits only, and other text by fuzzy match (at least 85% similar). Line items are paired with answer-key rows by description and amount.

- **Missed-review rate** = values that are wrong and not flagged ÷ all extracted (non-null) values.
- The command adds an `evaluation` block to `output/summary.json` and writes per-field and per-invoice detail plus every error to `eval/report.json`.

To score new PDFs, add their expected values to `eval/gold.json` under their `documentId`.

### Current results (rules mode with Tesseract OCR, from `scripts/docker_verify.sh`)

| Metric | Value |
|--------|-------|
| Invoices processed | 13 |
| Fields requiring human review | 76 (36 low confidence, 24 missing mandatory, 16 invalid format, 10 amount mismatch; a field can have several reasons) |
| Invoices requiring human review | 13 of 13 |
| Extracted values | 161 |
| Wrong values | 32, of which 31 flagged |
| **Missed-review rate** | **0.62%** (1 of 161) |
| Precision (extracted values that are correct) | 80.1% |
| Recall (answer-key values found) | 42.3% |

How this splits by invoice type:

- **Digital and multilingual text-layer invoices** (`digital_*`, Hong Kong, Vietnam) are read well: 83–100% of answer-key values found, with no unflagged errors.
- **Scanned and handwritten invoices** are where rules mode struggles. Tesseract reads printed scans reasonably (`scanned_03` finds 56%), but handwriting comes out as noise. Rules mode therefore extracts little from handwriting, and most of what it does extract fails format checks and is flagged rather than silently accepted.
- The one unflagged error is a wine description in `scanned_03` that wraps onto a second line which OCR placed out of order, so only the first line was kept.

Every invoice is flagged because each has at least one guessed field (for example, a supplier name found by company suffix rather than a label) or a missing mandatory field. A vision model (`--extractor ollama` or `api`) is the intended path for handwriting.

Without Tesseract (a local run with no OCR installed), only the 5 text-layer invoices are read. That run scores 0 unflagged errors out of 107 extracted values (97.2% precision, 34.1% recall).

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
| `invoice_extract/normalize.py` | Amount, date, currency, and text normalization |
| `invoice_extract/evaluate.py` | Scoring against the answer key (missed-review rate) |
| `eval/gold.json` | Hand-checked answer key for the sample invoices |
| `scripts/docker_verify.sh` | Build, test, extract, and score in Docker |
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
| `AMOUNT_INCONSISTENT` | `quantity × unitPrice ≠ lineAmount`, line tax ≠ amount × rate, line items do not sum to the total, tax ≥ total, or amount due > total |
| `INVALID_FORMAT` | Value does not fit its type: an amount that is really a date, percentage, or tax ID; an unparseable date or currency; an invoice number with fewer than 3 digits; a tax ID with fewer than 6 digits; a name or description containing stray OCR symbols |

Two header fields read from the exact same box are both flagged `LOW_CONFIDENCE` (except `totalAmount` and `amountDue`, which are often the same printed number).

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

### How is the missed-review rate measured?

Against the hand-checked answer key in `eval/gold.json`; see [Evaluation and results](#evaluation-and-results). The answer key covers the 13 sample invoices only, so new PDFs need their own entries before they can be scored.

### Do I need internet to run this?

**Setup** requires internet once (Python packages, Tesseract). **Rules mode** runs fully offline after that. **Ollama** needs internet once to install the model, then runs offline. **API mode** needs internet on every run. **Auto** follows whichever backend it selects.

### What do the bounding boxes represent?

Each `bbox` marks where the extracted text appears on the rasterized page at 200 DPI. Reviewers or downstream UI can overlay the box on the page image to verify the value. Boxes come from cited OCR or text-layer tokens; if a model cannot point to tokens, the field is flagged `LOW_CONFIDENCE`.
