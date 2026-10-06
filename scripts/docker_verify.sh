#!/usr/bin/env bash
# Build the image, run the tests, extract every PDF in ./data with OCR, then score against eval/gold.json.
# Usage: scripts/docker_verify.sh [extractor]   (default: rules)
set -euo pipefail
cd "$(dirname "$0")/.."

extractor="${1:-rules}"
mkdir -p output

docker compose build app
docker compose run --rm --entrypoint pytest app -q -p no:warnings
docker compose run --rm app --data /data --out /output --extractor "$extractor"
docker compose run --rm --entrypoint python app -m invoice_extract.evaluate \
  --gold eval/gold.json --pred /output --report eval/report.json
docker image ls project2-app --format 'Image size: {{.Size}}'
