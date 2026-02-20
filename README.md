# PDF OCR to TXT (Korean)

A GUI tool to extract text from PDFs. It supports OCR for scanned Korean manuals, plus layout-aware OCR, table handling, and quality reports.

## Features
- Single PDF or batch folder processing
- Auto mode: try text extraction first, then OCR if needed
- OCR-only and text-only modes
- Image preprocessing: background normalize, gentle denoise, robust deskew
- Auto gray/bw selection to avoid over-binarization
- Layout detection (optional)
- Table OCR (optional)
- Post-correction with Korean vocabulary
- Confidence filtering + retry
- Auto language switching (kor vs eng)
- Rescue pass for short-line/low-text pages (psm=3 grayscale)
- Optional TOC removal (front-matter cleanup)
- Skip low-text pages (covers/diagrams)
- Quality report (`quality_report.txt`)
- Output defaults to the input folder if not set

## Setup

1) Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

2) Install Python dependencies

```bash
pip install -r requirements.txt
```

3) Optional advanced OCR (layout + table OCR)

```bash
pip install -r requirements-advanced.txt
```

Install an appropriate `paddlepaddle` build for your OS using official PaddlePaddle docs.

4) Install system dependencies

- Tesseract OCR
- Korean language pack (`kor`)
- Poppler (for PDF to image conversion)

macOS (Homebrew):

```bash
brew install tesseract poppler
brew install tesseract-lang
```

Ubuntu:

```bash
sudo apt-get install tesseract-ocr tesseract-ocr-kor poppler-utils
```

## Run

```bash
python app.py
```

## Quality Report
The tool writes `quality_report.txt` in the output folder. It includes:
- Overall text cleanliness score
- Per-page score and stats
- OCR confidence, low-confidence counts, retries, fallbacks
- Language switches, rescues, skipped pages, TOC removal

## Tips to Improve OCR
- Increase DPI to `300–400`
- Use `kor` instead of `kor+eng` when possible
- Enable layout detection and table OCR
- Use `OCR only` when PDFs have broken text layers
- If OCR looks worse, set `Preprocess` to `gray`

## Notes
- If Tesseract is not on your PATH, set the Tesseract path in the UI.
- You can edit the `VOCAB` list in `app.py` to add domain-specific terms.
- If layout detection is ON but report shows `active=False`, install `paddlepaddle` (or Detectron2 for layoutparser).
