# OCR Chunk Tool

This project provides:

1. A **preprocess engine** that wraps the existing OCR tool and produces
   structured outputs.
2. A **chunk tool** that performs semantic chunking (LlamaIndex) with a
   configurable chunk size and embedding model selection.
3. A **preview tool** that lets you review chunks and linked figures/tables.

## Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Preprocess (Engine)

```bash
python engine.py --pdf "/path/to/file.pdf" --out "/path/to/output"
```

Outputs:

- `<base>.txt`
- `<base>_clean.txt`
- `<base>_llm_clean.txt` (if enabled)
- `<base>_figure_table.json`
- `<base>_figure_table.txt`
- `manifest.json`

## Chunk

```bash
python chunk_tool.py --manifest "/path/to/manifest.json" --chunk-size 1000 --chunk-overlap 100 \
  --semantic-threshold 95 --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --llm-media-summary --llm-chunk-refine
```

Outputs:

- `chunks.json`

## Preview

```bash
python preview_tool.py --manifest "/path/to/manifest.json" --chunks "/path/to/chunks.json"
```

## Lightweight Media Extract

Use this when you only want figure/table regions plus sidecar JSON, without the full OCR/chunk pipeline:

```bash
python media_extract.py --pdf "/path/to/file.pdf" --out "/path/to/output" --pages "5,10,14,17"
```

Notes:
- Default DPI is `170` for faster processing.
- Figure OCR is off by default.
- The extractor prefers the PDF text layer for text/table content, then falls back to OCR only when needed.

Outputs:

- `<base>_media_only.json`
- `<base>_media_only.txt`
- `<base>_media/`

## Batched Full-PDF Media Extract

Use this for a whole document without a single long-running monolithic pass:

```bash
python media_extract_batch.py --pdf "/path/to/file.pdf" --out "/path/to/output" --pages "1-24" --batch-size 6
```

This runs page windows sequentially in one long-lived Python process and writes merged output after each batch.

Outputs:

- `<base>_media_only.json` (merged, incrementally updated)
- `<base>_media_only.txt` (merged)
- `<base>_media/` (merged crops)
- `batch_*` subdirectories with per-batch intermediate outputs

## Preview (with Dynamic Chunk Size)

Open the preview tool and use **Rechunk** to adjust:
- chunk size
- overlap
- semantic threshold
- embedding model
- text source (raw/clean/llm_clean)

## Full UI (Preprocess + Chunk + Preview)

```bash
python full_ui.py
```
```
