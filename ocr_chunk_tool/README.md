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
