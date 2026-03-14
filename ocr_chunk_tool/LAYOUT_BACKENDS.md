# Layout Backends

The OCR/media pipeline now supports an explicit preferred layout backend:

- `auto`
- `paddle`
- `layoutparser`
- `deepdoctection`

Default behavior is unchanged:

- `auto` keeps the current path: `paddle -> layoutparser`
- `deepdoctection` is opt-in for benchmarking and only works when the package is installed in the OCR tool environment

## Where It Applies

- `media_extract.py`
- `media_extract_batch.py`
- `media_api_server.py`

Each extraction payload now records:

- `layout_backend_requested`
- `layout_backends_observed`
- page-level `layout_backend`

## CLI Examples

Run batched media extraction with a preferred backend:

```bash
python ocr_chunk_tool/media_extract_batch.py \
  --pdf /path/to/document.pdf \
  --out /tmp/media-out \
  --layout-backend paddle
```

Benchmark multiple backends on the same PDF:

```bash
python ocr_chunk_tool/benchmark_layout_backends.py \
  --pdf /path/to/document.pdf \
  --out /tmp/layout-benchmark \
  --backends auto,paddle,layoutparser,deepdoctection
```

The benchmark writes:

- backend-specific extraction outputs under `/tmp/layout-benchmark/<backend>/`
- a summary report at `/tmp/layout-benchmark/layout-backend-benchmark.json`

## Recommended Use In This Repo

For LAIQ figure/table extraction:

1. Keep `auto` as the default for normal runs.
2. Benchmark `deepdoctection` on `academic_paper` and `manual` fixtures only.
3. Compare:
   - `items_kept`
   - `figure_count`
   - `table_count`
   - `caption_candidates_detected`
   - `caption_anchor_items_generated`
4. Only promote a new default if it improves recall without breaking caption anchoring.
