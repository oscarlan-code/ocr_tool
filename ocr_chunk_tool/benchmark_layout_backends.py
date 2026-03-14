import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from media_extract_batch import run_batched


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _summarize_payload(payload: Dict) -> Dict:
    items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    figures = sum(1 for item in items if str(item.get("type") or "") == "figure")
    tables = sum(1 for item in items if str(item.get("type") or "") == "table")
    pages = sorted(
        {
            int(item.get("source_pdf_page") or item.get("page_number") or item.get("page") or 0)
            for item in items
            if int(item.get("source_pdf_page") or item.get("page_number") or item.get("page") or 0) > 0
        }
    )
    return {
        "items_kept": len(items),
        "figure_count": figures,
        "table_count": tables,
        "pages": pages,
        "document_family": payload.get("document_family"),
        "layout_backend_requested": payload.get("layout_backend_requested"),
        "layout_backends_observed": payload.get("layout_backends_observed", []),
        "caption_candidates_detected": ((payload.get("stats") or {}).get("caption_candidates_detected")),
        "caption_anchor_items_generated": ((payload.get("stats") or {}).get("caption_anchor_items_generated")),
    }


def run_benchmark(
    pdf_path: str,
    out_dir: str,
    backends: List[str],
    pages: str,
    dpi: int,
    lang: str,
    use_text_layer_first: bool,
) -> str:
    pdf = os.path.abspath(pdf_path)
    root = Path(out_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    report = {
        "pdf_path": pdf,
        "pages": pages,
        "dpi": dpi,
        "lang": lang,
        "results": [],
    }

    for backend in backends:
        backend_out = root / backend
        backend_out.mkdir(parents=True, exist_ok=True)
        out_json = run_batched(
            pdf_path=pdf,
            out_dir=str(backend_out),
            page_spec=pages,
            dpi=dpi,
            lang=lang,
            use_text_layer_first=use_text_layer_first,
            layout_backend=backend,
        )
        payload = _load_json(out_json)
        report["results"].append(
            {
                "backend": backend,
                "output_json": out_json,
                "summary": _summarize_payload(payload),
            }
        )

    report_path = root / "layout-backend-benchmark.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    return str(report_path)


def main():
    parser = argparse.ArgumentParser(description="Benchmark OCR/media layout backends on the same PDF.")
    parser.add_argument("--pdf", required=True, help="Input PDF")
    parser.add_argument("--out", required=True, help="Benchmark output directory")
    parser.add_argument(
        "--backends",
        default="auto,paddle,layoutparser,deepdoctection",
        help="Comma-separated layout backends to run",
    )
    parser.add_argument("--pages", default="", help="Optional page spec")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--lang", default="eng")
    parser.add_argument(
        "--no-text-layer-first",
        action="store_true",
        help="Disable PDF text-layer extraction and force OCR for text/table regions",
    )
    args = parser.parse_args()

    backends = [item.strip().lower() for item in str(args.backends or "").split(",") if item.strip()]
    report_path = run_benchmark(
        pdf_path=args.pdf,
        out_dir=args.out,
        backends=backends or ["auto"],
        pages=args.pages,
        dpi=args.dpi,
        lang=args.lang,
        use_text_layer_first=not args.no_text_layer_first,
    )
    print(f"Benchmark report: {report_path}")


if __name__ == "__main__":
    main()
