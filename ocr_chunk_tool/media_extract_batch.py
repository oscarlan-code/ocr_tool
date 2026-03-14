import argparse
import json
import os
import shutil
from typing import List

from document_family import classify_document_family
from media_extract import _ensure_pdf_page_count, _parse_page_spec, _write_media_text, _detect_poppler_path, extract_media


def _chunk_pages(pages: List[int], batch_size: int) -> List[List[int]]:
    return [pages[idx : idx + batch_size] for idx in range(0, len(pages), batch_size)]


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_merged_payload(
    pdf_path: str,
    dpi: int,
    pages_requested: List[int],
    items: List[dict],
    filtered_out: List[dict],
    batch_results: List[dict],
    page_analyses: List[dict],
    layout_backend: str,
) -> dict:
    family_info = classify_document_family(
        [str(page.get("text_excerpt") or "") for page in page_analyses],
        source_name=os.path.basename(pdf_path),
    )
    observed_layout_backends = sorted(
        {
            str(page.get("layout_backend") or "").strip()
            for page in page_analyses
            if str(page.get("layout_backend") or "").strip()
        }
    )
    for collection in (items, filtered_out):
        for item in collection:
            item["document_family"] = family_info.get("family", "scanned_unknown")
            item["document_family_confidence"] = family_info.get("confidence", 0.0)
    return {
        "file": os.path.basename(pdf_path),
        "source_path": pdf_path,
        "dpi": dpi,
        "pages_requested": pages_requested,
        "document_family": family_info.get("family", "scanned_unknown"),
        "document_family_confidence": family_info.get("confidence", 0.0),
        "document_family_scores": family_info.get("scores", {}),
        "document_family_matched_terms": family_info.get("matched_terms", {}),
        "layout_backend_requested": layout_backend,
        "layout_backends_observed": observed_layout_backends,
        "page_analyses": page_analyses,
        "batch_results": batch_results,
        "stats": {
            "items_kept": len(items),
            "items_filtered_out": len(filtered_out),
            "items_detected_total": len(items) + len(filtered_out),
            "batches_completed": len(batch_results),
            "caption_candidates_detected": sum(
                int(page.get("caption_count", 0) or 0) for page in page_analyses
            ),
            "layout_backends_observed": observed_layout_backends,
        },
        "items": items,
        "filtered_out": filtered_out,
    }


def _copy_media_file(src: str, merged_media_dir: str) -> str:
    if not src or not os.path.isfile(src):
        return src
    os.makedirs(merged_media_dir, exist_ok=True)
    dest = os.path.join(merged_media_dir, os.path.basename(src))
    if os.path.abspath(src) != os.path.abspath(dest):
        shutil.copy2(src, dest)
    return dest


def run_batched(
    pdf_path: str,
    out_dir: str,
    page_spec: str = "",
    batch_size: int = 6,
    dpi: int = 170,
    lang: str = "eng",
    include_figure_ocr: bool = False,
    use_text_layer_first: bool = True,
    layout_backend: str = "auto",
) -> str:
    poppler_path = _detect_poppler_path()
    total_pages = _ensure_pdf_page_count(pdf_path, poppler_path)
    if total_pages <= 0:
        raise RuntimeError("Unable to determine PDF page count.")

    target_pages = _parse_page_spec(page_spec, total_pages)
    if not target_pages:
        raise RuntimeError("No valid pages selected.")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    merged_media_dir = os.path.join(out_dir, f"{base}_media")
    os.makedirs(merged_media_dir, exist_ok=True)

    merged_items = []
    merged_filtered = []
    batch_results = []
    merged_page_analyses = []
    out_json = os.path.join(out_dir, f"{base}_media_only.json")
    out_txt = os.path.join(out_dir, f"{base}_media_only.txt")

    for batch_idx, page_batch in enumerate(_chunk_pages(target_pages, batch_size), start=1):
        batch_label = f"batch_{page_batch[0]:03d}_{page_batch[-1]:03d}"
        batch_out = os.path.join(out_dir, batch_label)
        batch_json = extract_media(
            pdf_path=pdf_path,
            out_dir=batch_out,
            page_spec=",".join(str(page) for page in page_batch),
            dpi=dpi,
            lang=lang,
            include_figure_ocr=include_figure_ocr,
            use_text_layer_first=use_text_layer_first,
            layout_backend=layout_backend,
        )
        batch_payload = _load_json(batch_json)

        for item in batch_payload.get("items", []):
            item_copy = dict(item)
            item_copy["image_path"] = _copy_media_file(item_copy.get("image_path", ""), merged_media_dir)
            merged_items.append(item_copy)

        for item in batch_payload.get("filtered_out", []):
            item_copy = dict(item)
            # filtered-out images are usually deleted already, but normalize the path if present
            if item_copy.get("image_path"):
                item_copy["image_path"] = os.path.join(merged_media_dir, os.path.basename(item_copy["image_path"]))
            merged_filtered.append(item_copy)

        merged_page_analyses.extend(
            page for page in batch_payload.get("page_analyses", []) if isinstance(page, dict)
        )

        batch_results.append(
            {
                "batch_index": batch_idx,
                "pages": page_batch,
                "output_dir": batch_out,
                "items_kept": len(batch_payload.get("items", [])),
                "items_filtered_out": len(batch_payload.get("filtered_out", [])),
                "layout_backends_observed": batch_payload.get("layout_backends_observed", []),
            }
        )

        merged_payload = _build_merged_payload(
            pdf_path=pdf_path,
            dpi=dpi,
            pages_requested=target_pages,
            items=merged_items,
            filtered_out=merged_filtered,
            batch_results=batch_results,
            page_analyses=sorted(
                merged_page_analyses,
                key=lambda page: int(page.get("page_number", 0) or 0),
            ),
            layout_backend=layout_backend,
        )
        with open(out_json, "w", encoding="utf-8") as handle:
            json.dump(merged_payload, handle, ensure_ascii=False, indent=2)
        _write_media_text(out_txt, merged_items)

    return out_json


def main():
    parser = argparse.ArgumentParser(description="Run lightweight media extraction in page batches and merge results.")
    parser.add_argument("--pdf", required=True, help="PDF file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--pages",
        default="",
        help="1-based page list/ranges from the source PDF, e.g. 1-24 or 5,10,14-17",
    )
    parser.add_argument("--batch-size", type=int, default=6, help="Number of source PDF pages per batch")
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--lang", default="eng")
    parser.add_argument("--figure-ocr", action="store_true", help="Run OCR on figure regions")
    parser.add_argument(
        "--layout-backend",
        default="auto",
        choices=["auto", "paddle", "layoutparser", "deepdoctection"],
        help="Preferred layout detector backend. Defaults to the current auto path.",
    )
    parser.add_argument(
        "--no-text-layer-first",
        action="store_true",
        help="Disable PDF text-layer extraction and force OCR for text/table region content",
    )
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    out_dir = os.path.abspath(args.out)
    out_json = run_batched(
        pdf_path=pdf_path,
        out_dir=out_dir,
        page_spec=args.pages,
        batch_size=max(1, int(args.batch_size)),
        dpi=args.dpi,
        lang=args.lang,
        include_figure_ocr=args.figure_ocr,
        use_text_layer_first=not args.no_text_layer_first,
        layout_backend=args.layout_backend,
    )
    print(f"Merged Media JSON: {out_json}")


if __name__ == "__main__":
    main()
