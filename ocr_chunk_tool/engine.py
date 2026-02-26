import argparse
import json
import os
import sys
from datetime import datetime


def _detect_ocr_tool_dir():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(base_dir, "..")),
        os.path.abspath(os.path.join(base_dir, "..", "ocr tool")),
    ]
    for candidate in candidates:
        if os.path.isfile(os.path.join(candidate, "app.py")):
            return candidate
    return candidates[0]


OCR_TOOL_DIR = _detect_ocr_tool_dir()
if OCR_TOOL_DIR not in sys.path:
    sys.path.insert(0, OCR_TOOL_DIR)

try:
    from app import (
        _load_dotenv,
        _detect_poppler_path,
        extract_text_pdf_pages,
        ocr_pdf_pages,
        clean_output_text,
        llm_clean_text,
        scan_large_media,
        TEXT_MIN_CHARS,
        DEFAULT_LANG,
        DEFAULT_DPI,
        DEFAULT_TESS_OEM,
        DEFAULT_TESS_PSM_TEXT,
        DEFAULT_TESS_PSM_TABLE,
        DEFAULT_RETRY_PSM,
        DEFAULT_CONF_THRESHOLD,
        DEFAULT_PREPROCESS_MODE,
        DEFAULT_INK_RATIO_THRESHOLD,
        DEFAULT_HANGUL_RATIO,
        DEFAULT_LANG_SWITCH_MARGIN,
        DEFAULT_RESCUE_SHORT_RATIO,
        DEFAULT_RESCUE_PSM_TEXT,
        DEFAULT_RESCUE_PSM_TABLE,
        DEFAULT_REMOVE_TOC,
        DEFAULT_TOC_MAX_PAGES,
        DEFAULT_TOC_MIN_KEEP,
    )
except Exception as exc:
    raise RuntimeError(f"Failed to import OCR engine from {OCR_TOOL_DIR}: {exc}") from exc


def preprocess_pdf(
    pdf_path,
    out_dir,
    ocr_mode="auto",
    prefer_text_layer=True,
    lang=DEFAULT_LANG,
    dpi=DEFAULT_DPI,
    use_layout=True,
    use_table=True,
    use_vocab=True,
    preprocess_mode=DEFAULT_PREPROCESS_MODE,
    ink_ratio_threshold=float(DEFAULT_INK_RATIO_THRESHOLD),
    use_fallback=True,
    auto_lang=True,
    hangul_ratio_threshold=float(DEFAULT_HANGUL_RATIO),
    lang_switch_margin=int(DEFAULT_LANG_SWITCH_MARGIN),
    skip_low_text=True,
    min_chars=20,
    min_score=40,
    rescue_short_ratio=float(DEFAULT_RESCUE_SHORT_RATIO),
    rescue_psm_text=DEFAULT_RESCUE_PSM_TEXT,
    rescue_psm_table=DEFAULT_RESCUE_PSM_TABLE,
    remove_toc=DEFAULT_REMOVE_TOC,
    toc_max_pages=int(DEFAULT_TOC_MAX_PAGES),
    toc_min_keep=int(DEFAULT_TOC_MIN_KEEP),
    capture_media=True,
    summarize_tables=True,
    summarize_figures=True,
    deepseek_key="",
    deepseek_base="",
    deepseek_model="",
    clean_output=True,
    llm_clean=False,
    llm_skip_tables=True,
    auto_media_scan=True,
    media_scan_dpi=120,
    media_area_ratio=0.12,
    on_status=None,
    on_page=None,
):
    _load_dotenv()
    poppler_path = _detect_poppler_path()

    base = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    out_txt = os.path.join(out_dir, f"{base}.txt")
    out_clean = os.path.join(out_dir, f"{base}_clean.txt")
    out_llm = os.path.join(out_dir, f"{base}_llm_clean.txt")
    out_media_json = os.path.join(out_dir, f"{base}_figure_table.json")
    out_media_txt = os.path.join(out_dir, f"{base}_figure_table.txt")
    out_pages_json = os.path.join(out_dir, f"{base}_pages.json")

    pages = []
    text_layer_pages = []
    text_layer = ""
    media_items = []
    deepseek_calls = 0
    deepseek_errors = 0

    if on_status:
        on_status("Extracting text...")
    if prefer_text_layer or ocr_mode in ("auto", "text"):
        text_layer_pages = extract_text_pdf_pages(pdf_path, on_page=on_page)
        text_layer = "\n\n".join(text_layer_pages).strip()
    if ocr_mode in ("auto", "text"):
        pages = text_layer_pages
    text = text_layer if text_layer else "\n\n".join(pages).strip()

    should_ocr = (
        ocr_mode == "ocr"
        or (ocr_mode == "auto" and len(text) < TEXT_MIN_CHARS)
    )

    if not should_ocr and ocr_mode == "auto" and auto_media_scan and use_layout:
        if on_status:
            on_status("Scanning for large tables/figures...")
        if scan_large_media(
            pdf_path,
            poppler_path=poppler_path,
            dpi=media_scan_dpi,
            min_area_ratio=media_area_ratio,
        ):
            should_ocr = True

    if should_ocr:
        if on_status:
            on_status("Running OCR...")
        (
            pages,
            _page_confs,
            _page_lows,
            _page_retries,
            _page_fallbacks,
            _page_lang_switches,
            _page_stats_raw,
            _page_skipped,
            _page_rescues,
            _page_rescue_reasons,
            _page_toc_removed,
            media_items,
            deepseek_calls,
            deepseek_errors,
        ) = ocr_pdf_pages(
            pdf_path,
            lang=lang,
            dpi=dpi,
            oem=DEFAULT_TESS_OEM,
            psm_text=DEFAULT_TESS_PSM_TEXT,
            psm_table=DEFAULT_TESS_PSM_TABLE,
            conf_threshold=int(DEFAULT_CONF_THRESHOLD),
            retry_psm=DEFAULT_RETRY_PSM,
            use_layout=use_layout,
            use_table_ocr=use_table,
            use_vocab=use_vocab,
            preprocess_mode=preprocess_mode,
            ink_ratio_threshold=ink_ratio_threshold,
            use_fallback=use_fallback,
            auto_lang=auto_lang,
            hangul_ratio_threshold=hangul_ratio_threshold,
            lang_switch_margin=lang_switch_margin,
            skip_low_text=skip_low_text,
            min_chars=min_chars,
            min_score=min_score,
            rescue_short_ratio=rescue_short_ratio,
            rescue_psm_text=rescue_psm_text,
            rescue_psm_table=rescue_psm_table,
            remove_toc=remove_toc,
            toc_max_pages=toc_max_pages,
            toc_min_keep=toc_min_keep,
            capture_media=capture_media,
            media_dir=os.path.join(out_dir, f"{base}_media"),
            deepseek_key=deepseek_key,
            deepseek_base=deepseek_base,
            deepseek_model=deepseek_model,
            summarize_tables=summarize_tables and bool(deepseek_key),
            summarize_figures=summarize_figures and bool(deepseek_key),
            mem_cleanup=True,
            poppler_path=poppler_path,
            on_page=on_page,
        )
        text = "\n\n".join(pages).strip()
        if prefer_text_layer and len(text_layer) >= TEXT_MIN_CHARS:
            text = text_layer

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    clean_text = ""
    if clean_output and text:
        if on_status:
            on_status("Cleaning text...")
        clean_text = clean_output_text(text)
        with open(out_clean, "w", encoding="utf-8") as f:
            f.write(clean_text)

    llm_text = ""
    if llm_clean and clean_text and deepseek_key:
        if on_status:
            on_status("LLM cleaning...")
        llm_text, _, _ = llm_clean_text(
            clean_text,
            deepseek_key,
            deepseek_base,
            deepseek_model,
            skip_tables_figures=llm_skip_tables,
        )
        if llm_text:
            with open(out_llm, "w", encoding="utf-8") as f:
                f.write(llm_text)

    if pages:
        if on_status:
            on_status("Saving page data...")
        with open(out_pages_json, "w", encoding="utf-8") as f:
            json.dump(
                {"file": os.path.basename(pdf_path), "source_path": pdf_path, "pages": pages},
                f,
                ensure_ascii=False,
                indent=2,
            )

    if media_items:
        if on_status:
            on_status("Saving figure/table data...")
        with open(out_media_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "file": os.path.basename(pdf_path),
                    "source_path": pdf_path,
                    "items": media_items,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        with open(out_media_txt, "w", encoding="utf-8") as f:
            for item in media_items:
                f.write(
                    f"Page {item.get('page', 0):03d} "
                    f"[{item.get('type', '')}] "
                    f"bbox={item.get('bbox', [])} "
                    f"image={item.get('image_path', '')}\n"
                )
                title = (item.get("title") or "").strip()
                if title:
                    f.write("Title:\n" + title + "\n")
                summary = (item.get("summary") or "").strip()
                if summary:
                    f.write(summary + "\n")
                context = (item.get("context") or "").strip()
                if context:
                    f.write("Context:\n" + context + "\n")
                ocr_text = (item.get("ocr_text") or "").strip()
                if ocr_text:
                    f.write("OCR:\n" + ocr_text + "\n")
                f.write("\n")

    if on_status:
        on_status("Writing manifest...")
    manifest = {
        "source_pdf": pdf_path,
        "output_dir": out_dir,
        "text_file": out_txt,
        "clean_text_file": out_clean if clean_text else "",
        "llm_clean_text_file": out_llm if llm_text else "",
        "pages_file": out_pages_json if pages else "",
        "media_json": out_media_json if media_items else "",
        "media_txt": out_media_txt if media_items else "",
        "media_dir": os.path.join(out_dir, f"{base}_media") if media_items else "",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    manifest_path = os.path.join(out_dir, f"{base}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="OCR preprocess engine")
    parser.add_argument("--pdf", required=True, help="PDF file to process")
    parser.add_argument("--out", default="", help="Output directory (default: same as PDF)")
    parser.add_argument("--ocr-mode", default="auto", choices=["auto", "ocr", "text"])
    parser.add_argument("--lang", default=DEFAULT_LANG)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--chunk-input", default="clean", choices=["raw", "clean", "llm_clean"])
    parser.add_argument("--no-clean", action="store_true")
    parser.add_argument("--llm-clean", action="store_true")
    parser.add_argument("--llm-skip-tables", action="store_true")
    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--no-layout", action="store_true")
    parser.add_argument("--no-table", action="store_true")
    parser.add_argument("--media-scan-dpi", type=int, default=120)
    parser.add_argument("--media-area-ratio", type=float, default=0.12)
    args = parser.parse_args()

    pdf_path = os.path.abspath(args.pdf)
    out_dir = os.path.abspath(args.out) if args.out else os.path.dirname(pdf_path)

    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or os.environ.get("DEEPSEEK_KEY", "").strip()
    deepseek_base = os.environ.get("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1/chat/completions"
    deepseek_model = os.environ.get("DEEPSEEK_MODEL", "").strip() or "deepseek-chat"

    manifest = preprocess_pdf(
        pdf_path=pdf_path,
        out_dir=out_dir,
        ocr_mode=args.ocr_mode,
        lang=args.lang,
        dpi=args.dpi,
        use_layout=not args.no_layout,
        use_table=not args.no_table,
        clean_output=not args.no_clean,
        llm_clean=args.llm_clean,
        llm_skip_tables=args.llm_skip_tables,
        capture_media=not args.no_media,
        deepseek_key=deepseek_key,
        deepseek_base=deepseek_base,
        deepseek_model=deepseek_model,
        auto_media_scan=True,
        media_scan_dpi=max(args.media_scan_dpi, 80),
        media_area_ratio=args.media_area_ratio,
    )
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
