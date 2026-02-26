import argparse
import json
import os
import shutil
import sys
import re

from typing import List

try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
except Exception:
    Document = None
    SemanticSplitterNodeParser = None
    SentenceSplitter = None

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception:
    HuggingFaceEmbedding = None

try:
    import requests
except Exception:
    requests = None

# Load .env from OCR tool if available
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
    from app import _load_dotenv
except Exception:
    _load_dotenv = None


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_text_file(path):
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pages(manifest):
    pages_file = manifest.get("pages_file") or ""
    if not pages_file or not os.path.isfile(pages_file):
        return []
    with open(pages_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("pages") or []


def _load_media(manifest):
    media_file = manifest.get("media_json") or ""
    if not media_file or not os.path.isfile(media_file):
        return []
    with open(media_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("items") or []


def _load_deepseek_config():
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or os.environ.get("DEEPSEEK_KEY", "").strip()
    base = os.environ.get("DEEPSEEK_BASE_URL", "").strip() or "https://api.deepseek.com/v1/chat/completions"
    model = os.environ.get("DEEPSEEK_MODEL", "").strip() or "deepseek-chat"
    return key, base, model


_URL_RE = re.compile(r"https?://\\S+|www\\.\\S+", re.IGNORECASE)
_PRICE_RE = re.compile(r"\\$\\s*\\d+(?:\\.\\d+)?")
_QTY_RE = re.compile(r"\\b\\d+\\s*(?:per\\s*pack|pack|pcs?|pairs?|set|kg|g|lb|oz|pc)\\b", re.IGNORECASE)
_UI_NOISE_RE = re.compile(
    r"\\b(add to cart|checkout|continue shopping|cart|help|faq|contact|download|privacy|sitemap|about us|payment|"
    r"your order summary|purchase with purchase|promo item|unlock|promo items)\\b",
    re.IGNORECASE,
)


def _normalize_lines(text):
    lines = []
    for raw in (text or "").splitlines():
        line = " ".join(raw.strip().split())
        if line:
            lines.append(line)
    return lines


def _is_noise_line(line):
    if not line:
        return True
    if _URL_RE.search(line):
        return True
    if _UI_NOISE_RE.search(line):
        return True
    # mostly symbols/glyphs
    letters = sum(ch.isalnum() for ch in line)
    if letters == 0 and len(line) <= 6:
        return True
    # tiny fragments
    if len(line) <= 2:
        return True
    return False


def _filter_lines(lines):
    return [ln for ln in lines if not _is_noise_line(ln)]


def _extract_table_rows(lines):
    rows = []
    for i, line in enumerate(lines):
        if not _PRICE_RE.search(line):
            continue
        parts = [line]
        if i > 0 and not _PRICE_RE.search(lines[i - 1]) and not _is_noise_line(lines[i - 1]):
            parts.insert(0, lines[i - 1])
        if i + 1 < len(lines) and (_QTY_RE.search(lines[i + 1]) or "per pack" in lines[i + 1].lower()):
            parts.append(lines[i + 1])
        row = " ".join(p.strip() for p in parts if p.strip())
        if row and row not in rows:
            rows.append(row)
    return rows


def _build_media_prompt(item):
    kind = item.get("type", "figure")
    title = (item.get("title") or "").strip()
    context = (item.get("context") or "").strip()
    ocr_text = (item.get("ocr_text") or "").strip()
    parts = []
    if title:
        parts.append(f"Title: {title}")

    if kind == "table":
        base_text = ocr_text or context
        lines = _filter_lines(_normalize_lines(base_text))
        rows = _extract_table_rows(lines)
        if rows:
            parts.append("Table rows (approx):\n" + "\n".join(rows))
        elif lines:
            parts.append("Table text (filtered):\n" + "\n".join(lines))
    else:
        # figure: keep only filtered text to avoid UI noise
        if context:
            ctx_lines = _filter_lines(_normalize_lines(context))
            if ctx_lines:
                parts.append("Context:\n" + "\n".join(ctx_lines))
        if ocr_text:
            ocr_lines = _filter_lines(_normalize_lines(ocr_text))
            if ocr_lines:
                parts.append("OCR text:\n" + "\n".join(ocr_lines))

    body = "\n\n".join(parts).strip()
    if not body:
        return ""
    if kind == "table":
        return (
            "Summarize the table as a compact item list. "
            "Ignore UI/navigation text. Output bullets like: Item — Price — Pack/Qty (if present). "
            "Do not mention missing data.\n\n"
            f"{body}"
        )
    return (
        f"Summarize the {kind} based on the provided context and OCR text. "
        "Keep it short and factual. Ignore UI/navigation text.\n\n"
        f"{body}"
    )


def _deepseek_summarize(prompt, api_key, base_url, model, timeout=30):
    if not prompt or not api_key or not base_url or not model or requests is None:
        return "", "missing_config"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a technical documentation assistant. Summarize precisely."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        resp = requests.post(base_url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content, ""
    except Exception as exc:
        return "", str(exc)


def _build_chunk_refine_prompt(text):
    trimmed = (text or "").strip()
    if len(trimmed) > 3500:
        trimmed = trimmed[:3500] + "\n...[truncated]"
    return (
        "You are cleaning OCR chunks from a technical manual.\n"
        "Return STRICT JSON with keys: keep (true/false), reason (string), cleaned_text (string).\n"
        "Rules:\n"
        "- Do NOT add new information.\n"
        "- Fix OCR errors, spacing, and remove obvious noise only.\n"
        "- If the chunk is meaningless or mostly noise, set keep=false and cleaned_text=\"\".\n"
        "- If keep=true, cleaned_text must preserve all numbers/units.\n\n"
        f"Chunk:\n{trimmed}"
    )


def _parse_refine_response(text):
    if not text:
        return None
    text = text.strip()
    # Try to extract JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _extract_numbers(text):
    if not text:
        return []
    # Capture integers and decimals, including comma separators
    return re.findall(r"\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?", text)


def _numbers_match(original, cleaned):
    orig_nums = _extract_numbers(original)
    clean_nums = _extract_numbers(cleaned)
    return orig_nums == clean_nums


def _get_embeddings(model_name: str):
    if HuggingFaceEmbedding is None:
        return None
    return HuggingFaceEmbedding(model_name=model_name)


def _semantic_nodes(
    text: str,
    page_num=None,
    embed_model=None,
    buffer_size=1,
    breakpoint_percentile_threshold=95,
):
    if SemanticSplitterNodeParser is None or Document is None:
        return [Document(text=text, metadata={"page": page_num})] if Document else []
    if embed_model is None:
        embed_model = _get_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    if embed_model is None:
        return [Document(text=text, metadata={"page": page_num})] if Document else []
    parser = SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=buffer_size,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
    )
    doc = Document(text=text, metadata={"page": page_num})
    nodes = parser.get_nodes_from_documents([doc])
    return nodes


def _enforce_chunk_size(nodes, chunk_size: int, chunk_overlap: int) -> List[dict]:
    if not nodes:
        return []
    if not chunk_size or chunk_size <= 0:
        return [{"text": n.get_content(), "metadata": n.metadata} for n in nodes]
    if SentenceSplitter is None or Document is None:
        return [{"text": n.get_content(), "metadata": n.metadata} for n in nodes]

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    out = []
    for node in nodes:
        text = node.get_content()
        if len(text) <= chunk_size:
            out.append({"text": text, "metadata": dict(node.metadata)})
        else:
            sub_docs = splitter.get_nodes_from_documents(
                [Document(text=text, metadata=dict(node.metadata))]
            )
            for sub in sub_docs:
                out.append({"text": sub.get_content(), "metadata": sub.metadata})
    return out


def build_chunks(
    manifest_path,
    chunk_size=1000,
    chunk_overlap=100,
    text_preference="clean",
    semantic_buffer=1,
    semantic_threshold=95,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    copy_media=True,
    llm_media_summary=False,
    llm_chunk_refine=False,
    llm_drop=False,
    llm_min_chars=200,
):
    if _load_dotenv is not None:
        _load_dotenv()
        key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or os.environ.get("DEEPSEEK_KEY", "").strip()
        if key:
            print("LLM loaded: yes")
        else:
            print("LLM loaded: no (missing DEEPSEEK_API_KEY)")
    manifest = _load_manifest(manifest_path)
    pages = _load_pages(manifest)
    media = _load_media(manifest)

    if text_preference == "llm_clean":
        text = _load_text_file(manifest.get("llm_clean_text_file", ""))
    elif text_preference == "raw":
        text = _load_text_file(manifest.get("text_file", ""))
    else:
        text = _load_text_file(manifest.get("clean_text_file", "")) or _load_text_file(manifest.get("text_file", ""))

    media_by_page = {}
    for item in media:
        page = item.get("pdf_page") or item.get("page")
        if not page:
            continue
        media_by_page.setdefault(int(page), []).append(item)

    chunks = []
    chunk_id = 1

    embed_model = _get_embeddings(embedding_model)

    # Prepare chunk media output directory
    out_dir = os.path.dirname(manifest_path)
    chunk_media_dir = os.path.join(out_dir, "chunk_media")
    if copy_media:
        os.makedirs(chunk_media_dir, exist_ok=True)

    ds_key = ""
    ds_base = ""
    ds_model = ""
    if llm_media_summary:
        ds_key, ds_base, ds_model = _load_deepseek_config()

    def _attach_media(page, chunk_id):
        items = []
        for item in media_by_page.get(int(page), []):
            new_item = dict(item)
            img_path = new_item.get("image_path") or ""
            if copy_media and img_path and os.path.isfile(img_path):
                ext = os.path.splitext(img_path)[1] or ".png"
                local_name = f"{chunk_id}_{len(items)+1:02d}{ext}"
                local_path = os.path.join(chunk_media_dir, local_name)
                if not os.path.isfile(local_path):
                    try:
                        shutil.copy2(img_path, local_path)
                    except Exception:
                        local_path = ""
                new_item["image_local_path"] = local_path
            if llm_media_summary and not new_item.get("summary_contextual"):
                prompt = _build_media_prompt(new_item)
                if prompt and ds_key:
                    summary, err = _deepseek_summarize(prompt, ds_key, ds_base, ds_model)
                    if not err and summary:
                        new_item["summary_contextual"] = summary
            items.append(new_item)
        return items

    if pages:
        for idx, page_text in enumerate(pages, start=1):
            if not page_text.strip():
                continue
            nodes = _semantic_nodes(
                page_text,
                page_num=idx,
                embed_model=embed_model,
                buffer_size=semantic_buffer,
                breakpoint_percentile_threshold=semantic_threshold,
            )
            chunk_entries = _enforce_chunk_size(nodes, chunk_size, chunk_overlap)
            for entry in chunk_entries:
                page = entry["metadata"].get("page", idx)
                media_items = _attach_media(page, f"chunk_{chunk_id:06d}")
                chunk_record = {
                    "id": f"chunk_{chunk_id:06d}",
                    "page": page,
                    "text": entry["text"].strip(),
                    "media": media_items,
                }

                if llm_chunk_refine and len(chunk_record["text"]) >= llm_min_chars:
                    ds_key, ds_base, ds_model = _load_deepseek_config()
                    if ds_key:
                        prompt = _build_chunk_refine_prompt(chunk_record["text"])
                        resp, err = _deepseek_summarize(prompt, ds_key, ds_base, ds_model)
                        data = _parse_refine_response(resp) if not err else None
                        if data:
                            chunk_record["llm_keep"] = bool(data.get("keep", True))
                            chunk_record["llm_reason"] = str(data.get("reason", "")).strip()
                            llm_clean = str(data.get("cleaned_text", "")).strip()
                            if llm_clean and not _numbers_match(chunk_record["text"], llm_clean):
                                chunk_record["llm_keep"] = True
                                chunk_record["llm_reason"] = "number_changed"
                                chunk_record["llm_clean_text"] = ""
                            else:
                                chunk_record["llm_clean_text"] = llm_clean
                        else:
                            chunk_record["llm_keep"] = True
                            chunk_record["llm_reason"] = "parse_failed"
                            chunk_record["llm_clean_text"] = ""
                if llm_chunk_refine and llm_drop and chunk_record.get("llm_keep") is False:
                    chunk_id += 1
                    continue

                chunks.append(chunk_record)
                chunk_id += 1
    else:
        nodes = _semantic_nodes(
            text,
            page_num=None,
            embed_model=embed_model,
            buffer_size=semantic_buffer,
            breakpoint_percentile_threshold=semantic_threshold,
        )
        chunk_entries = _enforce_chunk_size(nodes, chunk_size, chunk_overlap)
        for entry in chunk_entries:
            chunk_record = {
                "id": f"chunk_{chunk_id:06d}",
                "page": entry["metadata"].get("page"),
                "text": entry["text"].strip(),
                "media": [],
            }
            if llm_chunk_refine and len(chunk_record["text"]) >= llm_min_chars:
                ds_key, ds_base, ds_model = _load_deepseek_config()
                if ds_key:
                    prompt = _build_chunk_refine_prompt(chunk_record["text"])
                    resp, err = _deepseek_summarize(prompt, ds_key, ds_base, ds_model)
                    data = _parse_refine_response(resp) if not err else None
                    if data:
                        chunk_record["llm_keep"] = bool(data.get("keep", True))
                        chunk_record["llm_reason"] = str(data.get("reason", "")).strip()
                        llm_clean = str(data.get("cleaned_text", "")).strip()
                        if llm_clean and not _numbers_match(chunk_record["text"], llm_clean):
                            chunk_record["llm_keep"] = True
                            chunk_record["llm_reason"] = "number_changed"
                            chunk_record["llm_clean_text"] = ""
                        else:
                            chunk_record["llm_clean_text"] = llm_clean
                    else:
                        chunk_record["llm_keep"] = True
                        chunk_record["llm_reason"] = "parse_failed"
                        chunk_record["llm_clean_text"] = ""
            if llm_chunk_refine and llm_drop and chunk_record.get("llm_keep") is False:
                chunk_id += 1
                continue
            chunks.append(chunk_record)
            chunk_id += 1

    model_tag = embedding_model.split("/")[-1].replace(" ", "_")
    out_path = os.path.join(
        out_dir,
        f"chunks_sem{semantic_threshold}_size{chunk_size}_over{chunk_overlap}_{model_tag}.json",
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "manifest": manifest_path,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "semantic_threshold": semantic_threshold,
                "semantic_buffer": semantic_buffer,
                "embedding_model": embedding_model,
                "chunk_media_dir": chunk_media_dir if copy_media else "",
                "text_preference": text_preference,
                "chunks": chunks,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Semantic chunk tool")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--text", default="clean", choices=["raw", "clean", "llm_clean"])
    parser.add_argument("--semantic-buffer", type=int, default=1)
    parser.add_argument("--semantic-threshold", type=int, default=95)
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--no-copy-media", action="store_true")
    parser.add_argument("--llm-media-summary", action="store_true")
    parser.add_argument("--llm-chunk-refine", action="store_true")
    parser.add_argument("--llm-drop", action="store_true")
    parser.add_argument("--llm-min-chars", type=int, default=200)
    args = parser.parse_args()

    out_path = build_chunks(
        args.manifest,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        text_preference=args.text,
        semantic_buffer=args.semantic_buffer,
        semantic_threshold=args.semantic_threshold,
        embedding_model=args.embedding_model,
        copy_media=not args.no_copy_media,
        llm_media_summary=args.llm_media_summary,
        llm_chunk_refine=args.llm_chunk_refine,
        llm_drop=args.llm_drop,
        llm_min_chars=args.llm_min_chars,
    )
    print(f"Chunks saved: {out_path}")


if __name__ == "__main__":
    main()
