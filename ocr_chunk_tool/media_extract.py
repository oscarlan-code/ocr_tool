import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import numpy as np
except Exception:
    np = None

from caption_parser import build_citation_aliases, extract_caption_candidates, match_caption_to_block
from document_family import classify_document_family, classify_page_family

try:
    import pdfplumber
except Exception:
    pdfplumber = None


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
        convert_from_path,
        pdfinfo_from_path,
        preprocess_image,
        detect_layout_blocks_with_backend,
        normalize_layout_backend,
        crop_image,
        _collect_context,
        _extract_caption_title,
        ocr_tesseract,
        ocr_table_paddle,
        DEFAULT_TESS_OEM,
        DEFAULT_TESS_PSM_TEXT,
        DEFAULT_TESS_PSM_TABLE,
        DEFAULT_CONF_THRESHOLD,
        Block,
    )
except Exception as exc:
    raise RuntimeError(f"Failed to import OCR functions from {OCR_TOOL_DIR}: {exc}") from exc


CAPTION_RE = re.compile(r"^(table|fig\.?|figure|diagram|wiring diagram|component and control locations)\b", re.IGNORECASE)
FIGURE_KEYWORDS = (
    "figure",
    "diagram",
    "wiring",
    "component",
    "location",
    "locations",
    "controls & features",
    "component and control",
    "panel",
)
LOW_VALUE_TITLE_KEYWORDS = (
    "owner's manual",
    "owners manual",
    "owner’s manual",
    "product registration",
    "register your unit",
    "customer information",
    "dealer locator",
    "warranty",
    "parts",
)
LOW_VALUE_CONTEXT_KEYWORDS = (
    "powerequipment.honda.com",
    "register your unit",
    "customer information",
    "dealer locator",
    "warranty",
    "replacement parts",
    "authorized honda servicing dealer",
)
TABLE_KEYWORDS = (
    "table",
    "schedule",
    "maintenance schedule",
    "troubleshooting",
    "possible cause",
    "correction",
)
GENERIC_TABLE_HEADER_LINES = {
    "possible",
    "cause",
    "correction",
    "possible cause",
    "item",
    "action",
    "see page",
    "page",
    "hours",
    "hour",
}
TABLE_SECTION_HINTS = (
    "start",
    "power",
    "troubleshooting",
    "maintenance",
    "schedule",
    "overload",
    "output",
    "engine",
)
CANONICAL_HEADINGS = (
    "Component and Control Locations",
    "Controls & Features",
    "Check Your Generator",
    "Maintenance Schedule",
    "Troubleshooting",
    "Engine Will Not Start",
    "Loss of Power",
    "No Power at the AC Receptacles",
    "Wiring Diagram",
)
HIGH_VALUE_MEDIA_TITLES = {
    "Safety Label Locations",
    "Use Information",
    "USE INFORMATION",
    "Component and Control Locations",
    "Controls & Features",
    "Check Your Generator",
    "THROTTLE® ECO",
    "OIL ALERT",
    "Starting the Engine",
    "Maintenance Schedule",
    "Troubleshooting",
    "Engine Will Not Start",
    "Loss of Power",
    "No Power at the AC Receptacles",
    "Specifications",
    "Wiring Diagram",
}
IMPERATIVE_START_RE = re.compile(
    r"^(check|turn|make|look|add|remove|wipe|stop|restart|refuel|clean|replace|adjust)\b",
    re.IGNORECASE,
)
SUBFIGURE_MARKER_WORD_RE = re.compile(r"^\(?([a-z])\)?$", re.IGNORECASE)


def _parse_page_spec(spec: str, total_pages: int) -> List[int]:
    if not spec:
        return list(range(1, total_pages + 1))

    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            if end < start:
                start, end = end, start
            out.extend(range(start, end + 1))
        else:
            out.append(int(part))

    deduped = []
    seen = set()
    for page in out:
        if page < 1 or page > total_pages:
            continue
        if page in seen:
            continue
        seen.add(page)
        deduped.append(page)
    return deduped


def _ensure_pdf_page_count(pdf_path: str, poppler_path: Optional[str]) -> int:
    info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
    return int(info.get("Pages", 0))


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _slugify_token(text: str) -> str:
    token = re.sub(r"[^a-z0-9]+", "-", _normalize_text(text).lower())
    return token.strip("-")


def _build_media_object_id(kind: str, page_number: int, file_name: str, label: str = "") -> str:
    label_part = _slugify_token(label)
    file_part = _slugify_token(os.path.splitext(os.path.basename(file_name))[0])
    suffix = label_part or file_part or f"{kind}-{page_number}"
    return f"page{int(page_number):03d}-{kind}-{suffix}"


def _normalize_heading_spelling(text: str) -> str:
    candidate = _normalize_text(text)
    if not candidate:
        return ""
    candidate = candidate.replace("VV", "W")
    candidate = candidate.replace("”", "").replace("“", "")
    replacements = {
        "Enaine": "Engine",
        "VVill": "Will",
        "VVi": "Wi",
        "ot": "of",
        "Output": "Output",
        "Fower": "Power",
        "Haintenance": "Maintenance",
        "Scheduie": "Schedule",
    }
    words = []
    for word in candidate.split():
        cleaned = word.strip()
        replacement = replacements.get(cleaned, None)
        words.append(replacement if replacement is not None else cleaned)
    return " ".join(words).strip()


def _looks_upper_heading(text: str) -> bool:
    stripped = _normalize_text(text)
    if not stripped:
        return False
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    return (upper / len(letters)) >= 0.7 and len(stripped.split()) <= 8


def _title_keywords(kind: str):
    return TABLE_KEYWORDS if kind == "table" else FIGURE_KEYWORDS


def _score_title_candidate(text: str, kind: str) -> int:
    candidate = _normalize_heading_spelling(text)
    if not candidate:
        return -999

    lower = candidate.lower()
    words = candidate.split()
    score = 0

    if CAPTION_RE.search(candidate):
        score += 8
    if any(keyword in lower for keyword in _title_keywords(kind)):
        score += 5
    if kind == "table" and lower in GENERIC_TABLE_HEADER_LINES:
        score -= 8
    if kind == "table" and any(hint in lower for hint in TABLE_SECTION_HINTS):
        score += 4
    if _looks_upper_heading(candidate):
        score += 4
    if len(words) <= 6:
        score += 2
    elif len(words) >= 12:
        score -= 4
    if candidate.startswith(("*", "+", "-", "\u2022")):
        score -= 4
    if IMPERATIVE_START_RE.search(candidate.lstrip("*+-• ").strip()):
        score -= 3
    if "," in candidate or candidate.endswith("."):
        score -= 4
    if len(candidate) > 90:
        score -= 3
    if lower.startswith(("for ", "if ", "when ", "with ", "after ")):
        score -= 2

    return score


def _choose_title(kind: str, context: str, ocr_text: str = ""):
    normalized_context_lines = []
    candidates = []
    for source_name, block in (("context", context), ("ocr", ocr_text)):
        for idx, raw_line in enumerate((block or "").splitlines()):
            line = _normalize_heading_spelling(raw_line)
            if not line:
                continue
            if source_name == "context":
                normalized_context_lines.append(line)
            score = _score_title_candidate(line, kind)
            if source_name == "context":
                score += 1
                if kind == "table" and idx == 0:
                    score += 4
                elif kind == "table" and idx == 1:
                    score += 2
            else:
                if kind == "table":
                    score -= 1
            candidates.append((line, score, source_name))

    if not candidates:
        fallback = _normalize_heading_spelling(_extract_caption_title(context, ocr_text))
        return fallback, False, -999

    best_line, best_score, _ = max(candidates, key=lambda item: item[1])
    fallback = _normalize_heading_spelling(_extract_caption_title(context, ocr_text))
    if best_score < 1 and fallback:
        best_line = fallback
        best_score = _score_title_candidate(best_line, kind)
    best_line = _canonicalize_heading(best_line)

    if kind == "table":
        best_line = _refine_table_title(best_line, normalized_context_lines, ocr_text)
        best_score = _score_title_candidate(best_line, kind)

    context_match = best_line in {_normalize_heading_spelling(line) for line in (context or "").splitlines() if _normalize_heading_spelling(line)}
    return best_line, context_match, best_score


def _canonicalize_heading(candidate: str) -> str:
    normalized = _normalize_heading_spelling(candidate)
    if not normalized:
        return ""
    lower = normalized.lower()
    best_match = normalized
    best_ratio = 0.0
    for heading in CANONICAL_HEADINGS:
        ratio = difflib.SequenceMatcher(None, lower, heading.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = heading
    if best_ratio >= 0.78:
        return best_match
    return normalized


def _infer_troubleshooting_subheading(ocr_text: str) -> str:
    lower = _normalize_heading_spelling(ocr_text).lower()
    if not lower:
        return ""
    if any(token in lower for token in ("out of fuel", "fuel valve", "oil alert", "spark plug", "bad fuel", "fuel filter")):
        return "Engine Will Not Start"
    if any(token in lower for token in ("ac", "overload", "output", "circuit protector", "cooling air", "faulty appliance", "power tool")):
        return "Loss of Power"
    return ""


def _refine_table_title(best_line: str, context_lines: List[str], ocr_text: str) -> str:
    specific_context_titles = []
    for line in context_lines:
        canon = _canonicalize_heading(line)
        if canon in ("Engine Will Not Start", "Loss of Power", "No Power at the AC Receptacles"):
            specific_context_titles.append(canon)

    best_canon = _canonicalize_heading(best_line)
    if best_canon == "Maintenance Schedule":
        return best_canon

    joined_context = " ".join(context_lines).lower()
    is_troubleshooting_context = (
        "troubleshooting" in joined_context
        or "start" in joined_context
        or "power" in joined_context
    )

    inferred = _infer_troubleshooting_subheading(ocr_text) if is_troubleshooting_context else ""
    if inferred:
        return inferred

    if best_line.lower() in GENERIC_TABLE_HEADER_LINES:
        if specific_context_titles:
            return specific_context_titles[0]

    if best_canon == "Troubleshooting" and specific_context_titles:
        return specific_context_titles[0]

    return best_canon


def _bbox_metrics(bbox, page_width: int, page_height: int):
    x1, y1, x2, y2 = bbox
    width = max(0, int(x2) - int(x1))
    height = max(0, int(y2) - int(y1))
    area = width * height
    page_area = max(page_width * page_height, 1)
    area_ratio = area / page_area
    aspect_ratio = (width / max(height, 1)) if height else 0.0
    return width, height, area_ratio, aspect_ratio


def _refine_bbox_to_content(
    gray_page,
    bbox,
    page_width: int,
    page_height: int,
    *,
    white_threshold: int = 245,
    padding: int = 8,
    min_content_pixels: int = 48,
    min_retained_ratio: float = 0.28,
):
    if np is None:
        return bbox
    crop = crop_image(gray_page, bbox)
    if crop is None or getattr(crop, "size", 0) == 0:
        return bbox
    mask = crop < white_threshold
    if int(mask.sum()) < min_content_pixels:
        return bbox
    coords = np.argwhere(mask)
    if coords.size == 0:
        return bbox
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    orig_x1, orig_y1, orig_x2, orig_y2 = [int(value) for value in bbox]
    refined_x1 = max(0, orig_x1 + int(x_min) - padding)
    refined_y1 = max(0, orig_y1 + int(y_min) - padding)
    refined_x2 = min(page_width, orig_x1 + int(x_max) + 1 + padding)
    refined_y2 = min(page_height, orig_y1 + int(y_max) + 1 + padding)

    orig_width = max(1, orig_x2 - orig_x1)
    orig_height = max(1, orig_y2 - orig_y1)
    refined_width = max(1, refined_x2 - refined_x1)
    refined_height = max(1, refined_y2 - refined_y1)
    if (refined_width / orig_width) < min_retained_ratio or (refined_height / orig_height) < min_retained_ratio:
        return bbox
    return [refined_x1, refined_y1, refined_x2, refined_y2]


def _bbox_center_x(bbox) -> float:
    return (float(bbox[0]) + float(bbox[2])) / 2.0


def _bbox_horizontal_overlap_ratio(a, b) -> float:
    ax1, _, ax2, _ = [float(value) for value in a]
    bx1, _, bx2, _ = [float(value) for value in b]
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width = max(ax2 - ax1, bx2 - bx1, 1.0)
    return overlap / width


def _clip_bbox(bbox, page_width: int, page_height: int):
    x1, y1, x2, y2 = [int(round(value)) for value in bbox]
    x1 = max(0, min(page_width - 1, x1))
    y1 = max(0, min(page_height - 1, y1))
    x2 = max(x1 + 1, min(page_width, x2))
    y2 = max(y1 + 1, min(page_height, y2))
    return [x1, y1, x2, y2]


def _is_same_caption_column(caption_bbox, other_bbox, page_width: int) -> bool:
    overlap = _bbox_horizontal_overlap_ratio(caption_bbox, other_bbox)
    if overlap >= 0.22:
        return True
    margin = max(18.0, page_width * 0.03)
    center_x = _bbox_center_x(caption_bbox)
    ox1, _, ox2, _ = [float(value) for value in other_bbox]
    return (ox1 - margin) <= center_x <= (ox2 + margin)


def _caption_signature(candidate: dict) -> tuple:
    return (
        _normalize_text(str(candidate.get("type") or "")).lower(),
        _normalize_text(str(candidate.get("label") or "")),
        _normalize_text(str(candidate.get("caption_text") or "")),
        tuple(int(value) for value in (candidate.get("caption_bbox") or [0, 0, 0, 0])),
    )


def _find_caption_anchor_bbox(
    *,
    kind: str,
    caption_candidate: dict,
    caption_candidates: List[dict],
    text_block_records: List[tuple],
    layout_blocks: List[Block],
    page_width: int,
    page_height: int,
):
    caption_bbox = caption_candidate.get("caption_bbox")
    if not isinstance(caption_bbox, list) or len(caption_bbox) != 4:
        return None

    x_margin = max(14, int(page_width * 0.012))
    y_margin = max(8, int(page_height * 0.01))
    min_height = max(48, int(page_height * 0.07))
    default_height = max(min_height, int(page_height * 0.18))

    same_column_text_bboxes = [
        bbox
        for bbox, _ in text_block_records
        if _is_same_caption_column(caption_bbox, bbox, page_width)
    ]
    same_column_caption_bboxes = [
        candidate.get("caption_bbox")
        for candidate in caption_candidates
        if candidate is not caption_candidate
        and isinstance(candidate.get("caption_bbox"), list)
        and len(candidate.get("caption_bbox")) == 4
        and _is_same_caption_column(caption_bbox, candidate.get("caption_bbox"), page_width)
    ]

    column_bboxes = same_column_text_bboxes + [caption_bbox]
    x1 = min(int(bbox[0]) for bbox in column_bboxes) - x_margin
    x2 = max(int(bbox[2]) for bbox in column_bboxes) + x_margin

    same_kind_blocks = [
        list(block.bbox)
        for block in layout_blocks
        if block.kind == kind and _is_same_caption_column(caption_bbox, block.bbox, page_width)
    ]

    if kind == "figure":
        caption_top = int(caption_bbox[1])
        caption_bottom = int(caption_bbox[3])
        anchor_candidates = []
        for bbox in same_kind_blocks:
            block_top = int(bbox[1])
            block_bottom = int(bbox[3])
            if block_top >= caption_bottom:
                continue
            gap = max(0, caption_top - block_bottom)
            anchor_candidates.append((gap, bbox))

        selected_blocks = []
        if anchor_candidates:
            min_gap = min(gap for gap, _ in anchor_candidates)
            gap_tolerance = max(18, int(page_height * 0.03))
            selected_blocks = [
                bbox
                for gap, bbox in anchor_candidates
                if gap <= (min_gap + gap_tolerance)
            ]

        if selected_blocks:
            x1 = min(int(bbox[0]) for bbox in selected_blocks) - x_margin
            x2 = max(int(bbox[2]) for bbox in selected_blocks) + x_margin
            top = min(int(bbox[1]) for bbox in selected_blocks) - y_margin
            bottom = max(int(bbox[3]) for bbox in selected_blocks) + y_margin
        else:
            bottom = caption_top - y_margin
            upper_text_bottoms = [
                int(bbox[3])
                for bbox in same_column_text_bboxes
                if int(bbox[3]) <= int(caption_bbox[1])
            ]
            upper_caption_bottoms = [
                int(bbox[3])
                for bbox in same_column_caption_bboxes
                if int(bbox[3]) <= int(caption_bbox[1])
            ]
            top = max(upper_text_bottoms + upper_caption_bottoms, default=max(0, bottom - default_height)) + y_margin

        if bottom - top < min_height:
            top = max(0, bottom - max(default_height, min_height))
        if bottom <= top:
            return None
        return _clip_bbox([x1, top, x2, bottom], page_width, page_height)

    top = int(caption_bbox[3]) + y_margin
    anchor_block = None
    anchor_gap = None
    for bbox in same_kind_blocks:
        if int(bbox[3]) <= int(caption_bbox[3]):
            continue
        gap = max(0, int(bbox[1]) - int(caption_bbox[3]))
        if anchor_gap is None or gap < anchor_gap:
            anchor_gap = gap
            anchor_block = bbox

    if anchor_block is not None:
        x1 = min(x1, int(anchor_block[0]) - x_margin)
        x2 = max(x2, int(anchor_block[2]) + x_margin)
        bottom = int(anchor_block[3]) + y_margin
    else:
        lower_text_tops = [
            int(bbox[1])
            for bbox in same_column_text_bboxes
            if int(bbox[1]) >= int(caption_bbox[3])
        ]
        lower_caption_tops = [
            int(bbox[1])
            for bbox in same_column_caption_bboxes
            if int(bbox[1]) >= int(caption_bbox[3])
        ]
        bottom = min(lower_text_tops + lower_caption_tops, default=min(page_height, top + default_height)) - y_margin

    if bottom - top < min_height:
        bottom = min(page_height, top + max(default_height, min_height))
    if bottom <= top:
        return None
    return _clip_bbox([x1, top, x2, bottom], page_width, page_height)


def _extract_subfigure_markers(
    words: List[dict],
    parent_bbox,
    expected_labels: List[str],
):
    if not words or not expected_labels:
        return {}
    x1, y1, x2, y2 = [float(value) for value in parent_bbox]
    x_margin = max(18.0, (x2 - x1) * 0.08)
    y_margin = max(18.0, (y2 - y1) * 0.18)
    label_set = {str(label).lower() for label in expected_labels if str(label).strip()}
    markers = {}
    for word in words:
        text = _normalize_text(word.get("text", ""))
        match = SUBFIGURE_MARKER_WORD_RE.match(text)
        if not match:
            continue
        label = match.group(1).lower()
        if label not in label_set:
            continue
        cx = (float(word["x0"]) + float(word["x1"])) / 2.0
        cy = (float(word["top"]) + float(word["bottom"])) / 2.0
        if not ((x1 - x_margin) <= cx <= (x2 + x_margin) and (y1 - y_margin) <= cy <= (y2 + y_margin)):
            continue
        existing = markers.get(label)
        if existing is None or cy < existing["cy"]:
            markers[label] = {"cx": cx, "cy": cy}
    return markers


def _split_bbox_equally(parent_bbox, labels: List[str]):
    x1, y1, x2, y2 = [int(value) for value in parent_bbox]
    if not labels:
        return {}
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    horizontal = width >= height
    boxes = {}
    count = len(labels)
    for idx, label in enumerate(labels):
        if horizontal:
            sx1 = x1 + int(round(width * idx / count))
            sx2 = x1 + int(round(width * (idx + 1) / count))
            boxes[label] = [sx1, y1, max(sx1 + 1, sx2), y2]
        else:
            sy1 = y1 + int(round(height * idx / count))
            sy2 = y1 + int(round(height * (idx + 1) / count))
            boxes[label] = [x1, sy1, x2, max(sy1 + 1, sy2)]
    return boxes


def _split_bbox_by_projection(gray_page, parent_bbox, labels: List[str]):
    if np is None or gray_page is None or not labels:
        return {}
    x1, y1, x2, y2 = [int(value) for value in parent_bbox]
    crop = crop_image(gray_page, parent_bbox)
    if crop is None or getattr(crop, "size", 0) == 0:
        return {}
    height, width = crop.shape[:2]
    horizontal = width >= height
    count = len(labels)
    axis_sum = (crop < 245).sum(axis=0 if horizontal else 1)
    axis_len = width if horizontal else height
    if axis_len <= count * 4:
        return {}
    split_points: List[int] = []
    for idx in range(1, count):
        expected = int(round(axis_len * idx / count))
        window = max(12, axis_len // (count * 5))
        lo = max(1, expected - window)
        hi = min(axis_len - 2, expected + window)
        if hi <= lo:
            continue
        slice_values = axis_sum[lo:hi + 1]
        valley_offset = int(slice_values.argmin())
        split_points.append(lo + valley_offset)
    if len(split_points) != count - 1:
        return {}

    positions = [0] + sorted(split_points) + [axis_len]
    boxes = {}
    for idx, label in enumerate(labels):
        start = positions[idx]
        end = positions[idx + 1]
        if horizontal:
            boxes[label] = [x1 + start, y1, x1 + max(start + 1, end), y2]
        else:
            boxes[label] = [x1, y1 + start, x2, y1 + max(start + 1, end)]
    return boxes


def _derive_subfigure_bboxes(
    parent_bbox,
    subfigure_labels: List[str],
    scaled_pdf_words: List[dict],
    gray_page,
    page_width: int,
    page_height: int,
):
    labels = [str(label).lower() for label in subfigure_labels if str(label).strip()]
    if len(labels) < 2:
        return {}

    marker_map = _extract_subfigure_markers(scaled_pdf_words, parent_bbox, labels)
    if len(marker_map) < len(labels):
        projection_boxes = _split_bbox_by_projection(gray_page, parent_bbox, labels)
        if projection_boxes:
            return {
                label: _clip_bbox(bbox, page_width, page_height)
                for label, bbox in projection_boxes.items()
            }
        return {
            label: _clip_bbox(bbox, page_width, page_height)
            for label, bbox in _split_bbox_equally(parent_bbox, labels).items()
        }

    xs = [marker_map[label]["cx"] for label in labels if label in marker_map]
    ys = [marker_map[label]["cy"] for label in labels if label in marker_map]
    horizontal = (max(xs) - min(xs)) >= (max(ys) - min(ys))
    ordered = sorted(
        labels,
        key=lambda label: marker_map[label]["cx"] if horizontal else marker_map[label]["cy"],
    )

    x1, y1, x2, y2 = [float(value) for value in parent_bbox]
    positions = [marker_map[label]["cx"] if horizontal else marker_map[label]["cy"] for label in ordered]
    boxes = {}
    for idx, label in enumerate(ordered):
        if horizontal:
            left = x1 if idx == 0 else (positions[idx - 1] + positions[idx]) / 2.0
            right = x2 if idx == len(ordered) - 1 else (positions[idx] + positions[idx + 1]) / 2.0
            boxes[label] = _clip_bbox([left, y1, right, y2], page_width, page_height)
        else:
            top = y1 if idx == 0 else (positions[idx - 1] + positions[idx]) / 2.0
            bottom = y2 if idx == len(ordered) - 1 else (positions[idx] + positions[idx + 1]) / 2.0
            boxes[label] = _clip_bbox([x1, top, x2, bottom], page_width, page_height)
    return boxes


def _create_subfigure_items(
    *,
    parent_item: dict,
    page_image,
    media_dir: str,
    page_width: int,
    page_height: int,
    scaled_pdf_words: List[dict],
    gray_page,
):
    if _normalize_text(parent_item.get("type", "")).lower() != "figure":
        return []
    base_label = _normalize_text(parent_item.get("label", ""))
    subfigure_labels = [
        str(label).lower()
        for label in parent_item.get("subfigure_labels") or []
        if _normalize_text(label)
    ]
    if not base_label or len(subfigure_labels) < 2:
        return []

    subfigure_descriptions = parent_item.get("subfigure_descriptions")
    if not isinstance(subfigure_descriptions, dict):
        subfigure_descriptions = {}

    parent_bbox = parent_item.get("bbox") or parent_item.get("region_bbox")
    if not isinstance(parent_bbox, list) or len(parent_bbox) != 4:
        return []

    subfigure_boxes = _derive_subfigure_bboxes(
        parent_bbox,
        subfigure_labels,
        scaled_pdf_words,
        gray_page,
        page_width,
        page_height,
    )
    if not subfigure_boxes:
        return []

    parent_image_path = Path(str(parent_item.get("image_path") or ""))
    subfigure_items = []
    for label in subfigure_labels:
        bbox = subfigure_boxes.get(label)
        if not bbox:
            continue
        _, _, area_ratio, aspect_ratio = _bbox_metrics(bbox, page_width, page_height)
        if area_ratio < 0.004:
            continue
        bbox = _refine_bbox_to_content(
            gray_page,
            bbox,
            page_width,
            page_height,
        )

        sub_label = f"{base_label}{label}"
        sub_description = _normalize_text(subfigure_descriptions.get(label, ""))
        title = f"{sub_label}. {sub_description}".strip(" .") if sub_description else sub_label
        file_name = f"{parent_image_path.stem}_subfig_{label}{parent_image_path.suffix or '.png'}"
        image_path = os.path.join(media_dir, file_name)
        page_image.crop(tuple(bbox)).save(image_path)

        context_parts = [sub_description, _normalize_text(parent_item.get("caption_text", "")), _normalize_text(parent_item.get("context", ""))]
        context = "\n".join(part for part in context_parts if part).strip()
        citation_aliases = build_citation_aliases(base_label, label)
        parent_aliases = [
            _normalize_text(alias)
            for alias in parent_item.get("citation_aliases") or []
            if _normalize_text(alias)
        ]
        for alias in parent_aliases:
            citation_aliases.append(alias)
        citation_aliases = sorted(set(citation_aliases))

        subfigure_items.append(
            {
                **parent_item,
                "bbox": bbox,
                "region_bbox": bbox,
                "image_path": image_path,
                "title": title,
                "label": sub_label,
                "caption_text": title,
                "citation_aliases": citation_aliases,
                "subfigure_labels": [label],
                "subfigure_descriptions": {label: sub_description} if sub_description else {},
                "association_method": "subfigure_partition",
                "association_confidence": min(0.99, float(parent_item.get("association_confidence") or 0.7) + 0.05),
                "area_ratio": round(area_ratio, 4),
                "aspect_ratio": round(aspect_ratio, 4),
                "object_id": _build_media_object_id(
                    "figure",
                    int(parent_item.get("source_pdf_page") or parent_item.get("page") or 0),
                    file_name,
                    sub_label,
                ),
                "subfigure_of": parent_item.get("object_id"),
            }
        )
    return subfigure_items


def _ink_ratio(binary_crop) -> float:
    try:
        if binary_crop is None or getattr(binary_crop, "size", 0) == 0:
            return 0.0
        return float((binary_crop == 0).mean())
    except Exception:
        return 0.0


def _should_keep_figure(area_ratio: float, title: str, title_score: int, ink_ratio: float) -> bool:
    title_norm = _normalize_text(title)
    title_lower = title_norm.lower()
    strong_title = title_score >= 6 or any(keyword in title_lower for keyword in FIGURE_KEYWORDS)
    sentence_like = (
        len(title_norm.split()) >= 8
        or "," in title_norm
        or title_norm.endswith(".")
    )

    if area_ratio >= 0.12:
        return True
    if strong_title and area_ratio >= 0.03:
        return True
    if ink_ratio <= 0.015 and area_ratio >= 0.025:
        return True
    if sentence_like:
        return False
    if area_ratio < 0.03:
        return False
    return strong_title


def _detect_low_value_reason(kind: str, source_pdf_page: int, title: str, context: str) -> str:
    title_norm = _normalize_text(title)
    title_lower = title_norm.lower()
    context_lower = _normalize_text(context).lower()
    title_canon = _canonicalize_heading(title_norm)

    if title_canon in HIGH_VALUE_MEDIA_TITLES:
        return ""

    if any(keyword in title_lower for keyword in LOW_VALUE_TITLE_KEYWORDS):
        return "low_value_admin_section"
    if any(keyword in context_lower for keyword in LOW_VALUE_CONTEXT_KEYWORDS):
        return "low_value_admin_context"

    sentence_like = (
        len(title_norm.split()) >= 8
        or "," in title_norm
        or title_norm.endswith(".")
    )
    if source_pdf_page == 1 and "manual" in title_lower:
        return "cover_page"
    if source_pdf_page >= 19 and sentence_like:
        return "late_page_sentence_like_admin_capture"
    if source_pdf_page >= 19 and kind == "figure" and title_norm and title_norm.isupper():
        if "registration" in title_lower or "customer" in title_lower:
            return "late_page_admin_heading"
    return ""


def _write_media_text(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(
                f"Page {item.get('page', 0):03d} "
                f"(source {item.get('source_pdf_page', 0):03d}) "
                f"[{item.get('type', '')}] "
                f"bbox={item.get('bbox', [])} "
                f"image={item.get('image_path', '')} "
                f"family={item.get('document_family', item.get('page_family', ''))} "
                f"area_ratio={item.get('area_ratio', 0.0):.4f} "
                f"title_score={item.get('title_score', 0)} "
                f"false_positive={item.get('is_probable_false_positive', False)} "
                f"low_value={item.get('is_low_value', False)}\n"
            )
            label = (item.get("label") or "").strip()
            if label:
                f.write("Label:\n" + label + "\n")
            title = (item.get("title") or "").strip()
            if title:
                f.write("Title:\n" + title + "\n")
            caption_text = (item.get("caption_text") or "").strip()
            if caption_text and caption_text != title:
                f.write("Caption:\n" + caption_text + "\n")
            context = (item.get("context") or "").strip()
            if context:
                f.write("Context:\n" + context + "\n")
            ocr_text = (item.get("ocr_text") or "").strip()
            if ocr_text:
                f.write("OCR:\n" + ocr_text + "\n")
            f.write("\n")


def _sort_words(words: List[dict]) -> List[dict]:
    return sorted(
        words,
        key=lambda word: (
            float(word.get("top", 0.0)),
            float(word.get("x0", 0.0)),
        ),
    )


def _words_to_block_text(words: List[dict], line_tolerance: float = 10.0) -> str:
    if not words:
        return ""

    lines = []
    current_line = []
    current_top = None
    for word in _sort_words(words):
        top = float(word.get("top", 0.0))
        text = _normalize_text(word.get("text", ""))
        if not text:
            continue
        if current_top is None or abs(top - current_top) <= line_tolerance:
            current_line.append(text)
            current_top = top if current_top is None else current_top
            continue
        lines.append(" ".join(current_line))
        current_line = [text]
        current_top = top

    if current_line:
        lines.append(" ".join(current_line))
    return "\n".join(line for line in lines if line).strip()


def _build_line_records_from_words(words: List[dict], line_tolerance: float = 12.0) -> List[tuple]:
    if not words:
        return []

    lines = []
    current_line = []
    current_top = None
    for word in _sort_words(words):
        top = float(word.get("top", 0.0))
        text = _normalize_text(word.get("text", ""))
        if not text:
            continue
        if current_top is None or abs(top - current_top) <= line_tolerance:
            current_line.append(word)
            current_top = top if current_top is None else current_top
            continue
        lines.append(current_line)
        current_line = [word]
        current_top = top

    if current_line:
        lines.append(current_line)

    records = []
    for line_words in lines:
        text = " ".join(_normalize_text(word.get("text", "")) for word in line_words).strip()
        if not text:
            continue
        bbox = [
            int(min(float(word.get("x0", 0.0)) for word in line_words)),
            int(min(float(word.get("top", 0.0)) for word in line_words)),
            int(max(float(word.get("x1", 0.0)) for word in line_words)),
            int(max(float(word.get("bottom", 0.0)) for word in line_words)),
        ]
        records.append((bbox, text))
    return records


def _merge_text_block_records(primary_records: List[tuple], fallback_records: List[tuple]) -> List[tuple]:
    merged = list(primary_records)
    seen = {
        (
            tuple(int(value) for value in bbox),
            _normalize_text(text),
        )
        for bbox, text in primary_records
        if _normalize_text(text)
    }
    seen_texts = {_normalize_text(text) for _, text in primary_records if _normalize_text(text)}
    for bbox, text in fallback_records:
        normalized_text = _normalize_text(text)
        if not normalized_text:
            continue
        key = (tuple(int(value) for value in bbox), normalized_text)
        if key in seen or normalized_text in seen_texts:
            continue
        seen.add(key)
        seen_texts.add(normalized_text)
        merged.append((bbox, normalized_text))
    merged.sort(key=lambda item: (int(item[0][1]), int(item[0][0])))
    return merged


def _extract_pdf_words(page, page_width: int, page_height: int) -> List[dict]:
    if page is None:
        return []
    try:
        words = page.extract_words(
            x_tolerance=2,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True,
        ) or []
    except Exception:
        return []

    page_width_pdf = max(float(getattr(page, "width", 0.0) or 0.0), 1.0)
    page_height_pdf = max(float(getattr(page, "height", 0.0) or 0.0), 1.0)
    scale_x = page_width / page_width_pdf
    scale_y = page_height / page_height_pdf
    scaled_words = []
    for word in words:
        text = _normalize_text(word.get("text", ""))
        if not text:
            continue
        x0 = float(word.get("x0", 0.0)) * scale_x
        x1 = float(word.get("x1", 0.0)) * scale_x
        y0 = float(word.get("top", 0.0)) * scale_y
        y1 = float(word.get("bottom", 0.0)) * scale_y
        scaled_words.append(
            {
                "text": text,
                "x0": x0,
                "x1": x1,
                "top": y0,
                "bottom": y1,
            }
        )
    return scaled_words


def _extract_text_from_bbox(words: List[dict], bbox: List[int], line_tolerance: float = 10.0) -> str:
    if not words:
        return ""
    x1, y1, x2, y2 = [float(value) for value in bbox]
    selected = []
    for word in words:
        cx = (float(word["x0"]) + float(word["x1"])) / 2.0
        cy = (float(word["top"]) + float(word["bottom"])) / 2.0
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            selected.append(word)
    return _words_to_block_text(selected, line_tolerance=line_tolerance)


def extract_media(
    pdf_path: str,
    out_dir: str,
    page_spec: str = "",
    dpi: int = 170,
    lang: str = "eng",
    include_figure_ocr: bool = False,
    use_text_layer_first: bool = True,
    layout_backend: str = "auto",
) -> str:
    _load_dotenv()
    poppler_path = _detect_poppler_path()
    requested_layout_backend = normalize_layout_backend(layout_backend)

    total_pages = _ensure_pdf_page_count(pdf_path, poppler_path)
    if total_pages <= 0:
        raise RuntimeError("Unable to determine PDF page count.")

    target_pages = _parse_page_spec(page_spec, total_pages)
    if not target_pages:
        raise RuntimeError("No valid pages selected.")

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    media_dir = os.path.join(out_dir, f"{base}_media")
    os.makedirs(media_dir, exist_ok=True)

    items = []
    filtered_out = []
    page_analyses = []
    caption_anchor_generated = 0
    observed_layout_backends = set()
    sample_page_index = 0
    pdf_doc = None
    if use_text_layer_first and pdfplumber is not None:
        try:
            pdf_doc = pdfplumber.open(pdf_path)
        except Exception:
            pdf_doc = None

    try:
        for source_pdf_page in target_pages:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=source_pdf_page,
                last_page=source_pdf_page,
                poppler_path=poppler_path,
            )
            if not images:
                continue

            sample_page_index += 1
            img_bgr, gray, bw, _ = preprocess_image(images[0])
            page_height, page_width = gray.shape[:2]
            blocks, resolved_layout_backend = detect_layout_blocks_with_backend(
                img_bgr,
                preferred_backend=requested_layout_backend,
            )
            observed_layout_backends.add(str(resolved_layout_backend or "none"))

            # If the layout detector returns only figures/tables, keep a full-page text fallback
            if not any(block.kind == "text" for block in blocks):
                h, w = gray.shape[:2]
                blocks = list(blocks) + [type(blocks[0])(kind="text", bbox=(0, 0, w, h))]

            pdf_page = None
            scaled_pdf_words = []
            page_word_line_records = []
            if pdf_doc is not None:
                try:
                    pdf_page = pdf_doc.pages[source_pdf_page - 1]
                    scaled_pdf_words = _extract_pdf_words(pdf_page, page_width, page_height)
                    page_word_line_records = _build_line_records_from_words(scaled_pdf_words)
                except Exception:
                    pdf_page = None
                    scaled_pdf_words = []
                    page_word_line_records = []

            text_block_records = []
            figure_index = 0
            table_index = 0

            # Prefer PDF text layer for context blocks; fall back to OCR only when needed.
            for block in blocks:
                if block.kind != "text":
                    continue
                cleaned = ""
                if scaled_pdf_words:
                    cleaned = _extract_text_from_bbox(scaled_pdf_words, list(block.bbox))
                if len(cleaned) < 12:
                    crop = crop_image(gray, block.bbox)
                    text, _, _, _ = ocr_tesseract(
                        crop,
                        lang,
                        DEFAULT_TESS_OEM,
                        DEFAULT_TESS_PSM_TEXT,
                        int(DEFAULT_CONF_THRESHOLD),
                        retry_psm=None,
                    )
                    cleaned = (text or "").strip()
                if cleaned:
                    text_block_records.append((list(block.bbox), cleaned))

            if page_word_line_records:
                text_block_records = _merge_text_block_records(
                    text_block_records,
                    page_word_line_records,
                )

            page_text_fragments = [text for _, text in text_block_records if _normalize_text(text)]
            if not page_text_fragments and scaled_pdf_words:
                full_page_text = _words_to_block_text(scaled_pdf_words)
                if full_page_text:
                    page_text_fragments.append(full_page_text)

            page_family_info = classify_page_family(
                "\n".join(page_text_fragments),
                source_name=os.path.basename(pdf_path),
            )
            caption_candidates = extract_caption_candidates(
                text_block_records,
                page_number=source_pdf_page,
            )
            page_analyses.append(
                {
                    "page_number": source_pdf_page,
                    "family": page_family_info.get("family", "scanned_unknown"),
                    "family_confidence": page_family_info.get("confidence", 0.0),
                    "layout_backend": str(resolved_layout_backend or "none"),
                    "caption_count": len(caption_candidates),
                    "text_excerpt": page_family_info.get("text_excerpt", ""),
                }
            )
            matched_caption_signatures = set()

            for block in blocks:
                if block.kind not in ("figure", "table"):
                    continue

                current_bbox = list(block.bbox)
                crop_gray = crop_image(gray, current_bbox)
                crop_bgr = crop_image(img_bgr, current_bbox)
                crop_bw = crop_image(bw, current_bbox)

                if block.kind == "table":
                    table_index += 1
                    file_name = f"page{source_pdf_page:03d}_table_{table_index:03d}.png"
                    ocr_source = "none"
                    ocr_text = ""
                    if scaled_pdf_words:
                        ocr_text = _extract_text_from_bbox(scaled_pdf_words, current_bbox)
                        if ocr_text:
                            ocr_source = "pdf_text_layer"
                    if len((ocr_text or "").strip()) < 24:
                        ocr_text, avg_conf = ocr_table_paddle(crop_bgr)
                        ocr_source = "ocr_paddle"
                        if not (ocr_text or "").strip() or avg_conf < int(DEFAULT_CONF_THRESHOLD):
                            ocr_text, _, _, _ = ocr_tesseract(
                                crop_gray,
                                lang,
                                DEFAULT_TESS_OEM,
                                DEFAULT_TESS_PSM_TABLE,
                                int(DEFAULT_CONF_THRESHOLD),
                                retry_psm=None,
                            )
                            ocr_source = "ocr_tesseract"
                else:
                    figure_index += 1
                    file_name = f"page{source_pdf_page:03d}_figure_{figure_index:03d}.png"
                    if include_figure_ocr:
                        ocr_text, _, _, _ = ocr_tesseract(
                            crop_gray,
                            lang,
                            DEFAULT_TESS_OEM,
                            DEFAULT_TESS_PSM_TEXT,
                            int(DEFAULT_CONF_THRESHOLD),
                            retry_psm=None,
                        )
                        ocr_source = "ocr_tesseract"
                    else:
                        ocr_text = ""
                        ocr_source = "disabled"

                image_path = os.path.join(media_dir, file_name)

                context = _collect_context(text_block_records, current_bbox)
                fallback_title, context_title_match, title_score = _choose_title(
                    block.kind,
                    context,
                    (ocr_text or "").strip(),
                )
                title = fallback_title
                caption_match = match_caption_to_block(
                    block_kind=block.kind,
                    block_bbox=current_bbox,
                    caption_candidates=caption_candidates,
                    page_width=page_width,
                    page_height=page_height,
                )
                caption_label = ""
                caption_text = ""
                caption_bbox = None
                caption_source = ""
                caption_confidence = 0.0
                citation_aliases: List[str] = []
                subfigure_labels: List[str] = []
                subfigure_descriptions = {}
                association_method = "context_title"
                association_confidence = round(
                    max(0.0, min(0.8, 0.25 + max(title_score, 0) / 12.0)),
                    3,
                )
                if caption_match:
                    candidate = caption_match["candidate"]
                    matched_caption_signatures.add(_caption_signature(candidate))
                    caption_label = _normalize_text(str(candidate.get("label") or ""))
                    caption_text = _normalize_text(str(candidate.get("caption_text") or ""))
                    caption_bbox = candidate.get("caption_bbox")
                    caption_source = _normalize_text(str(candidate.get("caption_source") or ""))
                    caption_confidence = float(candidate.get("caption_confidence") or 0.0)
                    citation_aliases = [
                        _normalize_text(alias)
                        for alias in candidate.get("citation_aliases") or []
                        if _normalize_text(alias)
                    ]
                    subfigure_labels = [
                        _normalize_text(label)
                        for label in candidate.get("subfigure_labels") or []
                        if _normalize_text(label)
                    ]
                    subfigure_descriptions = (
                        candidate.get("subfigure_descriptions")
                        if isinstance(candidate.get("subfigure_descriptions"), dict)
                        else {}
                    )
                    association_method = str(caption_match.get("association_method") or "caption_proximity")
                    association_confidence = float(caption_match.get("association_confidence") or association_confidence)
                    if caption_text:
                        title = caption_text
                        context_title_match = True
                        title_score = max(title_score, _score_title_candidate(title, block.kind))
                    if page_family_info.get("family") == "academic_paper":
                        synthetic_bbox = _find_caption_anchor_bbox(
                            kind=block.kind,
                            caption_candidate=candidate,
                            caption_candidates=caption_candidates,
                            text_block_records=text_block_records,
                            layout_blocks=blocks,
                            page_width=page_width,
                            page_height=page_height,
                        )
                        if synthetic_bbox:
                            current_bbox = synthetic_bbox
                            crop_gray = crop_image(gray, current_bbox)
                            crop_bgr = crop_image(img_bgr, current_bbox)
                            crop_bw = crop_image(bw, current_bbox)
                            context = _collect_context(text_block_records, current_bbox)
                            association_method = "caption_anchor_preferred"
                            association_confidence = min(0.99, association_confidence + 0.04)
                            if block.kind == "table":
                                ocr_text = ""
                                if scaled_pdf_words:
                                    ocr_text = _extract_text_from_bbox(scaled_pdf_words, current_bbox)
                                    if ocr_text:
                                        ocr_source = "pdf_text_layer"
                                if len((ocr_text or "").strip()) < 24:
                                    ocr_text, avg_conf = ocr_table_paddle(crop_bgr)
                                    ocr_source = "ocr_paddle"
                                    if not (ocr_text or "").strip() or avg_conf < int(DEFAULT_CONF_THRESHOLD):
                                        ocr_text, _, _, _ = ocr_tesseract(
                                            crop_gray,
                                            lang,
                                            DEFAULT_TESS_OEM,
                                            DEFAULT_TESS_PSM_TABLE,
                                            int(DEFAULT_CONF_THRESHOLD),
                                            retry_psm=None,
                                        )
                                        ocr_source = "ocr_tesseract"
                            elif include_figure_ocr:
                                refreshed_ocr_text = _extract_text_from_bbox(scaled_pdf_words, current_bbox) if scaled_pdf_words else ""
                                if refreshed_ocr_text:
                                    ocr_text = refreshed_ocr_text
                                    ocr_source = "pdf_text_layer"
                                elif len((ocr_text or "").strip()) < 16:
                                    ocr_text, _, _, _ = ocr_tesseract(
                                        crop_gray,
                                        lang,
                                        DEFAULT_TESS_OEM,
                                        DEFAULT_TESS_PSM_TEXT,
                                        int(DEFAULT_CONF_THRESHOLD),
                                        retry_psm=None,
                                    )
                                    ocr_source = "ocr_tesseract"
                refined_bbox = _refine_bbox_to_content(
                    gray,
                    current_bbox,
                    page_width,
                    page_height,
                )
                if refined_bbox != current_bbox:
                    current_bbox = refined_bbox
                images[0].crop(tuple(current_bbox)).save(image_path)
                _, _, area_ratio, aspect_ratio = _bbox_metrics(current_bbox, page_width, page_height)
                ocr_char_count = len((ocr_text or "").strip())
                ink_ratio = _ink_ratio(crop_bw)
                probable_false_positive = False
                false_positive_reason = ""
                low_value_reason = _detect_low_value_reason(
                    block.kind,
                    source_pdf_page,
                    title,
                    context,
                )
                is_low_value = bool(low_value_reason)
                if block.kind == "figure":
                    keep_figure = _should_keep_figure(area_ratio, title, title_score, ink_ratio)
                    probable_false_positive = not keep_figure
                    if probable_false_positive:
                        false_positive_reason = "small_or_text_like_figure_region"

                item_record = {
                    "page": sample_page_index,
                    "source_pdf_page": source_pdf_page,
                    "pdf_page": source_pdf_page,
                    "page_index": source_pdf_page - 1,
                    "type": block.kind,
                    "bbox": current_bbox,
                    "region_bbox": current_bbox,
                    "image_path": image_path,
                    "ocr_text": (ocr_text or "").strip(),
                    "ocr_source": ocr_source,
                    "title": title,
                    "label": caption_label,
                    "caption_text": caption_text,
                    "caption_bbox": caption_bbox,
                    "caption_source": caption_source,
                    "caption_confidence": round(caption_confidence, 3) if caption_confidence else 0.0,
                    "citation_aliases": citation_aliases,
                    "subfigure_labels": subfigure_labels,
                    "subfigure_descriptions": subfigure_descriptions,
                    "association_method": association_method,
                    "association_confidence": round(association_confidence, 3),
                    "page_family": page_family_info.get("family", "scanned_unknown"),
                    "page_family_confidence": page_family_info.get("confidence", 0.0),
                    "layout_backend": str(resolved_layout_backend or "none"),
                    "context": context,
                    "area_ratio": round(area_ratio, 4),
                    "aspect_ratio": round(aspect_ratio, 4),
                    "context_title_match": context_title_match,
                    "title_score": title_score,
                    "ocr_char_count": ocr_char_count,
                    "ink_ratio": round(ink_ratio, 4),
                    "is_probable_false_positive": probable_false_positive,
                    "false_positive_reason": false_positive_reason,
                    "is_low_value": is_low_value,
                    "low_value_reason": low_value_reason,
                    "object_id": _build_media_object_id(
                        block.kind,
                        source_pdf_page,
                        file_name,
                        caption_label or title,
                    ),
                }

                if probable_false_positive or is_low_value:
                    filtered_out.append(item_record)
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                    continue

                items.append(item_record)
                items.extend(
                    _create_subfigure_items(
                        parent_item=item_record,
                        page_image=images[0],
                        media_dir=media_dir,
                        page_width=page_width,
                        page_height=page_height,
                        scaled_pdf_words=scaled_pdf_words,
                        gray_page=gray,
                    )
                )

            if page_family_info.get("family") == "academic_paper":
                for caption_candidate in caption_candidates:
                    candidate_signature = _caption_signature(caption_candidate)
                    if candidate_signature in matched_caption_signatures:
                        continue

                    kind = "table" if str(caption_candidate.get("type") or "").strip().lower() == "table" else "figure"
                    synthetic_bbox = _find_caption_anchor_bbox(
                        kind=kind,
                        caption_candidate=caption_candidate,
                        caption_candidates=caption_candidates,
                        text_block_records=text_block_records,
                        layout_blocks=blocks,
                        page_width=page_width,
                        page_height=page_height,
                    )
                    if not synthetic_bbox:
                        continue

                    if any(
                        item.get("source_pdf_page") == source_pdf_page
                        and _normalize_text(str(item.get("label") or "")) == _normalize_text(str(caption_candidate.get("label") or ""))
                        and _normalize_text(str(item.get("type") or "")).lower() == kind
                        for item in items
                    ):
                        continue

                    crop_gray = crop_image(gray, synthetic_bbox)
                    crop_bgr = crop_image(img_bgr, synthetic_bbox)
                    crop_bw = crop_image(bw, synthetic_bbox)

                    if kind == "table":
                        table_index += 1
                        file_name = f"page{source_pdf_page:03d}_table_{table_index:03d}.png"
                        ocr_source = "none"
                        ocr_text = ""
                        if scaled_pdf_words:
                            ocr_text = _extract_text_from_bbox(scaled_pdf_words, synthetic_bbox)
                            if ocr_text:
                                ocr_source = "pdf_text_layer"
                        if len((ocr_text or "").strip()) < 24:
                            ocr_text, avg_conf = ocr_table_paddle(crop_bgr)
                            ocr_source = "ocr_paddle"
                            if not (ocr_text or "").strip() or avg_conf < int(DEFAULT_CONF_THRESHOLD):
                                ocr_text, _, _, _ = ocr_tesseract(
                                    crop_gray,
                                    lang,
                                    DEFAULT_TESS_OEM,
                                    DEFAULT_TESS_PSM_TABLE,
                                    int(DEFAULT_CONF_THRESHOLD),
                                    retry_psm=None,
                                )
                                ocr_source = "ocr_tesseract"
                    else:
                        figure_index += 1
                        file_name = f"page{source_pdf_page:03d}_figure_{figure_index:03d}.png"
                        ocr_text = ""
                        ocr_source = "disabled"
                        if scaled_pdf_words:
                            ocr_text = _extract_text_from_bbox(scaled_pdf_words, synthetic_bbox)
                            if ocr_text:
                                ocr_source = "pdf_text_layer"
                        if include_figure_ocr and len((ocr_text or "").strip()) < 16:
                            ocr_text, _, _, _ = ocr_tesseract(
                                crop_gray,
                                lang,
                                DEFAULT_TESS_OEM,
                                DEFAULT_TESS_PSM_TEXT,
                                int(DEFAULT_CONF_THRESHOLD),
                                retry_psm=None,
                            )
                            ocr_source = "ocr_tesseract"

                    image_path = os.path.join(media_dir, file_name)
                    refined_bbox = _refine_bbox_to_content(
                        gray,
                        synthetic_bbox,
                        page_width,
                        page_height,
                    )
                    if refined_bbox != synthetic_bbox:
                        synthetic_bbox = refined_bbox
                    images[0].crop(tuple(synthetic_bbox)).save(image_path)

                    caption_label = _normalize_text(str(caption_candidate.get("label") or ""))
                    caption_text = _normalize_text(str(caption_candidate.get("caption_text") or ""))
                    title = caption_text or caption_label or f"{kind.title()} on page {source_pdf_page}"
                    context = _collect_context(text_block_records, synthetic_bbox)
                    if caption_text and caption_text not in context:
                        context = "\n".join(part for part in [caption_text, context] if part).strip()

                    caption_confidence = float(caption_candidate.get("caption_confidence") or 0.0)
                    citation_aliases = [
                        _normalize_text(alias)
                        for alias in caption_candidate.get("citation_aliases") or []
                        if _normalize_text(alias)
                    ]
                    subfigure_labels = [
                        _normalize_text(label)
                        for label in caption_candidate.get("subfigure_labels") or []
                        if _normalize_text(label)
                    ]
                    subfigure_descriptions = (
                        caption_candidate.get("subfigure_descriptions")
                        if isinstance(caption_candidate.get("subfigure_descriptions"), dict)
                        else {}
                    )
                    title_score = max(_score_title_candidate(title, kind), 9 if caption_label else 6)
                    _, _, area_ratio, aspect_ratio = _bbox_metrics(synthetic_bbox, page_width, page_height)
                    ocr_char_count = len((ocr_text or "").strip())
                    ink_ratio = _ink_ratio(crop_bw)
                    probable_false_positive = False
                    false_positive_reason = ""
                    low_value_reason = _detect_low_value_reason(
                        kind,
                        source_pdf_page,
                        title,
                        context,
                    )
                    is_low_value = bool(low_value_reason)
                    if kind == "figure":
                        keep_figure = _should_keep_figure(area_ratio, title, title_score, ink_ratio)
                        if not keep_figure and (caption_label or caption_text):
                            keep_figure = area_ratio >= 0.01 and ink_ratio <= 0.35
                        probable_false_positive = not keep_figure
                        if probable_false_positive:
                            false_positive_reason = "caption_anchor_region_too_small_or_text_like"

                    item_record = {
                        "page": sample_page_index,
                        "source_pdf_page": source_pdf_page,
                        "pdf_page": source_pdf_page,
                        "page_index": source_pdf_page - 1,
                        "type": kind,
                        "bbox": synthetic_bbox,
                        "region_bbox": synthetic_bbox,
                        "image_path": image_path,
                        "ocr_text": (ocr_text or "").strip(),
                        "ocr_source": ocr_source,
                        "title": title,
                        "label": caption_label,
                        "caption_text": caption_text,
                        "caption_bbox": caption_candidate.get("caption_bbox"),
                        "caption_source": _normalize_text(str(caption_candidate.get("caption_source") or "")),
                        "caption_confidence": round(caption_confidence, 3) if caption_confidence else 0.0,
                        "citation_aliases": citation_aliases,
                        "subfigure_labels": subfigure_labels,
                        "subfigure_descriptions": subfigure_descriptions,
                        "association_method": "caption_anchor_region",
                        "association_confidence": round(min(0.99, 0.55 + caption_confidence * 0.35), 3),
                        "page_family": page_family_info.get("family", "scanned_unknown"),
                        "page_family_confidence": page_family_info.get("confidence", 0.0),
                        "layout_backend": str(resolved_layout_backend or "none"),
                        "context": context,
                        "area_ratio": round(area_ratio, 4),
                        "aspect_ratio": round(aspect_ratio, 4),
                        "context_title_match": True,
                        "title_score": title_score,
                        "ocr_char_count": ocr_char_count,
                        "ink_ratio": round(ink_ratio, 4),
                        "is_probable_false_positive": probable_false_positive,
                        "false_positive_reason": false_positive_reason,
                        "is_low_value": is_low_value,
                        "low_value_reason": low_value_reason,
                        "object_id": _build_media_object_id(
                            kind,
                            source_pdf_page,
                            file_name,
                            caption_label or title,
                        ),
                    }

                    if probable_false_positive or is_low_value:
                        filtered_out.append(item_record)
                        try:
                            os.remove(image_path)
                        except Exception:
                            pass
                        continue

                    matched_caption_signatures.add(candidate_signature)
                    caption_anchor_generated += 1
                    items.append(item_record)
                    items.extend(
                        _create_subfigure_items(
                            parent_item=item_record,
                            page_image=images[0],
                            media_dir=media_dir,
                            page_width=page_width,
                            page_height=page_height,
                            scaled_pdf_words=scaled_pdf_words,
                            gray_page=gray,
                        )
                    )

            try:
                images[0].close()
            except Exception:
                pass
    finally:
        try:
            if pdf_doc is not None:
                pdf_doc.close()
        except Exception:
            pass

    document_family_info = classify_document_family(
        [str(page.get("text_excerpt") or "") for page in page_analyses],
        source_name=os.path.basename(pdf_path),
    )
    for collection in (items, filtered_out):
        for item in collection:
            item["document_family"] = document_family_info.get("family", "scanned_unknown")
            item["document_family_confidence"] = document_family_info.get("confidence", 0.0)

    out_json = os.path.join(out_dir, f"{base}_media_only.json")
    out_txt = os.path.join(out_dir, f"{base}_media_only.txt")
    payload = {
        "file": os.path.basename(pdf_path),
        "source_path": pdf_path,
        "dpi": dpi,
        "pages_requested": target_pages,
        "document_family": document_family_info.get("family", "scanned_unknown"),
        "document_family_confidence": document_family_info.get("confidence", 0.0),
        "document_family_scores": document_family_info.get("scores", {}),
        "document_family_matched_terms": document_family_info.get("matched_terms", {}),
        "layout_backend_requested": requested_layout_backend,
        "layout_backends_observed": sorted(observed_layout_backends),
        "page_analyses": page_analyses,
        "stats": {
            "items_kept": len(items),
            "items_filtered_out": len(filtered_out),
            "items_detected_total": len(items) + len(filtered_out),
            "caption_candidates_detected": sum(
                int(page.get("caption_count", 0) or 0) for page in page_analyses
            ),
            "caption_anchor_items_generated": caption_anchor_generated,
            "layout_backends_observed": sorted(observed_layout_backends),
        },
        "items": items,
        "filtered_out": filtered_out,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_media_text(out_txt, items)

    return out_json


def main():
    parser = argparse.ArgumentParser(description="Lightweight figure/table extractor for PDFs")
    parser.add_argument("--pdf", required=True, help="PDF file")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument(
        "--pages",
        default="",
        help="1-based page list/ranges from the source PDF, e.g. 5,10,14-17",
    )
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
    out_json = extract_media(
        pdf_path=pdf_path,
        out_dir=out_dir,
        page_spec=args.pages,
        dpi=args.dpi,
        lang=args.lang,
        include_figure_ocr=args.figure_ocr,
        use_text_layer_first=not args.no_text_layer_first,
        layout_backend=args.layout_backend,
    )
    print(f"Media JSON: {out_json}")


if __name__ == "__main__":
    main()
