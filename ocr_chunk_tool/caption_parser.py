from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


BBox = Sequence[int]
TextBlockRecord = Tuple[Sequence[int], str]


CAPTION_LABEL_RE = re.compile(
    r"^(?P<kind>table|fig(?:ure)?\.?|diagram)\s*(?P<number>\d+[A-Za-z]?)\s*[:.\-)]?\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
CAPTION_LABEL_ANYWHERE_RE = re.compile(
    r"(?P<kind>table|fig(?:ure)?\.?|diagram)\s*(?P<number>\d+[A-Za-z]?)\s*[:.\-)]?\s*",
    re.IGNORECASE,
)
SUBFIGURE_LABEL_RE = re.compile(r"\(([a-z])\)")
SUBFIGURE_SEGMENT_RE = re.compile(
    r"\(([a-z])\)\s*([^()]+?)(?=(?:[,;]\s*\([a-z]\)|\s+and\s+\([a-z]\)|$))",
    re.IGNORECASE,
)
INLINE_REFERENCE_STOPWORDS = {
    "this",
    "that",
    "such",
    "the",
    "for",
    "since",
    "as",
    "is",
    "was",
    "are",
    "were",
    "hence",
    "where",
    "whose",
    "when",
    "because",
    "and",
    "or",
    "but",
    "with",
    "from",
    "of",
    "to",
    "in",
}


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalize_label(kind: str, number: str) -> str:
    number = _normalize_text(number)
    if "table" in kind.lower():
        return f"Table {number}"
    return f"Fig. {number}"


def _bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _horizontal_overlap_ratio(a: BBox, b: BBox) -> float:
    ax1, _, ax2, _ = [float(value) for value in a]
    bx1, _, bx2, _ = [float(value) for value in b]
    overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    width = max(ax2 - ax1, bx2 - bx1, 1.0)
    return overlap / width


def _vertical_gap(a: BBox, b: BBox) -> float:
    _, ay1, _, ay2 = [float(value) for value in a]
    _, by1, _, by2 = [float(value) for value in b]
    if by1 >= ay2:
        return by1 - ay2
    if ay1 >= by2:
        return ay1 - by2
    return 0.0


def build_citation_aliases(label: str, subfigure: str = "") -> List[str]:
    normalized = _normalize_text(label)
    if not normalized:
        return []
    aliases = {normalized}
    lower = normalized.lower()
    sub = _normalize_text(subfigure).lower()
    if lower.startswith("fig. "):
        number = normalized.split(" ", 1)[1]
        aliases.add(f"Fig {number}")
        aliases.add(f"Figure {number}")
        if sub:
            aliases.add(f"Fig. {number}{sub}")
            aliases.add(f"Fig {number}{sub}")
            aliases.add(f"Figure {number}{sub}")
            aliases.add(f"Fig. {number}({sub})")
            aliases.add(f"Fig {number}({sub})")
            aliases.add(f"Figure {number}({sub})")
    elif lower.startswith("table "):
        number = normalized.split(" ", 1)[1]
        aliases.add(f"Table {number}")
        if sub:
            aliases.add(f"Table {number}{sub}")
            aliases.add(f"Table {number}({sub})")
    return sorted(aliases)


def extract_subfigure_descriptions(caption_text: str) -> Dict[str, str]:
    normalized = _normalize_text(caption_text)
    if not normalized:
        return {}
    descriptions: Dict[str, str] = {}
    for label, description in SUBFIGURE_SEGMENT_RE.findall(normalized):
        clean_label = _normalize_text(label).lower()
        clean_description = _normalize_text(description)
        clean_description = re.sub(r"^(?:and|or)\s+", "", clean_description, flags=re.IGNORECASE)
        clean_description = clean_description.strip(" ,;:.")
        if clean_label and clean_description:
            descriptions[clean_label] = clean_description
    return descriptions


def _first_word(text: str) -> str:
    match = re.search(r"[A-Za-z]+", _normalize_text(text))
    if not match:
        return ""
    return match.group(0).lower()


def _looks_embedded_caption(prefix: str, rest: str, match_count: int) -> bool:
    normalized_prefix = _normalize_text(prefix)
    normalized_rest = _normalize_text(rest)
    if not normalized_rest:
        return False
    if not normalized_prefix:
        return True
    leading_char = normalized_rest[:1]
    if leading_char == ",":
        return False
    content_text = normalized_rest.lstrip(" .:-)")
    first_word = _first_word(content_text)
    if any(first_word.startswith(stopword) for stopword in INLINE_REFERENCE_STOPWORDS):
        return False
    if match_count > 1 and leading_char in ".:-)":
        return True
    if content_text.startswith("("):
        return True
    if re.search(r"[A-Z]{2,}", content_text):
        return True
    if len(content_text.split()) <= 8:
        return True
    return False


def _extract_caption_matches_from_line(
    line: str,
    *,
    next_line: str = "",
) -> List[Tuple[str, str, str, float, str]]:
    normalized_line = _normalize_text(line)
    if not normalized_line:
        return []

    matches = list(CAPTION_LABEL_ANYWHERE_RE.finditer(normalized_line))
    if not matches:
        return []

    extracted: List[Tuple[str, str, str, float, str]] = []
    for index, match in enumerate(matches):
        kind_raw = match.group("kind")
        number_raw = match.group("number")
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized_line)
        segment = normalized_line[start:end].strip(" ,;")
        label = _normalize_label(kind_raw, number_raw)
        kind = "table" if "table" in kind_raw.lower() else "figure"
        prefix = normalized_line[:start]
        rest = normalized_line[match.end():end].strip()
        is_line_start = start == 0
        if not is_line_start and not _looks_embedded_caption(prefix, rest, len(matches)):
            continue

        caption_text = segment
        if rest and len(rest.split()) < 2 and next_line and not CAPTION_LABEL_RE.match(next_line):
            caption_text = f"{segment} {next_line}".strip()
        elif not rest and next_line and not CAPTION_LABEL_RE.match(next_line):
            caption_text = f"{segment} {next_line}".strip()

        confidence = 0.92 if is_line_start else (0.86 if len(matches) > 1 else 0.78)
        source = "text_block" if is_line_start else "text_block_embedded"
        extracted.append((kind, label, _normalize_text(caption_text), confidence, source))
    return extracted


def extract_caption_candidates(
    text_block_records: Iterable[TextBlockRecord],
    *,
    page_number: int,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []

    for bbox, raw_text in text_block_records:
        lines = [_normalize_text(line) for line in str(raw_text or "").splitlines()]
        lines = [line for line in lines if line]
        for index, line in enumerate(lines):
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            matches = _extract_caption_matches_from_line(line, next_line=next_line)
            for kind, label, caption_text, confidence, source in matches:
                candidates.append(
                    {
                        "page_number": int(page_number),
                        "type": kind,
                        "label": label,
                        "caption_text": _normalize_text(caption_text),
                        "caption_bbox": [int(value) for value in bbox],
                        "caption_source": source,
                        "caption_confidence": round(confidence, 3),
                        "subfigure_labels": sorted(set(SUBFIGURE_LABEL_RE.findall(caption_text.lower()))),
                        "subfigure_descriptions": extract_subfigure_descriptions(caption_text),
                        "citation_aliases": build_citation_aliases(label),
                    }
                )

    deduped: List[Dict[str, object]] = []
    seen = set()
    for candidate in candidates:
        key = (
            _normalize_text(candidate.get("type", "")).lower(),
            _normalize_text(candidate.get("label", "")).lower(),
            tuple(candidate.get("caption_bbox") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    return deduped


def match_caption_to_block(
    *,
    block_kind: str,
    block_bbox: BBox,
    caption_candidates: Sequence[Dict[str, object]],
    page_width: int,
    page_height: int,
) -> Optional[Dict[str, object]]:
    best_match: Optional[Dict[str, object]] = None
    best_score = -999.0
    target_kind = _normalize_text(block_kind).lower()

    for candidate in caption_candidates:
        if _normalize_text(candidate.get("type", "")).lower() != target_kind:
            continue
        caption_bbox = candidate.get("caption_bbox")
        if not isinstance(caption_bbox, list) or len(caption_bbox) != 4:
            continue

        overlap_score = _horizontal_overlap_ratio(block_bbox, caption_bbox)
        gap = _vertical_gap(block_bbox, caption_bbox)
        _, block_cy = _bbox_center(block_bbox)
        _, caption_cy = _bbox_center(caption_bbox)
        caption_below = caption_cy >= block_cy

        score = float(candidate.get("caption_confidence", 0.5))
        score += overlap_score * 0.35
        score -= min(gap / max(float(page_height), 1.0), 1.0)

        if target_kind == "figure" and caption_below:
            score += 0.12
        if target_kind == "table" and not caption_below:
            score += 0.12

        if gap > (page_height * 0.28):
            score -= 0.45

        if score > best_score:
            best_score = score
            best_match = {
                "candidate": candidate,
                "association_method": "caption_proximity",
                "association_confidence": round(max(0.0, min(score, 0.99)), 3),
            }

    return best_match
