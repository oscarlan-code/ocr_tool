from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple


FAMILY_SCORES = (
    "academic_paper",
    "manual",
    "datasheet",
    "electrical_schematic",
    "pid_or_hydraulic",
    "report_or_slide",
)


ACADEMIC_TERMS = (
    "abstract",
    "introduction",
    "references",
    "keywords",
    "received",
    "accepted",
    "microelectronics reliability",
    "ieee",
    "journal",
    "doi",
    "et al",
)

MANUAL_TERMS = (
    "owner's manual",
    "owners manual",
    "operator manual",
    "safety information",
    "maintenance schedule",
    "controls & features",
    "troubleshooting",
    "before operation",
)

DATASHEET_TERMS = (
    "absolute maximum ratings",
    "electrical characteristics",
    "recommended operating conditions",
    "pin configuration",
    "application information",
    "typical application",
    "package outline",
    "ordering information",
)

REPORT_TERMS = (
    "executive summary",
    "agenda",
    "conclusion",
    "appendix",
    "background",
    "findings",
    "recommendations",
)


ACADEMIC_CAPTION_RE = re.compile(r"\b(?:fig(?:ure)?\.?\s*\d+|table\s*\d+)\b", re.IGNORECASE)
ELECTRICAL_DESIGNATOR_RE = re.compile(
    r"\b(?:R|C|U|Q|D|L|J|TP|LED|SW|IC|CN|JP)\d{1,4}[A-Z]?\b",
    re.IGNORECASE,
)
ELECTRICAL_SIGNAL_RE = re.compile(
    r"\b(?:gnd|vcc|vin|vout|avcc|dvcc|5v|3v3|12v|led|audio|speaker|amp(?:lifier)?)\b",
    re.IGNORECASE,
)
PID_TAG_RE = re.compile(
    r"\b(?:PIT|PI|PCV|PSV|LIT|TIT|LSH|LSL|FT|PT|TT|ERC)\b[-\s]?\d*",
    re.IGNORECASE,
)
HYDRAULIC_TERM_RE = re.compile(
    r"\b(?:hydraulic|accumulator|barg|valve cabinet|loading arm|pump|cylinder|manifold)\b",
    re.IGNORECASE,
)
SCHEMATIC_TERM_RE = re.compile(
    r"\b(?:schematic|circuit diagram|block diagram|wiring diagram|pcb|power supply|driver stage)\b",
    re.IGNORECASE,
)


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").split()).strip()


def _score_term_hits(text: str, terms: Iterable[str]) -> Tuple[int, List[str]]:
    score = 0
    matched: List[str] = []
    for term in terms:
        if term in text:
            score += 2
            matched.append(term)
    return score, matched


def _count_matches(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _safe_excerpt(texts: List[str]) -> str:
    joined = " ".join(texts)
    joined = _normalize_text(joined)
    if len(joined) <= 500:
        return joined
    return joined[:497].rstrip() + "..."


def classify_document_family(
    text_blocks: Iterable[str],
    *,
    source_name: str = "",
) -> Dict[str, object]:
    texts = [_normalize_text(text) for text in text_blocks if _normalize_text(text)]
    corpus = "\n".join(texts).lower()
    source = _normalize_text(source_name).lower()

    if not corpus and not source:
        return {
            "family": "scanned_unknown",
            "confidence": 0.15,
            "scores": {family: 0 for family in FAMILY_SCORES},
            "matched_terms": {},
            "text_excerpt": "",
        }

    scores = {family: 0 for family in FAMILY_SCORES}
    matched_terms: Dict[str, List[str]] = {family: [] for family in FAMILY_SCORES}

    academic_score, academic_terms = _score_term_hits(corpus, ACADEMIC_TERMS)
    scores["academic_paper"] += academic_score
    matched_terms["academic_paper"].extend(academic_terms)
    academic_caption_hits = _count_matches(ACADEMIC_CAPTION_RE, corpus)
    if academic_caption_hits:
        scores["academic_paper"] += min(academic_caption_hits, 8)
        matched_terms["academic_paper"].append(f"caption_refs:{academic_caption_hits}")

    manual_score, manual_terms = _score_term_hits(corpus, MANUAL_TERMS)
    scores["manual"] += manual_score
    matched_terms["manual"].extend(manual_terms)

    datasheet_score, datasheet_terms = _score_term_hits(corpus, DATASHEET_TERMS)
    scores["datasheet"] += datasheet_score
    matched_terms["datasheet"].extend(datasheet_terms)

    report_score, report_terms = _score_term_hits(corpus, REPORT_TERMS)
    scores["report_or_slide"] += report_score
    matched_terms["report_or_slide"].extend(report_terms)

    electrical_designator_hits = _count_matches(ELECTRICAL_DESIGNATOR_RE, corpus)
    if electrical_designator_hits:
        scores["electrical_schematic"] += min(10, electrical_designator_hits // 2 + 2)
        matched_terms["electrical_schematic"].append(
            f"designators:{electrical_designator_hits}"
        )
    electrical_signal_hits = _count_matches(ELECTRICAL_SIGNAL_RE, corpus)
    if electrical_signal_hits:
        scores["electrical_schematic"] += min(8, electrical_signal_hits // 2 + 1)
        matched_terms["electrical_schematic"].append(f"signals:{electrical_signal_hits}")
    schematic_hits = _count_matches(SCHEMATIC_TERM_RE, corpus)
    if schematic_hits:
        scores["electrical_schematic"] += min(6, schematic_hits * 2)
        matched_terms["electrical_schematic"].append(f"schematic_terms:{schematic_hits}")

    pid_hits = _count_matches(PID_TAG_RE, corpus)
    if pid_hits:
        scores["pid_or_hydraulic"] += min(12, pid_hits // 2 + 3)
        matched_terms["pid_or_hydraulic"].append(f"pid_tags:{pid_hits}")
    hydraulic_hits = _count_matches(HYDRAULIC_TERM_RE, corpus)
    if hydraulic_hits:
        scores["pid_or_hydraulic"] += min(10, hydraulic_hits * 2)
        matched_terms["pid_or_hydraulic"].append(f"hydraulic_terms:{hydraulic_hits}")

    if source.endswith(".pdf") and "datasheet" in source:
        scores["datasheet"] += 3
        matched_terms["datasheet"].append("filename:datasheet")
    if "manual" in source:
        scores["manual"] += 3
        matched_terms["manual"].append("filename:manual")
    if any(token in source for token in ("schematic", "circuit", "driver", "amp")):
        scores["electrical_schematic"] += 2
        matched_terms["electrical_schematic"].append("filename:electrical")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_family, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else 0

    if top_score <= 1:
        family = "scanned_unknown"
        confidence = 0.2
    else:
        family = top_family
        margin = max(top_score - runner_up_score, 0)
        confidence = min(0.98, 0.35 + (top_score / 20.0) + (margin / 25.0))

    return {
        "family": family,
        "confidence": round(confidence, 3),
        "scores": scores,
        "matched_terms": {
            key: value for key, value in matched_terms.items() if value
        },
        "text_excerpt": _safe_excerpt(texts),
    }


def classify_page_family(text: str, *, source_name: str = "") -> Dict[str, object]:
    return classify_document_family([text], source_name=source_name)
