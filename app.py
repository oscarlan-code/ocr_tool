import os
import json
import gc
import queue
import re
import threading
import traceback
import subprocess
import tkinter as tk
from shutil import which
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

try:
    import pdfplumber
    from pdf2image import convert_from_path, pdfinfo_from_path
    import pytesseract
except Exception as exc:
    pdfplumber = None
    convert_from_path = None
    pdfinfo_from_path = None
    pytesseract = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

try:
    import numpy as np
    import cv2
except Exception as exc:
    np = None
    cv2 = None
    _CV_ERROR = exc
else:
    _CV_ERROR = None

try:
    from rapidfuzz import process as fuzz_process
except Exception as exc:
    fuzz_process = None
    _FUZZ_ERROR = exc
else:
    _FUZZ_ERROR = None

try:
    from paddleocr import PPStructure, PaddleOCR
except Exception:
    PPStructure = None
    PaddleOCR = None

try:
    import requests
except Exception as exc:
    requests = None
    _REQUESTS_ERROR = exc
else:
    _REQUESTS_ERROR = None

try:
    import layoutparser as lp
except Exception:
    lp = None

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

DEFAULT_LANG = "kor+eng"
DEFAULT_DPI = 300
TEXT_MIN_CHARS = 50
QUALITY_REPORT_NAME = "quality_report.txt"

DEFAULT_TESS_OEM = "1"
DEFAULT_TESS_PSM_TEXT = "6"
DEFAULT_TESS_PSM_TABLE = "4"
DEFAULT_RETRY_PSM = "11"
DEFAULT_CONF_THRESHOLD = "60"

DEFAULT_PREPROCESS_MODE = "auto"  # auto | gray | bw
DEFAULT_INK_RATIO_THRESHOLD = "0.25"

DEFAULT_MIN_CHARS = "20"
DEFAULT_MIN_SCORE = "40"
DEFAULT_HANGUL_RATIO = "0.12"
DEFAULT_LATIN_RATIO = 0.20
DEFAULT_LANG_SWITCH_MARGIN = "5"
DEFAULT_RESCUE_SHORT_RATIO = "0.60"
DEFAULT_RESCUE_PSM_TEXT = "6"
DEFAULT_RESCUE_PSM_TABLE = "4"
DEFAULT_REMOVE_TOC = True
DEFAULT_TOC_MAX_PAGES = "5"
DEFAULT_TOC_MIN_KEEP = "8"
DEFAULT_USE_DEEPSEEK = True
DEFAULT_DEEPSEEK_BASE = "https://api.deepseek.com/v1/chat/completions"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_SUMMARIZE_TABLES = True
DEFAULT_SUMMARIZE_FIGURES = True
DEFAULT_MEM_CLEANUP = True
DEFAULT_CLEAN_OUTPUT = True
DEFAULT_LLM_CLEAN = False
DEFAULT_LLM_SKIP_TABLES = True
DEFAULT_AUTO_MEDIA_SCAN = True
DEFAULT_MEDIA_AREA_RATIO = "0.12"
DEFAULT_MEDIA_SCAN_DPI = "120"
DEFAULT_PREVIEW_MEDIA = True

VOCAB = [
    "점검", "보수", "유압", "윤활", "냉각", "주축", "이송축", "설정",
    "압력", "온도", "모터", "베어링", "필터", "릴레이", "스위치",
    "스핀들", "토크", "오일", "공압", "클램프", "언클램프", "ATC",
    "PMC", "파라메타", "리미트", "센서", "정지", "운전", "경고",
    "절삭유", "윤활유", "체결", "교환", "정기", "매일", "매주", "매월",
    "반기", "정도", "레벨", "베드", "테이블", "기어", "기어박스",
    "회전", "원점", "좌표", "축", "Z축", "X축", "Y축", "W축",
    "인디게이터", "테스트", "바", "기계", "작업", "안전",
    "rpm", "mm", "bar", "N-m",
]


@dataclass
class Block:
    kind: str
    bbox: tuple


_LAYOUT_ENGINE = None
_LAYOUT_ENGINE_KIND = None
_TABLE_OCR_ENGINE = None


def _ensure_deps():
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing Python dependencies. Install with: pip install -r requirements.txt"
        ) from _IMPORT_ERROR
    if _CV_ERROR is not None:
        raise RuntimeError(
            "Missing OpenCV/numpy. Install with: pip install -r requirements.txt"
        ) from _CV_ERROR
    if _FUZZ_ERROR is not None:
        raise RuntimeError(
            "Missing rapidfuzz. Install with: pip install -r requirements.txt"
        ) from _FUZZ_ERROR


def _ensure_output_dir(in_path, out_dir):
    if out_dir:
        return out_dir
    if os.path.isdir(in_path):
        return in_path
    if os.path.isfile(in_path):
        return os.path.dirname(in_path)
    return ""


def is_pdf(path):
    return os.path.isfile(path) and path.lower().endswith(".pdf")


def _detect_poppler_path():
    env_path = os.environ.get("POPPLER_PATH")
    candidates = [
        env_path,
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/opt/local/bin",
    ]
    for path in candidates:
        if not path:
            continue
        if os.path.isfile(os.path.join(path, "pdftoppm")):
            return path
    exe = which("pdftoppm")
    if exe:
        return os.path.dirname(exe)
    return None


def _load_dotenv():
    # Lightweight .env loader (no external dependency).
    candidates = [
        os.path.join(os.getcwd(), ".env"),
        os.path.join(os.path.dirname(__file__), ".env"),
    ]
    loaded = False
    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            loaded = True
        except Exception:
            continue
    return loaded


def _safe_int(value, fallback):
    try:
        return int(value)
    except Exception:
        return fallback


def _safe_float(value, fallback):
    try:
        return float(value)
    except Exception:
        return fallback


def pdf_has_images(pdf_path):
    exe = which("pdfimages")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "-list", pdf_path],
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout or "").strip().splitlines()
        # pdfimages -list prints a header plus table rows when images exist.
        # If only header lines exist, treat as no images.
        data_lines = [line for line in output if line.strip()][2:]
        return len(data_lines) > 0
    except Exception:
        return None


def scan_large_media(pdf_path, poppler_path, dpi=120, min_area_ratio=0.12):
    if convert_from_path is None or pdfinfo_from_path is None or cv2 is None or np is None:
        return False
    engine, kind = get_layout_engine()
    if engine is None:
        return False
    has_images = pdf_has_images(pdf_path)
    if has_images is False:
        return False
    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        total = int(info.get("Pages", 0))
    except Exception:
        return False
    if total <= 0:
        return False
    for page_num in range(1, total + 1):
        try:
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                poppler_path=poppler_path,
                first_page=page_num,
                last_page=page_num,
                fmt="png",
            )
        except Exception:
            continue
        if not images:
            continue
        try:
            img = np.array(images[0])
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception:
            try:
                images[0].close()
            except Exception:
                pass
            continue
        h, w = img_bgr.shape[:2]
        page_area = max(w * h, 1)
        blocks = detect_layout_blocks(img_bgr)
        try:
            images[0].close()
        except Exception:
            pass
        for block in blocks:
            if block.kind not in ("table", "figure"):
                continue
            x1, y1, x2, y2 = block.bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if (area / page_area) >= min_area_ratio:
                return True
    return False


def _page_stats(text):
    total = len(text)
    if total == 0:
        return {
            "total": 0,
            "hangul": 0,
            "latin": 0,
            "digits": 0,
            "other": 0,
            "short_lines": 0,
            "lines": 0,
            "score": 0,
            "status": "BAD",
        }

    hangul = 0
    latin = 0
    digits = 0
    whitespace = 0

    for ch in text:
        if "\uac00" <= ch <= "\ud7a3":
            hangul += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
        elif ch.isdigit():
            digits += 1
        elif ch.isspace():
            whitespace += 1

    other = total - (hangul + latin + digits + whitespace)

    lines = [ln for ln in text.splitlines() if ln.strip()]
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 6)
    short_ratio = short_lines / len(lines) if lines else 1.0
    other_ratio = other / total

    score = 100 - (other_ratio * 70 + short_ratio * 30)
    if score < 0:
        score = 0
    if score >= 75:
        status = "OK"
    elif score >= 50:
        status = "WARN"
    else:
        status = "BAD"

    return {
        "total": total,
        "hangul": hangul,
        "latin": latin,
        "digits": digits,
        "other": other,
        "short_lines": short_lines,
        "lines": len(lines),
        "score": int(round(score)),
        "status": status,
    }


def _short_line_ratio(text):
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0
    short_lines = sum(1 for ln in lines if len(ln.strip()) < 6)
    return short_lines / len(lines)


def _line_is_noise(line):
    if not line:
        return True
    stripped = line.strip()
    if not stripped:
        return True
    length = len(stripped)
    if length == 0:
        return True

    hangul = sum(1 for ch in stripped if "\uac00" <= ch <= "\ud7a3")
    latin = sum(1 for ch in stripped if ch.isalpha() and ch.isascii())
    digits = sum(1 for ch in stripped if ch.isdigit())
    alnum = hangul + latin + digits
    other = length - alnum

    # Drop ultra-short lines with almost no alnum content.
    if length < 6 and alnum < 3:
        return True

    # Drop mostly-symbol lines (tables/line art).
    if length > 0 and (alnum / length) < 0.30:
        return True

    # Drop repeated characters like "cccccccc" or "----".
    if length >= 8:
        unique = set(stripped)
        if len(unique) <= 2:
            return True

    # Drop pure punctuation/leader lines.
    if re.fullmatch(r"[\W_]+", stripped):
        return True
    if re.fullmatch(r"(\.{2,}|…{2,}|[-=]{3,})", stripped):
        return True

    return False


def clean_output_text(text):
    lines = []
    for line in text.splitlines():
        if _line_is_noise(line):
            continue
        lines.append(line.strip())
    # Collapse multiple blank lines
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _x_overlap_ratio(bbox, target_bbox):
    x1, _, x2, _ = bbox
    tx1, _, tx2, _ = target_bbox
    inter = max(0, min(x2, tx2) - max(x1, tx1))
    tw = max(tx2 - tx1, 1)
    return inter / tw


def _extract_caption_title(context_text, ocr_text=""):
    caption_re = re.compile(r"^(table|fig\\.?|figure|표|그림|도표)", re.IGNORECASE)
    if context_text:
        for line in context_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if caption_re.search(stripped):
                return stripped
        for line in context_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    if ocr_text:
        for line in ocr_text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if caption_re.search(stripped):
                return stripped
        for line in ocr_text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped
    return ""


def _collect_context(text_blocks, target_bbox, max_above=2, max_below=2, min_x_overlap=0.2):
    if not text_blocks:
        return ""
    tx1, ty1, tx2, ty2 = target_bbox
    above = []
    below = []
    caption_re = re.compile(r"^(table|fig\\.?|figure|표|그림|도표)", re.IGNORECASE)
    for bbox, text in text_blocks:
        if not text:
            continue
        x1, y1, x2, y2 = bbox
        overlap = _x_overlap_ratio(bbox, target_bbox)
        if y2 <= ty1:
            above.append((ty1 - y2, overlap, text))
        elif y1 >= ty2:
            below.append((y1 - ty2, overlap, text))

    def _pick(blocks, limit, prefer_overlap=True):
        if not blocks:
            return []
        filtered = [b for b in blocks if b[1] >= min_x_overlap] if prefer_overlap else blocks
        if not filtered:
            filtered = blocks
        return sorted(filtered, key=lambda x: x[0])[:limit]

    above = _pick(above, max_above)
    below = _pick(below, max_below)
    context_lines = [t for _, _, t in reversed(above)] + [t for _, _, t in below]

    # Prefer nearby captions if present.
    tcy = (ty1 + ty2) / 2.0
    caption_lines = []
    for bbox, text in text_blocks:
        if not text:
            continue
        if caption_re.search(text.strip()):
            _, y1, _, y2 = bbox
            cy = (y1 + y2) / 2.0
            caption_lines.append((abs(cy - tcy), text.strip()))
    caption_lines = [t for _, t in sorted(caption_lines, key=lambda x: x[0])[:2]]
    if caption_lines:
        # Avoid duplicates while preserving order.
        seen = set()
        merged = []
        for line in caption_lines + context_lines:
            if line in seen:
                continue
            seen.add(line)
            merged.append(line)
        context_lines = merged

    if context_lines:
        return "\n".join(context_lines)

    # Fallback: closest by vertical center
    nearest = []
    for bbox, text in text_blocks:
        if not text:
            continue
        x1, y1, x2, y2 = bbox
        cy = (y1 + y2) / 2.0
        nearest.append((abs(cy - tcy), text))
    nearest = sorted(nearest, key=lambda x: x[0])[:max_above + max_below]
    return "\n".join([t for _, t in nearest if t])


def _llm_clean_prompt(text, skip_tables_figures=False):
    trimmed = _truncate_text(text, max_chars=3500)
    if not trimmed:
        return ""
    rules = [
        "Fix obvious OCR errors and spacing only.",
        "DO NOT add, remove, or invent content.",
        "Keep all numbers, units, and symbols exactly.",
        "If unsure, keep the original text.",
    ]
    if skip_tables_figures:
        rules.append(
            "Remove any table-like or figure-like fragments. "
            "Drop lines that look like columns/rows, repeated separators, "
            "or labels like 'Table', '표', 'Figure', 'Fig.', '그림', 'Diagram'. "
            "Keep narrative sentences and step-by-step instructions."
        )
    rules_text = "\n".join(f"- {r}" for r in rules)
    return (
        "You are cleaning OCR text from a technical manual.\n"
        "Rules:\n"
        f"{rules_text}\n"
        "Return only the cleaned text.\n\n"
        f"OCR text:\n{trimmed}"
    )


def llm_clean_text(text, api_key, base_url, model, chunk_size=2500, skip_tables_figures=False):
    if not text:
        return "", 0, 0
    chunks = []
    buff = []
    count = 0
    for line in text.splitlines():
        if len("\n".join(buff)) + len(line) + 1 > chunk_size and buff:
            chunks.append("\n".join(buff))
            buff = []
        buff.append(line)
    if buff:
        chunks.append("\n".join(buff))

    cleaned_chunks = []
    calls = 0
    errors = 0
    for chunk in chunks:
        prompt = _llm_clean_prompt(chunk, skip_tables_figures=skip_tables_figures)
        if not prompt:
            continue
        out, err = deepseek_summarize(prompt, api_key, base_url, model)
        if err:
            errors += 1
            cleaned_chunks.append(chunk)
        else:
            calls += 1
            cleaned_chunks.append(out)
    return "\n\n".join(cleaned_chunks).strip(), calls, errors


def _truncate_text(text, max_chars=4000):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _build_table_prompt(text, context_text=""):
    trimmed = _truncate_text(text, max_chars=4000)
    if not trimmed:
        return ""
    context = _truncate_text(context_text or "", max_chars=1200)
    return (
        "Summarize this table into concise bullet points. Preserve units and key numbers. "
        "If OCR is partial or messy, say so briefly.\n\n"
        f"Table OCR:\n{trimmed}\n\n"
        f"Context:\n{context}" if context else f"Table OCR:\n{trimmed}"
    )


def _build_figure_prompt(text, context_text=""):
    trimmed = _truncate_text(text, max_chars=2000)
    context = _truncate_text(context_text or "", max_chars=1200)
    if not trimmed and not context:
        return ""
    return (
        "Describe the figure based only on the following OCR text labels or captions. "
        "If there is not enough information, say 'Figure description unavailable from OCR text.'\n\n"
        f"OCR text:\n{trimmed}\n\n"
        f"Context:\n{context}" if context else f"OCR text:\n{trimmed}"
    )


def deepseek_summarize(prompt, api_key, base_url, model, timeout=30):
    if not prompt:
        return "", "empty_prompt"
    if not api_key or not base_url or not model:
        return "", "missing_config"
    if requests is None:
        return "", "requests_missing"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a technical documentation assistant. Summarize precisely and briefly.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    try:
        response = requests.post(base_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return content, ""
    except Exception as exc:
        return "", str(exc)


def _is_toc_line(line):
    line = line.strip()
    if not line:
        return False
    score = 0
    if re.search(r"\bMA\d+[-–]\d+\b", line):
        score += 2
    if re.search(r"(\\.\\.{2,}|…{2,})", line):
        score += 1
    if re.match(r"^\d+(\.\d+)*", line):
        score += 1
    if re.search(r"\bPART\s*\d+\b", line, re.IGNORECASE):
        score += 1
    if re.search(r"(table of contents|목차|contents)", line, re.IGNORECASE):
        score += 2
    if re.search(r"(.)\1{5,}", line):
        score += 1
    return score >= 2


def _strip_toc_page(text, in_toc, non_toc_run, min_keep_run, max_pages, page_index):
    if not in_toc or page_index >= max_pages:
        return text, False, non_toc_run, 0

    lines = text.splitlines()
    new_lines = []
    removed = 0
    for line in lines:
        if in_toc:
            if _is_toc_line(line):
                removed += 1
                continue
            if not line.strip():
                removed += 1
                continue
            non_toc_run += 1
            if non_toc_run >= min_keep_run:
                in_toc = False
                new_lines.append(line)
            else:
                removed += 1
        else:
            new_lines.append(line)

    return "\n".join(new_lines), in_toc, non_toc_run, removed


def _script_ratios(text):
    total = 0
    hangul = 0
    latin = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if "\uac00" <= ch <= "\ud7a3":
            hangul += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
    if total == 0:
        return 0.0, 0.0
    return hangul / total, latin / total


def _choose_lang(text, hangul_ratio_threshold):
    hangul_ratio, latin_ratio = _script_ratios(text)
    if hangul_ratio >= hangul_ratio_threshold:
        return "kor"
    if latin_ratio >= DEFAULT_LATIN_RATIO:
        return "eng"
    return None


def _maybe_invert(binary):
    if binary is None:
        return binary
    total = binary.size
    if total == 0:
        return binary
    zeros = int(np.sum(binary == 0))
    if zeros < total * 0.01:
        return cv2.bitwise_not(binary)
    return binary


def _ink_ratio(binary):
    if binary is None or binary.size == 0:
        return 1.0
    return float(np.mean(binary == 0))


def _estimate_skew_angle(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=120,
        minLineLength=max(gray.shape[1] // 3, 80),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
        if -30 < ang < 30:
            angles.append(ang)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _rotate_image(img, angle):
    (h, w) = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_image(pil_img):
    img_rgb = np.array(pil_img)
    if img_rgb.ndim == 2:
        gray = img_rgb
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=25, sigmaY=25)
    norm = cv2.divide(gray, bg, scale=255)
    norm = cv2.fastNlMeansDenoising(norm, None, h=10, templateWindowSize=7, searchWindowSize=21)

    angle = _estimate_skew_angle(norm)
    if abs(angle) > 0.2:
        img_bgr = _rotate_image(img_bgr, angle)
        norm = _rotate_image(norm, angle)

    bw = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    bw = _maybe_invert(bw)

    return img_bgr, norm, bw, angle


def choose_preprocessed(gray, bw, mode, ink_ratio_threshold):
    mode = (mode or "auto").lower()
    if mode == "gray":
        return gray, "gray"
    if mode == "bw":
        return bw, "bw"
    ratio = _ink_ratio(bw)
    if ratio > ink_ratio_threshold:
        return gray, "gray"
    return bw, "bw"


def get_layout_engine():
    global _LAYOUT_ENGINE, _LAYOUT_ENGINE_KIND
    if _LAYOUT_ENGINE is not None:
        return _LAYOUT_ENGINE, _LAYOUT_ENGINE_KIND

    if PPStructure is not None:
        try:
            _LAYOUT_ENGINE = PPStructure(show_log=False)
            _LAYOUT_ENGINE_KIND = "paddle"
            return _LAYOUT_ENGINE, _LAYOUT_ENGINE_KIND
        except Exception:
            _LAYOUT_ENGINE = None
            _LAYOUT_ENGINE_KIND = None

    if lp is not None:
        try:
            model = lp.Detectron2LayoutModel(
                "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
            )
            _LAYOUT_ENGINE = model
            _LAYOUT_ENGINE_KIND = "layoutparser"
        except Exception:
            _LAYOUT_ENGINE = None
            _LAYOUT_ENGINE_KIND = None

    return _LAYOUT_ENGINE, _LAYOUT_ENGINE_KIND


def get_table_ocr_engine():
    global _TABLE_OCR_ENGINE
    if _TABLE_OCR_ENGINE is not None:
        return _TABLE_OCR_ENGINE
    if PaddleOCR is None:
        return None
    try:
        _TABLE_OCR_ENGINE = PaddleOCR(use_angle_cls=False, lang="korean")
    except Exception:
        _TABLE_OCR_ENGINE = None
    return _TABLE_OCR_ENGINE


def detect_layout_blocks(img_bgr):
    engine, kind = get_layout_engine()
    h, w = img_bgr.shape[:2]
    if engine is None:
        return [Block(kind="text", bbox=(0, 0, w, h))]

    blocks = []
    if kind == "paddle":
        try:
            result = engine(img_bgr)
            for item in result:
                bbox = item.get("bbox")
                if not bbox:
                    continue
                btype = item.get("type", "text")
                if btype in ("figure", "figure_caption"):
                    mapped = "figure"
                elif btype in ("table",):
                    mapped = "table"
                else:
                    mapped = "text"
                blocks.append(Block(kind=mapped, bbox=tuple(bbox)))
        except Exception:
            blocks = [Block(kind="text", bbox=(0, 0, w, h))]
    elif kind == "layoutparser":
        try:
            layout = engine.detect(img_bgr)
            for b in layout:
                bbox = (int(b.block.x_1), int(b.block.y_1), int(b.block.x_2), int(b.block.y_2))
                label = str(b.type).lower()
                if "table" in label:
                    mapped = "table"
                elif "figure" in label:
                    mapped = "figure"
                else:
                    mapped = "text"
                blocks.append(Block(kind=mapped, bbox=bbox))
        except Exception:
            blocks = [Block(kind="text", bbox=(0, 0, w, h))]

    if not blocks:
        blocks = [Block(kind="text", bbox=(0, 0, w, h))]

    blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return blocks


def crop_image(img, bbox):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return img
    return img[y1:y2, x1:x2]


def tesseract_config(oem, psm):
    return f"--oem {oem} --psm {psm}"


def tesseract_confidence(img, lang, config, conf_threshold):
    try:
        data = pytesseract.image_to_data(
            img,
            lang=lang,
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confs = []
        for c in data.get("conf", []):
            try:
                confs.append(int(float(c)))
            except Exception:
                continue
        if not confs:
            return 0, 0
        low = sum(1 for c in confs if c < conf_threshold)
        avg = int(round(sum(confs) / len(confs)))
        return avg, low
    except Exception:
        return 0, 0


def ocr_tesseract(img, lang, oem, psm, conf_threshold, retry_psm=None):
    config = tesseract_config(oem, psm)
    text = pytesseract.image_to_string(img, lang=lang, config=config)
    avg_conf, low = tesseract_confidence(img, lang, config, conf_threshold)

    if retry_psm and avg_conf < conf_threshold:
        retry_config = tesseract_config(oem, retry_psm)
        retry_text = pytesseract.image_to_string(img, lang=lang, config=retry_config)
        retry_conf, retry_low = tesseract_confidence(img, lang, retry_config, conf_threshold)
        if retry_conf > avg_conf:
            return retry_text, retry_conf, retry_low, True

    return text, avg_conf, low, False


def ocr_table_paddle(img_bgr):
    engine = get_table_ocr_engine()
    if engine is None:
        return "", 0

    try:
        result = engine.ocr(img_bgr, cls=False)
        lines = []
        confs = []
        for row in result:
            for box, (text, score) in row:
                y = sum(p[1] for p in box) / 4.0
                lines.append((y, text))
                confs.append(score)
        lines.sort(key=lambda x: x[0])
        joined = "\n".join(t[1] for t in lines)
        avg_conf = int(round(sum(confs) / len(confs) * 100)) if confs else 0
        return joined, avg_conf
    except Exception:
        return "", 0


def should_correct_token(token):
    if len(token) < 2:
        return False
    if any(ch.isdigit() for ch in token):
        return False
    for ch in token:
        if "\uac00" <= ch <= "\ud7a3" or ch.isalpha():
            return True
    return False


def correct_text(text, vocab, threshold=88):
    if fuzz_process is None:
        return text

    parts = re.findall(r"\s+|\S+", text)
    corrected = []
    for part in parts:
        if part.isspace():
            corrected.append(part)
            continue
        if not should_correct_token(part):
            corrected.append(part)
            continue
        if part in vocab:
            corrected.append(part)
            continue
        match = fuzz_process.extractOne(part, vocab)
        if match and match[1] >= threshold:
            corrected.append(match[0])
        else:
            corrected.append(part)
    return "".join(corrected)


def extract_text_pdf_pages(pdf_path, on_page=None):
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append(text)
            if on_page:
                on_page(idx, total)
    return pages


def ocr_pdf_pages(
    pdf_path,
    lang,
    dpi,
    oem,
    psm_text,
    psm_table,
    conf_threshold,
    retry_psm,
    use_layout,
    use_table_ocr,
    use_vocab,
    preprocess_mode,
    ink_ratio_threshold,
    use_fallback,
    auto_lang,
    hangul_ratio_threshold,
    lang_switch_margin,
    skip_low_text,
    min_chars,
    min_score,
    rescue_short_ratio,
    rescue_psm_text,
    rescue_psm_table,
    remove_toc,
    toc_max_pages,
    toc_min_keep,
    capture_media=False,
    media_dir=None,
    deepseek_key=None,
    deepseek_base=None,
    deepseek_model=None,
    summarize_tables=False,
    summarize_figures=False,
    mem_cleanup=False,
    poppler_path=None,
    on_page=None,
):
    pages = []
    page_confs = []
    page_low_conf = []
    page_retry = []
    page_fallbacks = []
    page_lang_switches = []
    page_stats_raw = []
    page_skipped = []
    page_rescues = []
    page_rescue_reasons = []
    page_toc_removed = []
    toc_active = True
    toc_non_toc_run = 0
    media_items = []
    deepseek_calls = 0
    deepseek_errors = 0

    total = 0
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
    except Exception:
        total = 0

    if total == 0 and pdfinfo_from_path is not None:
        try:
            info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
            total = int(info.get("Pages", 0))
        except Exception:
            total = 0

    for page_num in range(1, total + 1):
        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_num,
            last_page=page_num,
            poppler_path=poppler_path,
        )
        if not images:
            pages.append("")
            page_confs.append(0)
            page_low_conf.append(0)
            page_retry.append(False)
            page_fallbacks.append(0)
            page_lang_switches.append(0)
            page_stats_raw.append(_page_stats(""))
            page_skipped.append(True)
            page_rescues.append(False)
            page_rescue_reasons.append("no_image")
            page_toc_removed.append(0)
            if on_page:
                on_page(page_num, total)
            continue

        img_bgr, gray, bw, angle = preprocess_image(images[0])

        if use_layout:
            blocks = detect_layout_blocks(img_bgr)
        else:
            h, w = gray.shape[:2]
            blocks = [Block(kind="text", bbox=(0, 0, w, h))]
        if not any(b.kind in ("text", "table") for b in blocks):
            # Fallback to full-page OCR if layout returns only figures.
            h, w = gray.shape[:2]
            blocks = [Block(kind="text", bbox=(0, 0, w, h))]

        block_texts = []
        block_confs = []
        block_lows = []
        block_retries = []
        fallback_count = 0
        lang_switch_count = 0
        table_block_count = 0
        table_index = 0
        figure_index = 0
        text_block_records = []
        pending_media = []

        for block in blocks:
            if block.kind == "table":
                table_block_count += 1

            crop_gray = crop_image(gray, block.bbox)
            crop_bw = crop_image(bw, block.bbox)
            crop_bgr = crop_image(img_bgr, block.bbox)

            primary_img, primary_mode = choose_preprocessed(
                crop_gray, crop_bw, preprocess_mode, ink_ratio_threshold
            )
            alt_img = crop_gray if primary_mode == "bw" else crop_bw

            lang_used = lang
            used_fallback = False
            is_table = block.kind == "table"
            is_figure = block.kind == "figure"

            media_path = ""
            if capture_media and media_dir and (is_table or is_figure):
                os.makedirs(media_dir, exist_ok=True)
                if is_table:
                    table_index += 1
                    filename = f"page{page_num:03d}_table_{table_index:03d}.png"
                else:
                    figure_index += 1
                    filename = f"page{page_num:03d}_figure_{figure_index:03d}.png"
                media_path = os.path.join(media_dir, filename)
                cv2.imwrite(media_path, crop_bgr)

            if is_figure:
                fig_ocr_text = ""
                if summarize_figures:
                    fig_ocr_text, _, _, _ = ocr_tesseract(
                        primary_img,
                        lang_used,
                        oem,
                        psm_text,
                        conf_threshold,
                        retry_psm=retry_psm,
                    )
                if capture_media and (is_table or is_figure):
                    pending_media.append(
                        {
                            "page": page_num,
                            "pdf_page": page_num,
                            "page_index": page_num - 1,
                            "type": "figure",
                            "bbox": list(block.bbox),
                            "image_path": media_path,
                            "ocr_text": fig_ocr_text.strip(),
                            "summary": "",
                            "title": "",
                            "context": "",
                        }
                    )
                continue

            if block.kind == "table" and use_table_ocr:
                text, avg_conf = ocr_table_paddle(crop_bgr)
                low = 0
                retried = False
                if not text or avg_conf < conf_threshold:
                    text, avg_conf, low, retried = ocr_tesseract(
                        primary_img,
                        lang_used,
                        oem,
                        psm_table,
                        conf_threshold,
                        retry_psm=retry_psm,
                    )
                    used_fallback = True

                if auto_lang and avg_conf < conf_threshold:
                    alt_lang = _choose_lang(text, hangul_ratio_threshold)
                    if alt_lang and alt_lang != lang_used:
                        alt_text, alt_conf, alt_low, alt_retried = ocr_tesseract(
                            primary_img,
                            alt_lang,
                            oem,
                            psm_table,
                            conf_threshold,
                            retry_psm=retry_psm,
                        )
                        if alt_conf >= avg_conf + lang_switch_margin:
                            text, avg_conf, low, retried = alt_text, alt_conf, alt_low, alt_retried
                            lang_used = alt_lang
                            lang_switch_count += 1

                if use_fallback and alt_img is not None and avg_conf < conf_threshold:
                    alt_text, alt_conf, alt_low, alt_retried = ocr_tesseract(
                        alt_img,
                        lang_used,
                        oem,
                        psm_table,
                        conf_threshold,
                        retry_psm=retry_psm,
                    )
                    if alt_conf > avg_conf:
                        text, avg_conf, low, retried = alt_text, alt_conf, alt_low, alt_retried
                        used_fallback = True
            else:
                text, avg_conf, low, retried = ocr_tesseract(
                    primary_img,
                    lang_used,
                    oem,
                    psm_text,
                    conf_threshold,
                    retry_psm=retry_psm,
                )

                if auto_lang and avg_conf < conf_threshold:
                    alt_lang = _choose_lang(text, hangul_ratio_threshold)
                    if alt_lang and alt_lang != lang_used:
                        alt_text, alt_conf, alt_low, alt_retried = ocr_tesseract(
                            primary_img,
                            alt_lang,
                            oem,
                            psm_text,
                            conf_threshold,
                            retry_psm=retry_psm,
                        )
                        if alt_conf >= avg_conf + lang_switch_margin:
                            text, avg_conf, low, retried = alt_text, alt_conf, alt_low, alt_retried
                            lang_used = alt_lang
                            lang_switch_count += 1

                used_fallback = False
                if use_fallback and alt_img is not None and avg_conf < conf_threshold:
                    alt_text, alt_conf, alt_low, alt_retried = ocr_tesseract(
                        alt_img,
                        lang_used,
                        oem,
                        psm_text,
                        conf_threshold,
                        retry_psm=retry_psm,
                    )
                    if alt_conf > avg_conf:
                        text, avg_conf, low, retried = alt_text, alt_conf, alt_low, alt_retried
                        used_fallback = True

            if is_table and capture_media:
                table_ocr_text = text.strip()
                pending_media.append(
                    {
                        "page": page_num,
                        "pdf_page": page_num,
                        "page_index": page_num - 1,
                        "type": "table",
                        "bbox": list(block.bbox),
                        "image_path": media_path,
                        "ocr_text": table_ocr_text,
                        "summary": "",
                        "title": "",
                        "context": "",
                    }
                )

            if used_fallback:
                fallback_count += 1

            if use_vocab:
                text = correct_text(text, VOCAB)

            block_texts.append(text.strip())
            block_confs.append(avg_conf)
            block_lows.append(low)
            block_retries.append(retried)
            if block.kind == "text":
                text_block_records.append((list(block.bbox), text.strip()))

        page_text = "\n\n".join([t for t in block_texts if t])
        if remove_toc:
            page_text, toc_active, toc_non_toc_run, toc_removed = _strip_toc_page(
                page_text,
                toc_active,
                toc_non_toc_run,
                toc_min_keep,
                toc_max_pages,
                page_num - 1,
            )
        else:
            toc_removed = 0
        stats_raw = _page_stats(page_text)
        short_ratio = _short_line_ratio(page_text)

        rescue_used = False
        rescue_reason = ""
        rescue_trigger = short_ratio >= rescue_short_ratio
        low_text_trigger = skip_low_text and (stats_raw["total"] < min_chars or stats_raw["score"] < min_score)
        if rescue_trigger or low_text_trigger:
            if rescue_trigger and low_text_trigger:
                rescue_reason = "short_lines+low_text"
            elif rescue_trigger:
                rescue_reason = "short_lines"
            else:
                rescue_reason = "low_text"

            rescue_psm = rescue_psm_table if table_block_count > 0 else rescue_psm_text
            rescue_text, rescue_conf, rescue_low, rescue_retry = ocr_tesseract(
                gray,
                DEFAULT_LANG,
                oem,
                rescue_psm,
                conf_threshold,
                retry_psm=None,
            )
            rescue_text = rescue_text.strip()
            rescue_stats = _page_stats(rescue_text)
            if rescue_stats["total"] >= min_chars and rescue_stats["score"] >= min_score:
                page_text = rescue_text
                stats_raw = rescue_stats
                rescue_used = True
                short_ratio = _short_line_ratio(page_text)
                # override block-level metrics with rescue pass metrics
                block_confs = [rescue_conf]
                block_lows = [rescue_low]
                block_retries = [rescue_retry]
            else:
                rescue_used = False

        skipped = False
        if skip_low_text and not rescue_used and (stats_raw["total"] < min_chars or stats_raw["score"] < min_score):
            page_text = ""
            skipped = True
            rescue_reason = rescue_reason or "low_text"

        pages.append(page_text)
        page_stats_raw.append(stats_raw)
        page_skipped.append(skipped)
        page_rescues.append(rescue_used)
        page_rescue_reasons.append(rescue_reason)
        page_toc_removed.append(toc_removed)

        if block_confs:
            avg_page_conf = int(round(sum(block_confs) / len(block_confs)))
            total_low = int(sum(block_lows))
            any_retry = any(block_retries)
        else:
            avg_page_conf = 0
            total_low = 0
            any_retry = False

        page_confs.append(avg_page_conf)
        page_low_conf.append(total_low)
        page_retry.append(any_retry)
        page_fallbacks.append(fallback_count)
        page_lang_switches.append(lang_switch_count)

        # Build context + summaries for tables/figures using nearby text blocks.
        if pending_media:
            for item in pending_media:
                context = _collect_context(text_block_records, item["bbox"])
                item["context"] = context
                item["title"] = _extract_caption_title(context, item.get("ocr_text", ""))
                if item["type"] == "table" and summarize_tables:
                    prompt = _build_table_prompt(item["ocr_text"], context_text=context)
                    if prompt:
                        summary, err = deepseek_summarize(
                            prompt, deepseek_key, deepseek_base, deepseek_model
                        )
                        if err:
                            deepseek_errors += 1
                        else:
                            deepseek_calls += 1
                        item["summary"] = summary
                elif item["type"] == "figure" and summarize_figures:
                    prompt = _build_figure_prompt(item["ocr_text"], context_text=context)
                    if prompt:
                        summary, err = deepseek_summarize(
                            prompt, deepseek_key, deepseek_base, deepseek_model
                        )
                        if err:
                            deepseek_errors += 1
                        else:
                            deepseek_calls += 1
                        item["summary"] = summary
            media_items.extend(pending_media)

        if on_page:
            on_page(page_num, total)

        # best-effort memory cleanup for large page images
        if mem_cleanup:
            try:
                images[0].close()
            except Exception:
                pass
            try:
                del img_bgr, gray, bw, blocks, block_texts, block_confs, block_lows, block_retries
            except Exception:
                pass
            try:
                del images
            except Exception:
                pass
            gc.collect()

    return (
        pages,
        page_confs,
        page_low_conf,
        page_retry,
        page_fallbacks,
        page_lang_switches,
        page_stats_raw,
        page_skipped,
        page_rescues,
        page_rescue_reasons,
        page_toc_removed,
        media_items,
        deepseek_calls,
        deepseek_errors,
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF OCR to TXT (Korean)")
        self.geometry("1020x820")
        self.minsize(900, 680)

        self.input_mode = tk.StringVar(value="single")
        self.ocr_mode = tk.StringVar(value="auto")
        self.lang_var = tk.StringVar(value=DEFAULT_LANG)
        self.dpi_var = tk.StringVar(value=str(DEFAULT_DPI))
        self.tesseract_path_var = tk.StringVar(value="")
        self.input_path_var = tk.StringVar(value="")
        self.output_path_var = tk.StringVar(value="")

        self.tess_oem_var = tk.StringVar(value=DEFAULT_TESS_OEM)
        self.tess_psm_text_var = tk.StringVar(value=DEFAULT_TESS_PSM_TEXT)
        self.tess_psm_table_var = tk.StringVar(value=DEFAULT_TESS_PSM_TABLE)
        self.retry_psm_var = tk.StringVar(value=DEFAULT_RETRY_PSM)
        self.conf_threshold_var = tk.StringVar(value=DEFAULT_CONF_THRESHOLD)

        self.preprocess_mode_var = tk.StringVar(value=DEFAULT_PREPROCESS_MODE)
        self.ink_ratio_var = tk.StringVar(value=DEFAULT_INK_RATIO_THRESHOLD)

        self.use_layout_var = tk.BooleanVar(value=True)
        self.use_table_var = tk.BooleanVar(value=True)
        self.use_vocab_var = tk.BooleanVar(value=True)
        self.retry_low_conf_var = tk.BooleanVar(value=True)
        self.fallback_var = tk.BooleanVar(value=True)
        self.fast_mode_var = tk.BooleanVar(value=False)

        self.auto_lang_var = tk.BooleanVar(value=True)
        self.hangul_ratio_var = tk.StringVar(value=DEFAULT_HANGUL_RATIO)
        self.auto_media_scan_var = tk.BooleanVar(value=DEFAULT_AUTO_MEDIA_SCAN)
        self.media_area_ratio_var = tk.StringVar(value=DEFAULT_MEDIA_AREA_RATIO)
        self.media_scan_dpi_var = tk.StringVar(value=str(DEFAULT_MEDIA_SCAN_DPI))
        self.skip_low_text_var = tk.BooleanVar(value=True)
        self.min_chars_var = tk.StringVar(value=DEFAULT_MIN_CHARS)
        self.min_score_var = tk.StringVar(value=DEFAULT_MIN_SCORE)
        self.remove_toc_var = tk.BooleanVar(value=DEFAULT_REMOVE_TOC)
        self.use_deepseek_var = tk.BooleanVar(value=DEFAULT_USE_DEEPSEEK)
        self.summarize_tables_var = tk.BooleanVar(value=DEFAULT_SUMMARIZE_TABLES)
        self.summarize_figures_var = tk.BooleanVar(value=DEFAULT_SUMMARIZE_FIGURES)
        self.deepseek_key_var = tk.StringVar(value="")
        self.deepseek_base_var = tk.StringVar(value=DEFAULT_DEEPSEEK_BASE)
        self.deepseek_model_var = tk.StringVar(value=DEFAULT_DEEPSEEK_MODEL)
        self.mem_cleanup_var = tk.BooleanVar(value=DEFAULT_MEM_CLEANUP)
        self.clean_output_var = tk.BooleanVar(value=DEFAULT_CLEAN_OUTPUT)
        self.llm_clean_var = tk.BooleanVar(value=DEFAULT_LLM_CLEAN)
        self.llm_skip_tables_var = tk.BooleanVar(value=DEFAULT_LLM_SKIP_TABLES)
        self.preview_media_var = tk.BooleanVar(value=DEFAULT_PREVIEW_MEDIA)

        self._queue = queue.Queue()
        self._running = False
        self._cancel = False

        self._build_ui()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        main = ttk.Frame(self, padding=12)
        main.pack(fill=tk.BOTH, expand=True)

        mode_frame = ttk.LabelFrame(main, text="Input")
        mode_frame.pack(fill=tk.X)
        ttk.Radiobutton(mode_frame, text="Single PDF", variable=self.input_mode, value="single").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(mode_frame, text="Folder of PDFs", variable=self.input_mode, value="folder").grid(row=0, column=1, sticky="w", padx=8, pady=6)

        path_frame = ttk.Frame(main)
        path_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(path_frame, text="Input path:").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.input_path_var).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(path_frame, text="Browse", command=self.browse_input).grid(row=0, column=2)
        path_frame.columnconfigure(1, weight=1)

        out_frame = ttk.Frame(main)
        out_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(out_frame, text="Output folder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(out_frame, textvariable=self.output_path_var).grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(out_frame, text="Browse", command=self.browse_output).grid(row=0, column=2)
        out_frame.columnconfigure(1, weight=1)

        opts = ttk.LabelFrame(main, text="Options")
        opts.pack(fill=tk.X, pady=(10, 0))
        ttk.Label(opts, text="OCR mode:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Radiobutton(opts, text="Auto (text then OCR)", variable=self.ocr_mode, value="auto").grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(opts, text="OCR only", variable=self.ocr_mode, value="ocr").grid(row=0, column=2, sticky="w")
        ttk.Radiobutton(opts, text="Text only", variable=self.ocr_mode, value="text").grid(row=0, column=3, sticky="w")
        ttk.Checkbutton(opts, text="Fast mode (speed)", variable=self.fast_mode_var).grid(row=0, column=4, sticky="w")

        ttk.Label(opts, text="Language (Tesseract):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.lang_var, width=20).grid(row=1, column=1, sticky="w")
        ttk.Label(opts, text="DPI:").grid(row=1, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.dpi_var, width=8).grid(row=1, column=3, sticky="w")

        ttk.Label(opts, text="Tesseract OEM:").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.tess_oem_var, width=8).grid(row=2, column=1, sticky="w")
        ttk.Label(opts, text="PSM (text):").grid(row=2, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.tess_psm_text_var, width=8).grid(row=2, column=3, sticky="w")

        ttk.Label(opts, text="PSM (table):").grid(row=3, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.tess_psm_table_var, width=8).grid(row=3, column=1, sticky="w")
        ttk.Label(opts, text="Retry PSM:").grid(row=3, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.retry_psm_var, width=8).grid(row=3, column=3, sticky="w")

        ttk.Label(opts, text="Conf threshold:").grid(row=4, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.conf_threshold_var, width=8).grid(row=4, column=1, sticky="w")
        ttk.Checkbutton(opts, text="Layout detection", variable=self.use_layout_var).grid(row=4, column=2, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="Table OCR", variable=self.use_table_var).grid(row=4, column=3, sticky="w")

        ttk.Label(opts, text="Preprocess:").grid(row=5, column=0, sticky="w", padx=8, pady=6)
        ttk.Combobox(
            opts,
            textvariable=self.preprocess_mode_var,
            values=["auto", "gray", "bw"],
            width=8,
            state="readonly",
        ).grid(row=5, column=1, sticky="w")
        ttk.Label(opts, text="Ink ratio max:").grid(row=5, column=2, sticky="e")
        ttk.Entry(opts, textvariable=self.ink_ratio_var, width=8).grid(row=5, column=3, sticky="w")

        ttk.Checkbutton(opts, text="Vocab correction", variable=self.use_vocab_var).grid(row=6, column=0, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="Retry low-conf", variable=self.retry_low_conf_var).grid(row=6, column=1, sticky="w")
        ttk.Checkbutton(opts, text="Fallback gray/bw", variable=self.fallback_var).grid(row=6, column=2, sticky="w", padx=8)

        ttk.Checkbutton(opts, text="Auto language", variable=self.auto_lang_var).grid(row=7, column=0, sticky="w", padx=8)
        ttk.Label(opts, text="Hangul ratio:").grid(row=7, column=1, sticky="e")
        ttk.Entry(opts, textvariable=self.hangul_ratio_var, width=8).grid(row=7, column=2, sticky="w")

        ttk.Checkbutton(opts, text="Skip low-text", variable=self.skip_low_text_var).grid(row=8, column=0, sticky="w", padx=8)
        ttk.Label(opts, text="Min chars:").grid(row=8, column=1, sticky="e")
        ttk.Entry(opts, textvariable=self.min_chars_var, width=8).grid(row=8, column=2, sticky="w")

        ttk.Label(opts, text="Min score:").grid(row=9, column=1, sticky="e")
        ttk.Entry(opts, textvariable=self.min_score_var, width=8).grid(row=9, column=2, sticky="w")
        ttk.Checkbutton(opts, text="Remove TOC", variable=self.remove_toc_var).grid(row=9, column=3, sticky="w", padx=8)

        ttk.Checkbutton(opts, text="DeepSeek summaries", variable=self.use_deepseek_var).grid(row=10, column=0, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="Tables", variable=self.summarize_tables_var).grid(row=10, column=1, sticky="w")
        ttk.Checkbutton(opts, text="Figures", variable=self.summarize_figures_var).grid(row=10, column=2, sticky="w")
        ttk.Checkbutton(opts, text="Memory cleanup", variable=self.mem_cleanup_var).grid(row=10, column=3, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="Clean output (extra file)", variable=self.clean_output_var).grid(row=10, column=4, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="LLM clean (extra file)", variable=self.llm_clean_var).grid(row=10, column=5, sticky="w", padx=8)
        ttk.Checkbutton(opts, text="LLM skip tables/figures", variable=self.llm_skip_tables_var).grid(row=10, column=6, sticky="w", padx=8)

        ttk.Label(opts, text="DeepSeek base URL:").grid(row=11, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.deepseek_base_var).grid(row=11, column=1, columnspan=3, sticky="we", padx=(0, 8))

        ttk.Label(opts, text="DeepSeek model:").grid(row=12, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.deepseek_model_var, width=20).grid(row=12, column=1, sticky="w")
        ttk.Label(opts, text="API key: from .env (DEEPSEEK_API_KEY)").grid(row=12, column=2, columnspan=2, sticky="w", padx=(0, 8))

        ttk.Label(opts, text="Tesseract path (optional):").grid(row=13, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(opts, textvariable=self.tesseract_path_var).grid(row=13, column=1, columnspan=3, sticky="we", padx=(0, 8))
        ttk.Checkbutton(
            opts,
            text="Auto scan large table/figure (auto mode)",
            variable=self.auto_media_scan_var,
        ).grid(row=14, column=0, sticky="w", padx=8, pady=(0, 6))
        ttk.Label(opts, text="Min area ratio:").grid(row=14, column=1, sticky="e", padx=4)
        ttk.Entry(opts, textvariable=self.media_area_ratio_var, width=8).grid(row=14, column=2, sticky="w")
        ttk.Label(opts, text="Scan DPI (>=80):").grid(row=14, column=3, sticky="e", padx=4)
        ttk.Entry(opts, textvariable=self.media_scan_dpi_var, width=6).grid(row=14, column=4, sticky="w")
        ttk.Checkbutton(opts, text="Preview media after run", variable=self.preview_media_var).grid(row=14, column=5, sticky="w", padx=8)
        opts.columnconfigure(1, weight=1)

        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, pady=(10, 0))
        self.run_btn = ttk.Button(action_frame, text="Run", command=self.start)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(action_frame, text="Stop", command=self.stop)
        self.stop_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(action_frame, text="Preview JSON", command=self.open_media_preview).pack(side=tk.LEFT, padx=8)
        ttk.Button(action_frame, text="Clear Log", command=self.clear_log).pack(side=tk.LEFT, padx=8)

        progress_frame = ttk.LabelFrame(main, text="Progress")
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(progress_frame, textvariable=self.status_var).pack(anchor="w", padx=8, pady=(6, 0))

        self.file_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.file_progress.pack(fill=tk.X, padx=8, pady=6)
        self.page_progress = ttk.Progressbar(progress_frame, mode="determinate")
        self.page_progress.pack(fill=tk.X, padx=8, pady=(0, 8))

        log_frame = ttk.LabelFrame(main, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def browse_input(self):
        mode = self.input_mode.get()
        if mode == "single":
            path = filedialog.askopenfilename(title="Select PDF", filetypes=[("PDF", "*.pdf")])
        else:
            path = filedialog.askdirectory(title="Select Folder")
        if path:
            self.input_path_var.set(path)
            if not self.output_path_var.get().strip():
                self.output_path_var.set(_ensure_output_dir(path, ""))

    def browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_path_var.set(path)

    def clear_log(self):
        self.log_text.delete("1.0", tk.END)

    def open_media_preview(self):
        path = filedialog.askopenfilename(
            title="Select figure/table JSON",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            messagebox.showerror("Preview error", f"Failed to read JSON: {exc}")
            return
        items = []
        if isinstance(data, dict):
            items = data.get("items", [])
        elif isinstance(data, list):
            items = data
        if not items:
            messagebox.showinfo("Preview", "No media items found in JSON.")
            return
        base_dir = os.path.dirname(path)
        source_dir = ""
        if isinstance(data, dict):
            src = data.get("source_path") or ""
            if src:
                source_dir = os.path.dirname(src)

        for item in items:
            img_path = (item.get("image_path") or "").strip()
            if img_path and not os.path.isabs(img_path):
                candidates = [os.path.join(base_dir, img_path)]
                if source_dir:
                    candidates.append(os.path.join(source_dir, img_path))
                for cand in candidates:
                    if os.path.isfile(cand):
                        item["image_path"] = cand
                        break
        self._queue.put(("media_preview", items))

    def _ui_log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def _set_status(self, msg):
        self.status_var.set(msg)

    def _set_progress(self, bar, current, total):
        bar["maximum"] = max(total, 1)
        bar["value"] = min(current, total)

    def _poll_queue(self):
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            kind = item[0]
            if kind == "log":
                self._ui_log(item[1])
            elif kind == "status":
                self._set_status(item[1])
            elif kind == "progress_files":
                self._set_progress(self.file_progress, item[1], item[2])
            elif kind == "progress_pages":
                self._set_progress(self.page_progress, item[1], item[2])
            elif kind == "done":
                self._running = False
                self.run_btn["state"] = tk.NORMAL
                self._set_status("Done")
            elif kind == "media_preview":
                self._show_media_preview(item[1])
        self.after(100, self._poll_queue)

    def _qlog(self, msg):
        self._queue.put(("log", msg))

    def _qstatus(self, msg):
        self._queue.put(("status", msg))

    def _qprogress_files(self, current, total):
        self._queue.put(("progress_files", current, total))

    def _qprogress_pages(self, current, total):
        self._queue.put(("progress_pages", current, total))

    def _show_media_preview(self, media_items):
        if not media_items:
            return
        if getattr(self, "_preview_window", None) is None or not self._preview_window.winfo_exists():
            self._preview_window = tk.Toplevel(self)
            self._preview_window.title("Figure/Table Preview")
            self._preview_window.geometry("900x700")
            self._preview_items = media_items
            self._preview_index = 0

            top = ttk.Frame(self._preview_window)
            top.pack(fill=tk.X, padx=8, pady=6)
            self._preview_title_var = tk.StringVar(value="")
            ttk.Label(top, textvariable=self._preview_title_var).pack(side=tk.LEFT)
            nav = ttk.Frame(self._preview_window)
            nav.pack(fill=tk.X, padx=8)
            ttk.Button(nav, text="Prev", command=self._preview_prev).pack(side=tk.LEFT)
            ttk.Button(nav, text="Next", command=self._preview_next).pack(side=tk.LEFT, padx=6)

            self._preview_image_label = ttk.Label(self._preview_window)
            self._preview_image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

            self._preview_text = tk.Text(self._preview_window, height=10, wrap="word")
            self._preview_text.pack(fill=tk.BOTH, expand=False, padx=8, pady=6)
        else:
            self._preview_items = media_items
            self._preview_index = 0
        self._preview_render()

    def _preview_prev(self):
        if not hasattr(self, "_preview_items") or not self._preview_items:
            return
        self._preview_index = max(0, self._preview_index - 1)
        self._preview_render()

    def _preview_next(self):
        if not hasattr(self, "_preview_items") or not self._preview_items:
            return
        self._preview_index = min(len(self._preview_items) - 1, self._preview_index + 1)
        self._preview_render()

    def _preview_render(self):
        if not hasattr(self, "_preview_items") or not self._preview_items:
            return
        item = self._preview_items[self._preview_index]
        title = (item.get("title") or "").strip()
        page = item.get("pdf_page", item.get("page", 0))
        kind = item.get("type", "")
        header = f"[{self._preview_index+1}/{len(self._preview_items)}] Page {page} - {kind}"
        if title:
            header += f" - {title}"
        self._preview_title_var.set(header)

        img_path = item.get("image_path", "")
        preview_img = None
        if img_path and os.path.isfile(img_path):
            try:
                if Image is not None and ImageTk is not None:
                    img = Image.open(img_path)
                    img.thumbnail((860, 420))
                    preview_img = ImageTk.PhotoImage(img)
                else:
                    preview_img = tk.PhotoImage(file=img_path)
            except Exception:
                preview_img = None
        if preview_img is not None:
            self._preview_image = preview_img
            self._preview_image_label.configure(image=preview_img, text="")
        else:
            self._preview_image_label.configure(image="", text="(No preview available)")

        summary = (item.get("summary") or "").strip()
        context = (item.get("context") or "").strip()
        ocr_text = (item.get("ocr_text") or "").strip()
        body = []
        if summary:
            body.append("Summary:\n" + summary)
        if context:
            body.append("Context:\n" + context)
        if ocr_text:
            body.append("OCR:\n" + ocr_text)
        self._preview_text.delete("1.0", tk.END)
        self._preview_text.insert(tk.END, "\n\n".join(body) if body else "(No text)")

    def start(self):
        if self._running:
            return
        try:
            _ensure_deps()
        except Exception as exc:
            messagebox.showerror("Missing dependencies", str(exc))
            return

        in_path = self.input_path_var.get().strip()
        out_dir = self.output_path_var.get().strip()
        out_dir = _ensure_output_dir(in_path, out_dir)
        if out_dir:
            self.output_path_var.set(out_dir)

        if not in_path:
            messagebox.showwarning("Missing input", "Choose a PDF or a folder.")
            return
        if not out_dir:
            messagebox.showwarning("Missing output", "Choose an output folder.")
            return
        if not os.path.isdir(out_dir):
            messagebox.showwarning("Invalid output", "Output folder does not exist.")
            return

        tesseract_path = self.tesseract_path_var.get().strip()
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self._running = True
        self._cancel = False
        self.run_btn["state"] = tk.DISABLED
        self.stop_btn["state"] = tk.NORMAL
        self._set_status("Starting...")
        self._set_progress(self.file_progress, 0, 1)
        self._set_progress(self.page_progress, 0, 1)

        _load_dotenv()

        thread = threading.Thread(target=self.run_job, args=(in_path, out_dir), daemon=True)
        thread.start()

    def stop(self):
        if not self._running:
            return
        self._cancel = True
        self._qlog("Stopping after current page...")

    def run_job(self, in_path, out_dir):
        try:
            self._qlog("Starting...")
            mode = self.input_mode.get()
            ocr_mode = self.ocr_mode.get()
            lang = self.lang_var.get().strip() or DEFAULT_LANG
            dpi = _safe_int(self.dpi_var.get(), DEFAULT_DPI)
            poppler_path = _detect_poppler_path()
            if not poppler_path:
                self._qlog("Poppler not found; PDF rendering may fail.")

            oem = self.tess_oem_var.get().strip() or DEFAULT_TESS_OEM
            psm_text = self.tess_psm_text_var.get().strip() or DEFAULT_TESS_PSM_TEXT
            psm_table = self.tess_psm_table_var.get().strip() or DEFAULT_TESS_PSM_TABLE
            conf_threshold = _safe_int(self.conf_threshold_var.get(), int(DEFAULT_CONF_THRESHOLD))
            retry_psm = self.retry_psm_var.get().strip() or DEFAULT_RETRY_PSM
            if not self.retry_low_conf_var.get():
                retry_psm = None

            preprocess_mode = (self.preprocess_mode_var.get().strip().lower() or DEFAULT_PREPROCESS_MODE)
            if preprocess_mode not in ("auto", "gray", "bw"):
                preprocess_mode = DEFAULT_PREPROCESS_MODE

            ink_ratio_threshold = _safe_float(self.ink_ratio_var.get(), float(DEFAULT_INK_RATIO_THRESHOLD))
            use_fallback = self.fallback_var.get()

            auto_lang = self.auto_lang_var.get()
            hangul_ratio_threshold = _safe_float(self.hangul_ratio_var.get(), float(DEFAULT_HANGUL_RATIO))
            lang_switch_margin = _safe_int(DEFAULT_LANG_SWITCH_MARGIN, 5)
            auto_media_scan = self.auto_media_scan_var.get()
            media_area_ratio = _safe_float(self.media_area_ratio_var.get(), float(DEFAULT_MEDIA_AREA_RATIO))
            media_scan_dpi = _safe_int(self.media_scan_dpi_var.get(), int(DEFAULT_MEDIA_SCAN_DPI))
            if media_scan_dpi < 80:
                media_scan_dpi = 80

            skip_low_text = self.skip_low_text_var.get()
            min_chars = _safe_int(self.min_chars_var.get(), int(DEFAULT_MIN_CHARS))
            min_score = _safe_int(self.min_score_var.get(), int(DEFAULT_MIN_SCORE))
            rescue_short_ratio = _safe_float(DEFAULT_RESCUE_SHORT_RATIO, 0.60)
            rescue_psm_text = DEFAULT_RESCUE_PSM_TEXT
            rescue_psm_table = DEFAULT_RESCUE_PSM_TABLE
            remove_toc = self.remove_toc_var.get()
            toc_max_pages = _safe_int(DEFAULT_TOC_MAX_PAGES, 5)
            toc_min_keep = _safe_int(DEFAULT_TOC_MIN_KEEP, 8)

            fast_mode = self.fast_mode_var.get()
            clean_output = self.clean_output_var.get()
            llm_clean = self.llm_clean_var.get()
            llm_skip_tables = self.llm_skip_tables_var.get()

            use_deepseek = self.use_deepseek_var.get()
            summarize_tables = self.summarize_tables_var.get()
            summarize_figures = self.summarize_figures_var.get()
            deepseek_key = self.deepseek_key_var.get().strip()
            deepseek_base = (self.deepseek_base_var.get().strip() or DEFAULT_DEEPSEEK_BASE)
            deepseek_model = (self.deepseek_model_var.get().strip() or DEFAULT_DEEPSEEK_MODEL)
            if not deepseek_key:
                deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip() or os.environ.get("DEEPSEEK_KEY", "").strip()
            if deepseek_base == DEFAULT_DEEPSEEK_BASE:
                env_base = os.environ.get("DEEPSEEK_BASE_URL", "").strip()
                if env_base:
                    deepseek_base = env_base
            if deepseek_model == DEFAULT_DEEPSEEK_MODEL:
                env_model = os.environ.get("DEEPSEEK_MODEL", "").strip()
                if env_model:
                    deepseek_model = env_model
            if use_deepseek:
                if not deepseek_key:
                    self._qlog("DeepSeek enabled but API key missing; summaries disabled.")
                    use_deepseek = False
                elif requests is None:
                    self._qlog("DeepSeek enabled but requests is not installed; summaries disabled.")
                    use_deepseek = False
            if not use_deepseek:
                summarize_tables = False
                summarize_figures = False
                if llm_clean:
                    self._qlog("LLM clean enabled but DeepSeek unavailable; LLM cleaning disabled.")
                    llm_clean = False

            mem_cleanup = self.mem_cleanup_var.get()

            layout_requested = self.use_layout_var.get()
            use_layout = layout_requested
            use_table = self.use_table_var.get()
            use_vocab = self.use_vocab_var.get()

            if fast_mode:
                self._qlog("Fast mode enabled: dpi=200, layout/table/deepseek off, retry/fallback/auto_lang off, LLM clean on.")
                dpi = min(dpi, 200)
                use_layout = False
                use_table = False
                use_deepseek = False
                summarize_tables = False
                summarize_figures = False
                auto_lang = False
                use_fallback = False
                retry_psm = None
                clean_output = True
                llm_clean = True
                llm_skip_tables = True
                use_deepseek = True

            layout_engine, layout_kind = get_layout_engine()
            if use_layout and layout_engine is None:
                self._qlog("Layout detection not available; install paddlepaddle or detectron2.")
                use_layout = False
            if use_table and get_table_ocr_engine() is None:
                self._qlog("Table OCR (Paddle) not available; using Tesseract for tables.")
            if use_table and not use_layout:
                self._qlog("Table OCR enabled but layout detection is off; tables will be treated as text.")
            if use_deepseek and not use_layout:
                self._qlog("DeepSeek summaries enabled but layout detection is off; no tables/figures will be detected.")

            if mode == "single":
                pdfs = [in_path] if is_pdf(in_path) else []
            else:
                pdfs = [
                    os.path.join(in_path, f)
                    for f in os.listdir(in_path)
                    if f.lower().endswith(".pdf")
                ]

            if not pdfs:
                self._qlog("No PDF files found.")
                self._queue.put(("done",))
                return

            report_path = os.path.join(out_dir, QUALITY_REPORT_NAME)
            report_lines = []

            total_files = len(pdfs)
            for idx, pdf_path in enumerate(pdfs, start=1):
                if self._cancel:
                    self._qlog("Stopped by user.")
                    break
                self._qprogress_files(idx - 1, total_files)
                base = os.path.splitext(os.path.basename(pdf_path))[0]
                out_path = os.path.join(out_dir, f"{base}.txt")

                self._qlog(f"Processing: {pdf_path}")
                self._qstatus(f"File {idx}/{total_files}")

                pages = []
                page_confs = []
                page_lows = []
                page_retries = []
                page_fallbacks = []
                page_lang_switches = []
                page_stats_raw = []
                page_skipped = []
                page_rescues = []
                page_rescue_reasons = []
                page_toc_removed = []
                media_items = []
                deepseek_calls = 0
                deepseek_errors = 0

                if ocr_mode in ("auto", "text"):
                    def on_page_text(cur, total):
                        self._qprogress_pages(cur, total)
                    pages = extract_text_pdf_pages(pdf_path, on_page=on_page_text)

                text = "\n\n".join(pages).strip()

                if ocr_mode == "ocr" or (ocr_mode == "auto" and len(text) < TEXT_MIN_CHARS):
                    if ocr_mode == "auto":
                        self._qlog("Text extraction weak; running OCR...")
                    def on_page_ocr(cur, total):
                        self._qprogress_pages(cur, total)
                        if self._cancel:
                            raise RuntimeError("Cancelled by user")
                    (
                        pages,
                        page_confs,
                        page_lows,
                        page_retries,
                        page_fallbacks,
                        page_lang_switches,
                        page_stats_raw,
                        page_skipped,
                        page_rescues,
                        page_rescue_reasons,
                        page_toc_removed,
                        media_items,
                        deepseek_calls,
                        deepseek_errors,
                    ) = ocr_pdf_pages(
                        pdf_path,
                        lang=lang,
                        dpi=dpi,
                        oem=oem,
                        psm_text=psm_text,
                        psm_table=psm_table,
                        conf_threshold=conf_threshold,
                        retry_psm=retry_psm,
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
                        capture_media=use_deepseek,
                        media_dir=os.path.join(out_dir, f"{base}_media"),
                        deepseek_key=deepseek_key,
                        deepseek_base=deepseek_base,
                        deepseek_model=deepseek_model,
                        summarize_tables=summarize_tables,
                        summarize_figures=summarize_figures,
                        mem_cleanup=mem_cleanup,
                        poppler_path=poppler_path,
                        on_page=on_page_ocr,
                    )
                    text = "\n\n".join(pages).strip()
                elif ocr_mode == "auto" and len(text) >= TEXT_MIN_CHARS and auto_media_scan:
                    if not use_layout:
                        self._qlog("Auto media scan skipped: layout detection is off.")
                    else:
                        self._qlog("Auto media scan: checking for large tables/figures...")
                        has_large_media = scan_large_media(
                            pdf_path,
                            poppler_path=poppler_path,
                            dpi=media_scan_dpi,
                            min_area_ratio=media_area_ratio,
                        )
                        if has_large_media:
                            self._qlog("Large table/figure detected; running OCR for media...")
                            def on_page_ocr(cur, total):
                                self._qprogress_pages(cur, total)
                                if self._cancel:
                                    raise RuntimeError("Cancelled by user")
                            (
                                _pages_ocr,
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
                                oem=oem,
                                psm_text=psm_text,
                                psm_table=psm_table,
                                conf_threshold=conf_threshold,
                                retry_psm=retry_psm,
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
                                capture_media=use_deepseek,
                                media_dir=os.path.join(out_dir, f"{base}_media"),
                                deepseek_key=deepseek_key,
                                deepseek_base=deepseek_base,
                                deepseek_model=deepseek_model,
                                summarize_tables=summarize_tables,
                                summarize_figures=summarize_figures,
                                mem_cleanup=mem_cleanup,
                                poppler_path=poppler_path,
                                on_page=on_page_ocr,
                            )
                        else:
                            self._qlog("Auto media scan: no large tables/figures found; skipping OCR.")

                if not text:
                    self._qlog("Warning: no text extracted.")

                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                self._qlog(f"Saved: {out_path}")
                clean_path = ""
                if clean_output and text:
                    clean_text = clean_output_text(text)
                    clean_path = os.path.join(out_dir, f"{base}_clean.txt")
                    with open(clean_path, "w", encoding="utf-8") as f:
                        f.write(clean_text)
                    self._qlog(f"Cleaned output: {clean_path}")

                llm_clean_path = ""
                llm_calls = 0
                llm_errors = 0
                if llm_clean and clean_output and clean_path:
                    llm_text, llm_calls, llm_errors = llm_clean_text(
                        clean_text,
                        deepseek_key,
                        deepseek_base,
                        deepseek_model,
                        skip_tables_figures=llm_skip_tables,
                    )
                    if llm_text:
                        llm_clean_path = os.path.join(out_dir, f"{base}_llm_clean.txt")
                        with open(llm_clean_path, "w", encoding="utf-8") as f:
                            f.write(llm_text)
                        self._qlog(f"LLM cleaned output: {llm_clean_path}")

                media_json_path = ""
                media_txt_path = ""
                if media_items:
                    media_json_path = os.path.join(out_dir, f"{base}_figure_table.json")
                    media_txt_path = os.path.join(out_dir, f"{base}_figure_table.txt")
                    with open(media_json_path, "w", encoding="utf-8") as f:
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
                    with open(media_txt_path, "w", encoding="utf-8") as f:
                        for item in media_items:
                            title = (item.get("title") or "").strip()
                            f.write(
                                f"Page {item.get('page', 0):03d} "
                                f"[{item.get('type', '')}] "
                                f"bbox={item.get('bbox', [])} "
                                f"image={item.get('image_path', '')}\n"
                            )
                            if title:
                                f.write("Title:\n" + title + "\n")
                            f.write(f"PDF page: {item.get('pdf_page', item.get('page', 0))}\n")
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

                    self._qlog(f"Figure/Table output: {media_txt_path}")
                    self._qlog(f"Figure/Table JSON: {media_json_path}")
                    if self.preview_media_var.get():
                        self._queue.put(("media_preview", media_items))

                self._qprogress_files(idx, total_files)

                report_lines.append(f"=== File: {os.path.basename(pdf_path)} ===")
                report_lines.append(f"Mode: {ocr_mode}")
                report_lines.append(f"Layout detection: requested={layout_requested} active={use_layout}")
                report_lines.append(f"Layout engine: {layout_kind or 'none'}")
                report_lines.append(f"Table OCR: {use_table}")
                report_lines.append(f"Vocab correction: {use_vocab}")
                report_lines.append(f"Preprocess: {preprocess_mode} (ink_ratio_max={ink_ratio_threshold})")
                report_lines.append(f"Fallback gray/bw: {use_fallback}")
                report_lines.append(
                    f"Auto language: {auto_lang} (hangul_ratio>={hangul_ratio_threshold}, margin={lang_switch_margin})"
                )
                report_lines.append(
                    f"Skip low-text: {skip_low_text} (min_chars={min_chars}, min_score={min_score})"
                )
                report_lines.append(
                    f"Rescue: short_line_ratio>={rescue_short_ratio}, psm_text={rescue_psm_text}, psm_table={rescue_psm_table}"
                )
                report_lines.append(
                    f"Remove TOC: {remove_toc} (max_pages={toc_max_pages}, min_keep={toc_min_keep})"
                )
                report_lines.append(
                    f"DeepSeek summaries: {use_deepseek} (tables={summarize_tables}, figures={summarize_figures})"
                )
                report_lines.append(f"Clean output: {clean_output}")
                report_lines.append(f"LLM clean output: {llm_clean} (skip_tables_figures={llm_skip_tables})")
                report_lines.append(f"Fast mode: {fast_mode}")
                report_lines.append(f"Poppler path: {poppler_path or 'not found'}")
                report_lines.append(f"Memory cleanup: {mem_cleanup}")
                if media_items:
                    report_lines.append(f"Figure/Table output: {os.path.basename(media_txt_path)}")
                    report_lines.append(f"Figure/Table JSON: {os.path.basename(media_json_path)}")
                report_lines.append(f"Total pages: {len(pages)}")

                overall = _page_stats(text)
                report_lines.append(
                    f"Overall score: {overall['score']} ({overall['status']}) | chars={overall['total']} hangul={overall['hangul']} latin={overall['latin']} digits={overall['digits']} other={overall['other']}"
                )

                if page_confs:
                    avg_conf = int(round(sum(page_confs) / len(page_confs)))
                    low_sum = int(sum(page_lows))
                    retry_count = sum(1 for v in page_retries if v)
                    fallback_total = sum(page_fallbacks) if page_fallbacks else 0
                    lang_switch_total = sum(page_lang_switches) if page_lang_switches else 0
                    rescue_total = sum(1 for v in page_rescues if v) if page_rescues else 0
                    report_lines.append(
                        f"OCR confidence: avg={avg_conf} low_words={low_sum} retries={retry_count} fallbacks={fallback_total} lang_switches={lang_switch_total} rescues={rescue_total}"
                    )
                if llm_clean:
                    report_lines.append(f"LLM clean: calls={llm_calls} errors={llm_errors} output={os.path.basename(llm_clean_path) if llm_clean_path else ''}")
                if media_items:
                    table_count = sum(1 for item in media_items if item.get("type") == "table")
                    figure_count = sum(1 for item in media_items if item.get("type") == "figure")
                    report_lines.append(
                        f"Media items: {len(media_items)} (tables={table_count}, figures={figure_count})"
                    )
                    report_lines.append(
                        f"DeepSeek: calls={deepseek_calls} errors={deepseek_errors}"
                    )

                for page_index, page_text in enumerate(pages, start=1):
                    stats = page_stats_raw[page_index - 1] if page_stats_raw else _page_stats(page_text)
                    conf_line = ""
                    if page_confs:
                        conf_line = (
                            f" conf={page_confs[page_index-1]}"
                            f" low={page_lows[page_index-1]}"
                            f" retry={page_retries[page_index-1]}"
                            f" fallback={page_fallbacks[page_index-1]}"
                            f" lang_switch={page_lang_switches[page_index-1]}"
                            f" skipped={page_skipped[page_index-1]}"
                            f" rescue={page_rescues[page_index-1]}"
                            f" rescue_reason={page_rescue_reasons[page_index-1] or 'none'}"
                            f" toc_removed={page_toc_removed[page_index-1]}"
                        )
                    report_lines.append(
                        f"Page {page_index:03d}: score={stats['score']} ({stats['status']}) chars={stats['total']} hangul={stats['hangul']} latin={stats['latin']} digits={stats['digits']} other={stats['other']} short_lines={stats['short_lines']}/{stats['lines']}{conf_line}"
                    )
                report_lines.append("")

            if report_lines:
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(report_lines))
                self._qlog(f"Quality report saved: {report_path}")

            self._qlog("Done.")
            self._queue.put(("done",))
        except Exception as exc:
            if "Cancelled by user" in str(exc):
                self._qlog("Stopped by user.")
                self._queue.put(("done",))
                return
            err_text = "ERROR: " + str(exc)
            self._qlog(err_text)
            self._qlog(traceback.format_exc())
            self._write_error_log(out_dir, err_text)
            self._queue.put(("done",))

    def _write_error_log(self, out_dir, err_text):
        try:
            log_path = os.path.join(out_dir, "ocr_tool_error.log") if out_dir else os.path.join(os.getcwd(), "ocr_tool_error.log")
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(err_text + "\n\n")
                f.write(traceback.format_exc())
            self._qlog(f"Error log saved: {log_path}")
        except Exception:
            pass


if __name__ == "__main__":
    app = App()
    app.mainloop()
