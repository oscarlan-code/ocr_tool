import base64
import json
import tempfile
from pathlib import Path
from typing import Any, Dict
import sys

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from media_extract_batch import run_batched


app = FastAPI(title="OCR Media Extract API", version="1.0.0")
PIPELINE_VERSION = "eval-ui-media-batch-v1"


def _to_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return default


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/extract-media")
async def extract_media(
    file: UploadFile = File(...),
    pages: str = Form(""),
    batch_size: int = Form(6),
    dpi: int = Form(170),
    lang: str = Form("eng"),
    figure_ocr: str = Form("false"),
    use_text_layer_first: str = Form("true"),
    layout_backend: str = Form("auto"),
) -> Dict[str, Any]:
    suffix = Path(file.filename or "document.pdf").suffix or ".pdf"
    if suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF input is supported")

    with tempfile.TemporaryDirectory(prefix="media-api-") as tmpdir:
        tmp_path = Path(tmpdir)
        pdf_path = tmp_path / f"input{suffix}"
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        pdf_path.write_bytes(content)

        try:
            out_json = run_batched(
                pdf_path=str(pdf_path),
                out_dir=str(out_dir),
                page_spec=(pages or "").strip(),
                batch_size=max(1, int(batch_size)),
                dpi=max(72, int(dpi)),
                lang=(lang or "eng").strip() or "eng",
                include_figure_ocr=_to_bool(figure_ocr, False),
                use_text_layer_first=_to_bool(use_text_layer_first, True),
                layout_backend=(layout_backend or "auto").strip() or "auto",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"media extraction failed: {exc}") from exc

        payload = _load_json(Path(out_json))
        images = []

        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            image_path = Path(str(item.get("image_path") or "").strip())
            if not image_path.is_file():
                continue
            try:
                encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            except Exception:
                continue
            images.append(
                {
                    "filename": image_path.name,
                    "content_base64": encoded,
                }
            )
            item["image_path"] = image_path.name

        return {
            "media_payload": payload,
            "images": images,
            "pipeline_version": PIPELINE_VERSION,
        }
