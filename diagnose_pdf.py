#!/usr/bin/env python3
import os
import sys
from shutil import which

try:
    from pdf2image import convert_from_path, pdfinfo_from_path
except Exception as exc:
    print("ERROR: pdf2image not installed:", exc)
    sys.exit(1)

try:
    import pytesseract
except Exception:
    pytesseract = None


def detect_poppler_path():
    candidates = [
        os.environ.get("POPPLER_PATH"),
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


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pdf.py /path/to/file.pdf")
        sys.exit(2)

    pdf_path = sys.argv[1]
    if not os.path.isfile(pdf_path):
        print("ERROR: file not found:", pdf_path)
        sys.exit(2)

    poppler_path = detect_poppler_path()
    print("Poppler path:", poppler_path or "NOT FOUND")

    try:
        info = pdfinfo_from_path(pdf_path, poppler_path=poppler_path)
        print("pdfinfo pages:", info.get("Pages"))
    except Exception as exc:
        print("pdfinfo error:", exc)

    try:
        images = convert_from_path(
            pdf_path,
            dpi=200,
            first_page=1,
            last_page=1,
            poppler_path=poppler_path,
        )
        print("convert_from_path pages:", len(images))
    except Exception as exc:
        print("convert_from_path error:", exc)
        sys.exit(1)

    if not images:
        print("ERROR: no images returned")
        sys.exit(1)

    img = images[0]
    temp_path = "/tmp/diagnose_page1.png"
    try:
        img.save(temp_path)
        print("Saved first page image:", temp_path, "size:", os.path.getsize(temp_path))
    except Exception as exc:
        print("image save error:", exc)

    if pytesseract is None:
        print("pytesseract not installed; skipping OCR test.")
        sys.exit(0)

    try:
        text = pytesseract.image_to_string(img, lang="eng")
        print("tesseract sample chars:", len(text))
        print("tesseract sample preview:", repr(text[:200]))
    except Exception as exc:
        print("tesseract error:", exc)


if __name__ == "__main__":
    main()
