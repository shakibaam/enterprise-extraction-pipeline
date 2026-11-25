"""PDF to text with pypdfium2 (Google's PDFium; Apache-2.0/BSD). Chosen in enterprise_variant_d/runs/pdf_parser_bench_20261113: as fast as
PyMuPDF and reads the same files, without PyMuPDF's AGPL licence; pypdf and pdfminer.six were 9 to 60 times slower.

fetch.py runs this module in a separate process for every PDF (pdf_to_text), so a PDF that hangs or crashes the parser is stopped by a timeout
and can't take the fetcher down. The child reads the PDF bytes on stdin and writes one JSON line: {"pages": n, "pages_read": k, "text": "..."}.
Text cleanup is the same as for HTML pages: lines stripped, blank lines dropped.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROVINCES_DIR = Path(__file__).resolve().parents[2]   # so the child can import on_the_fly_pipeline


class PdfError(Exception):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category, self.message = category, message


def extract(data: bytes, max_pages: int = 0) -> dict:
    """Text of a PDF given as bytes (all pages, or the first max_pages)."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(data)
    try:
        n = len(doc)
        k = n if max_pages <= 0 else min(n, max_pages)
        parts = []
        for i in range(k):
            page = doc[i]
            tp = page.get_textpage()
            parts.append(tp.get_text_range())
            tp.close()
            page.close()
    finally:
        doc.close()
    text = "\n".join(parts).replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return {"pages": n, "pages_read": k, "text": text}


def pdf_to_text(data: bytes, timeout_s: float, max_pages: int = 0, python: str | None = None) -> dict:
    """Run extract() in a child process (python: an interpreter that has pypdfium2; default this one).
    Raises PdfError("pdf_timeout" | "pdf_error", message)."""
    try:
        p = subprocess.run([python or sys.executable, "-m", "on_the_fly_pipeline.shared.pdf_text", str(max_pages)], input=data,
                           capture_output=True, timeout=timeout_s, cwd=PROVINCES_DIR)
    except subprocess.TimeoutExpired as exc:
        raise PdfError("pdf_timeout", f"parsing took more than {timeout_s} s") from exc
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", "replace").strip().splitlines()
        raise PdfError("pdf_error", (err[-1] if err else f"parser exited with code {p.returncode}")[:300])
    return json.loads(p.stdout)


if __name__ == "__main__":
    out = extract(sys.stdin.buffer.read(), int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
