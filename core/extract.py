"""Read text from DOCX and PDF uploads. Nothing is written to disk."""
import io
from docx import Document
from pypdf import PdfReader


class ExtractionError(ValueError):
    pass


def _docx(data: bytes) -> str:
    try:
        doc = Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError("This DOCX file could not be opened. Re-save it from Word and try again.") from exc
    parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:                      # tables often hold real text in manuscripts
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _pdf(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ExtractionError("This PDF is password-protected. Remove the password and try again.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError("This PDF could not be read. Try exporting it again.") from exc
    text = "\n".join(pages)
    if len(text.split()) < 30:
        raise ExtractionError("No readable text found. This PDF looks scanned (images only). "
                              "Export a text-based PDF, or paste the text instead.")
    # join hard-wrapped lines: a single newline inside a paragraph becomes a space
    lines = [l.strip() for l in text.split("\n")]
    lens = sorted(len(l) for l in lines if l)
    typical = lens[len(lens) * 3 // 4] if lens else 80      # long-line length ~ column width
    out, buf = [], []
    for line in lines:
        if line:
            buf.append(line)
            # a short line that ends a sentence usually ends the paragraph
            if len(line) < typical * 0.6 and line[-1] in ".!?":
                out.append(" ".join(buf)); buf = []
        elif buf:
            out.append(" ".join(buf)); buf = []
    if buf:
        out.append(" ".join(buf))
    return "\n\n".join(out)


def extract_text(name: str, data: bytes, max_mb: int = 10) -> str:
    if len(data) > max_mb * 1024 * 1024:
        raise ExtractionError(f"File is larger than {max_mb} MB.")
    lower = name.lower()
    if lower.endswith(".docx"):
        return _docx(data)
    if lower.endswith(".pdf"):
        return _pdf(data)
    raise ExtractionError("Only DOCX and PDF files are supported.")
