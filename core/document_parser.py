
import io
from docx import Document


def extract_text(uploaded_file):
    """
    Extract text from supported document formats.
    Supports DOCX and TXT files.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".txt"):
        content = uploaded_file.getvalue()
        return content.decode("utf-8", errors="replace")

    elif filename.endswith(".docx"):
        uploaded_file.seek(0)
        document = Document(uploaded_file)

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        return "\n\n".join(paragraphs)

    else:
        raise ValueError(
            "Unsupported file format. Please upload a DOCX or TXT file."
        )
