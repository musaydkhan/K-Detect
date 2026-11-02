
import re


def normalize_text(text):
    """
    Normalize extracted text while preserving paragraph structure.
    """

    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Replace tabs and repeated spaces
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n[ \t]*\n+", "\n\n", text)

    return text.strip()


def split_paragraphs(text):
    """
    Divide a document into paragraphs.
    """

    text = normalize_text(text)

    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def split_sentences(text):
    """
    Basic sentence segmentation.
    This initial version uses punctuation-based splitting.
    """

    text = normalize_text(text)

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9"“])',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]
