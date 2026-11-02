import re
import html
import unicodedata


def clean_text(text):
    """
    Normalize text before passing it to machine learning models.

    Preserves meaningful punctuation and paragraph structure.
    """

    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Normalize Unicode characters
    text = unicodedata.normalize("NFC", text)

    # Decode HTML entities such as &amp;
    text = html.unescape(text)

    # Convert HTML line-break tags into newlines
    text = re.sub(
        r"<\s*br\s*/?\s*>",
        "\n",
        text,
        flags=re.IGNORECASE
    )

    # Convert common block-level HTML tags into newlines
    text = re.sub(
        r"</?\s*(p|div|li|h[1-6])\b[^>]*>",
        "\n",
        text,
        flags=re.IGNORECASE
    )

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Normalize tabs and repeated spaces
    text = text.replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)

    # Remove unnecessary spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    # Preserve paragraph boundaries
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text_batch(texts):
    """
    Apply consistent preprocessing to a collection of texts.
    """

    return [clean_text(text) for text in texts]