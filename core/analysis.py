
import re
from collections import Counter
from core.text_processor import split_paragraphs, split_sentences


def analyze_text(text):
    """
    Generate descriptive linguistic statistics.
    These features do NOT determine AI authorship.
    """

    paragraphs = split_paragraphs(text)
    sentences = split_sentences(text)

    words = re.findall(r"\b[\w'-]+\b", text.lower())

    word_count = len(words)
    unique_words = len(set(words))

    vocabulary_diversity = (
        unique_words / word_count if word_count else 0
    )

    sentence_lengths = [
        len(re.findall(r"\b[\w'-]+\b", sentence))
        for sentence in sentences
    ]

    paragraph_lengths = [
        len(re.findall(r"\b[\w'-]+\b", paragraph))
        for paragraph in paragraphs
    ]

    average_sentence_length = (
        sum(sentence_lengths) / len(sentence_lengths)
        if sentence_lengths else 0
    )

    average_paragraph_length = (
        sum(paragraph_lengths) / len(paragraph_lengths)
        if paragraph_lengths else 0
    )

    frequent_words = Counter(words).most_common(15)

    return {
        "word_count": word_count,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "unique_words": unique_words,
        "vocabulary_diversity": round(vocabulary_diversity, 3),
        "average_sentence_length": round(
            average_sentence_length, 2
        ),
        "average_paragraph_length": round(
            average_paragraph_length, 2
        ),
        "frequent_words": frequent_words,
        "sentence_lengths": sentence_lengths,
        "paragraph_lengths": paragraph_lengths,
    }
