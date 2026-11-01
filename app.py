
import re
import streamlit as st
import pandas as pd
from collections import Counter
from docx import Document

st.set_page_config(
    page_title="K-Detect",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 K-Detect")
st.caption("Transparent academic writing analysis")

st.info(
    "K-Detect is an experimental writing-analysis tool. "
    "Its findings are not proof of AI authorship."
)

def extract_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def analyze_text(text):
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    sentences = [
        s.strip() for s in re.split(r"[.!?]+", text) if s.strip()
    ]

    word_count = len(words)
    sentence_count = len(sentences)
    unique_words = len(set(words))

    sentence_lengths = [
        len(re.findall(r"\b[\w'-]+\b", s))
        for s in sentences
    ]

    vocabulary_diversity = (
        unique_words / word_count if word_count else 0
    )

    average_sentence_length = (
        sum(sentence_lengths) / sentence_count
        if sentence_count else 0
    )

    sentence_variation = (
        pd.Series(sentence_lengths).std()
        if sentence_count > 1 else 0
    )

    common_words = Counter(words).most_common(10)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "unique_words": unique_words,
        "vocabulary_diversity": vocabulary_diversity,
        "average_sentence_length": average_sentence_length,
        "sentence_variation": sentence_variation,
        "common_words": common_words
    }

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "docx"]
)

text_input = st.text_area(
    "Or paste your academic writing here",
    height=250,
    placeholder="Paste your text here..."
)

text = text_input

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".txt"):
        text = uploaded_file.getvalue().decode(
            "utf-8", errors="replace"
        )
    elif uploaded_file.name.lower().endswith(".docx"):
        text = extract_docx(uploaded_file)

if st.button("Analyze Text", type="primary"):
    if not text.strip():
        st.warning("Please upload or enter some text first.")
    else:
        results = analyze_text(text)

        st.subheader("Analysis Results")

        col1, col2, col3 = st.columns(3)

        col1.metric("Words", results["word_count"])
        col2.metric("Sentences", results["sentence_count"])
        col3.metric("Unique Words", results["unique_words"])

        st.subheader("Linguistic Features")

        st.write(
            "Vocabulary diversity:",
            f'{results["vocabulary_diversity"]:.3f}'
        )

        st.write(
            "Average sentence length:",
            f'{results["average_sentence_length"]:.2f} words'
        )

        st.write(
            "Sentence length variation:",
            f'{results["sentence_variation"]:.2f}'
        )

        st.subheader("Most Frequent Words")

        common_df = pd.DataFrame(
            results["common_words"],
            columns=["Word", "Frequency"]
        )

        st.dataframe(common_df, use_container_width=True)

        st.warning(
            "These measurements describe writing characteristics. "
            "They do not establish whether a text was written by AI."
        )

        report = pd.DataFrame([{
            key: value
            for key, value in results.items()
            if key != "common_words"
        }])

        st.download_button(
            "Download Analysis Report",
            report.to_csv(index=False),
            file_name="k_detect_report.csv",
            mime="text/csv"
        )

st.divider()

st.caption(
    "K-Detect | Open-source research project | "
    "Early development version"
)
