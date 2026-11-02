
import re
import io
import math
from collections import Counter
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
from docx import Document

from docx.enum.text import WD_COLOR_INDEX



def create_annotated_docx(uploaded_file, highlighted_sentences):
    """
    Create a highlighted copy of an uploaded DOCX.
    Demonstration version: highlights the paragraph
    containing the first selected sentence.
    """

    uploaded_file.seek(0)
    document = Document(uploaded_file)

    if highlighted_sentences:
        target = " ".join(highlighted_sentences[0].split())

        for paragraph in document.paragraphs:
            paragraph_text = " ".join(paragraph.text.split())

            if target and target in paragraph_text:
                for run in paragraph.runs:
                    run.font.highlight_color = WD_COLOR_INDEX.YELLOW
                break

    output = io.BytesIO()
    document.save(output)
    output.seek(0)

    return output.getvalue()


from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# --------------------------------------------------
# APPLICATION CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="K-Detect | Academic Writing Analysis",
    page_icon="🔬",
    layout="wide"
)

st.title("K-Detect")
st.caption("Academic Writing Analysis and Linguistic Assessment")

st.info(
    "K-Detect is an experimental academic writing analysis tool. "
    "Its linguistic measurements are descriptive indicators, "
    "not proof of AI authorship or academic misconduct."
)


# --------------------------------------------------
# TEXT PROCESSING
# --------------------------------------------------

def extract_docx(uploaded_file):
    document = Document(uploaded_file)

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]

    return "\n\n".join(paragraphs)


def split_paragraphs(text):
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]


def split_sentences(text):
    """
    A lightweight rule-based sentence splitter.
    Abbreviations and unusual punctuation may affect accuracy.
    """
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def tokenize(text):
    return re.findall(r"\b[\w'-]+\b", text.lower())


def estimate_syllables(word):
    """
    Approximate English syllable count.
    This is a heuristic, not a phonetic analysis.
    """
    word = re.sub(r"[^a-z]", "", word.lower())

    if not word:
        return 0

    if len(word) <= 3:
        return 1

    word = re.sub(r"e$", "", word)

    groups = re.findall(r"[aeiouy]+", word)

    return max(1, len(groups))


def readability_metrics(words, sentences):
    if not words or not sentences:
        return {
            "flesch_reading_ease": None,
            "gunning_fog": None
        }

    total_words = len(words)
    total_sentences = len(sentences)

    syllables = sum(estimate_syllables(word) for word in words)

    complex_words = sum(
        1 for word in words
        if estimate_syllables(word) >= 3
    )

    average_sentence_length = total_words / total_sentences
    average_syllables = syllables / total_words

    flesch = (
        206.835
        - 1.015 * average_sentence_length
        - 84.6 * average_syllables
    )

    fog = 0.4 * (
        average_sentence_length
        + 100 * complex_words / total_words
    )

    return {
        "flesch_reading_ease": round(flesch, 2),
        "gunning_fog": round(fog, 2)
    }


# --------------------------------------------------
# LINGUISTIC ANALYSIS
# --------------------------------------------------

STOPWORDS = set("""
a an the and or but if while with without of at by for from
to in on into onto is am are was were be been being
have has had do does did this that these those it its
as not no nor so than then too very can could will would
should may might must shall i we you he she they them
their our your my me him her us who whom which what when
where why how all any both each few more most other some
such only own same also just about over under again further
once here there
""".split())


TRANSITION_GROUPS = {
    "Addition": [
        "additionally", "furthermore", "moreover",
        "also", "besides", "in addition"
    ],
    "Contrast": [
        "however", "nevertheless", "nonetheless",
        "although", "whereas", "conversely",
        "on the other hand"
    ],
    "Cause and effect": [
        "therefore", "consequently", "thus",
        "hence", "as a result", "accordingly"
    ],
    "Examples": [
        "for example", "for instance",
        "such as", "namely"
    ],
    "Conclusion": [
        "in conclusion", "to conclude",
        "overall", "in summary"
    ]
}


def count_transitions(text):
    lowered = text.lower()
    results = []

    for category, phrases in TRANSITION_GROUPS.items():
        for phrase in phrases:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            count = len(re.findall(pattern, lowered))

            if count:
                results.append({
                    "Category": category,
                    "Transition": phrase,
                    "Frequency": count
                })

    return pd.DataFrame(
        results,
        columns=["Category", "Transition", "Frequency"]
    ).sort_values(
        "Frequency",
        ascending=False
    ) if results else pd.DataFrame(
        columns=["Category", "Transition", "Frequency"]
    )


def repeated_phrases(words, n):
    if len(words) < n:
        return []

    phrases = [
        " ".join(words[i:i+n])
        for i in range(len(words) - n + 1)
    ]

    counts = Counter(phrases)

    return [
        {"Phrase": phrase, "Frequency": count}
        for phrase, count in counts.most_common(10)
        if count > 1
    ]


def analyze_text(text):
    words = tokenize(text)
    paragraphs = split_paragraphs(text)

    sentences = []
    for paragraph in paragraphs:
        sentences.extend(split_sentences(paragraph))

    word_count = len(words)
    sentence_count = len(sentences)
    paragraph_count = len(paragraphs)
    unique_words = len(set(words))

    sentence_lengths = [
        len(tokenize(sentence))
        for sentence in sentences
    ]

    paragraph_lengths = [
        len(tokenize(paragraph))
        for paragraph in paragraphs
    ]

    average_sentence_length = (
        sum(sentence_lengths) / sentence_count
        if sentence_count else 0
    )

    average_paragraph_length = (
        sum(paragraph_lengths) / paragraph_count
        if paragraph_count else 0
    )

    sentence_std = (
        pd.Series(sentence_lengths).std()
        if sentence_count > 1 else 0
    )

    vocabulary_diversity = (
        unique_words / word_count
        if word_count else 0
    )

    filtered_words = [
        word for word in words
        if word not in STOPWORDS and len(word) > 1
    ]

    frequent_words = Counter(filtered_words).most_common(15)

    readability = readability_metrics(words, sentences)

    short_sentences = sum(
        1 for length in sentence_lengths if length <= 10
    )

    medium_sentences = sum(
        1 for length in sentence_lengths if 11 <= length <= 20
    )

    long_sentences = sum(
        1 for length in sentence_lengths if length > 20
    )

    paragraph_data = []

    for index, paragraph in enumerate(paragraphs, start=1):
        paragraph_sentences = split_sentences(paragraph)
        paragraph_words = tokenize(paragraph)

        paragraph_data.append({
            "Paragraph": index,
            "Words": len(paragraph_words),
            "Sentences": len(paragraph_sentences),
            "Average sentence length": round(
                len(paragraph_words) / len(paragraph_sentences), 2
            ) if paragraph_sentences else 0
        })

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "paragraph_count": paragraph_count,
        "unique_words": unique_words,
        "vocabulary_diversity": vocabulary_diversity,
        "average_sentence_length": average_sentence_length,
        "average_paragraph_length": average_paragraph_length,
        "sentence_variation": sentence_std,
        "sentence_lengths": sentence_lengths,
        "paragraph_lengths": paragraph_lengths,
        "short_sentences": short_sentences,
        "medium_sentences": medium_sentences,
        "long_sentences": long_sentences,
        "frequent_words": frequent_words,
        "paragraph_data": paragraph_data,
        "readability": readability,
        "repeated_bigrams": repeated_phrases(words, 2),
        "repeated_trigrams": repeated_phrases(words, 3),
        "transitions": count_transitions(text),
        "longest_sentences": sorted(
            [
                {
                    "Sentence": sentence,
                    "Words": len(tokenize(sentence))
                }
                for sentence in sentences
            ],
            key=lambda item: item["Words"],
            reverse=True
        )[:5]
    }


# --------------------------------------------------
# PDF REPORT GENERATION
# --------------------------------------------------

def pdf_safe(value):
    """
    ReportLab's built-in fonts have limited Unicode coverage.
    Replace unsupported characters rather than breaking PDF creation.
    """
    return str(value).encode(
        "latin-1", errors="replace"
    ).decode("latin-1")


def make_pdf(results):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#17365D"),
        spaceBefore=13,
        spaceAfter=7
    ))

    styles.add(ParagraphStyle(
        name="SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11
    ))

    story = []

    story.append(Paragraph("K-Detect", styles["ReportTitle"]))
    story.append(Paragraph(
        "Academic Writing Analysis Report",
        styles["Heading2"]
    ))
    story.append(Spacer(1, 8))

    story.append(Paragraph(
        "Report scope: descriptive linguistic analysis. "
        "This report does not determine whether a document "
        "was written by a human or an AI system.",
        styles["BodyText"]
    ))

    story.append(Paragraph(
        "1. Document Overview",
        styles["SectionHeading"]
    ))

    overview = [
        ["Measurement", "Result"],
        ["Total words", results["word_count"]],
        ["Total sentences", results["sentence_count"]],
        ["Total paragraphs", results["paragraph_count"]],
        ["Unique words", results["unique_words"]],
        [
            "Vocabulary diversity",
            f'{results["vocabulary_diversity"]:.3f}'
        ],
        [
            "Average sentence length",
            f'{results["average_sentence_length"]:.2f} words'
        ],
        [
            "Average paragraph length",
            f'{results["average_paragraph_length"]:.2f} words'
        ],
        [
            "Sentence length standard deviation",
            f'{results["sentence_variation"]:.2f}'
        ]
    ]

    table = Table(overview, colWidths=[95 * mm, 65 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#F2F5F9")]),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)

    story.append(Paragraph(
        "2. Sentence Structure",
        styles["SectionHeading"]
    ))

    sentence_table = [
        ["Sentence category", "Count"],
        ["Short (10 words or fewer)", results["short_sentences"]],
        ["Medium (11-20 words)", results["medium_sentences"]],
        ["Long (more than 20 words)", results["long_sentences"]]
    ]

    table = Table(sentence_table, colWidths=[110 * mm, 50 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(table)

    story.append(Paragraph(
        "3. Readability Estimates",
        styles["SectionHeading"]
    ))

    flesch = results["readability"]["flesch_reading_ease"]
    fog = results["readability"]["gunning_fog"]

    story.append(Paragraph(
        f"Flesch Reading Ease: {flesch if flesch is not None else 'N/A'}",
        styles["BodyText"]
    ))

    story.append(Paragraph(
        f"Gunning Fog Index: {fog if fog is not None else 'N/A'}",
        styles["BodyText"]
    ))

    story.append(Paragraph(
        "These are approximate English-language readability "
        "estimates. They are not validated for every language, "
        "discipline, or academic writing context.",
        styles["SmallText"]
    ))

    story.append(Paragraph(
        "4. Paragraph-Level Analysis",
        styles["SectionHeading"]
    ))

    paragraph_table = [
        ["Paragraph", "Words", "Sentences", "Avg. sentence length"]
    ]

    for item in results["paragraph_data"]:
        paragraph_table.append([
            item["Paragraph"],
            item["Words"],
            item["Sentences"],
            item["Average sentence length"]
        ])

    table = Table(
        paragraph_table,
        colWidths=[30 * mm, 30 * mm, 35 * mm, 65 * mm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(table)

    story.append(Paragraph(
        "5. Frequent Vocabulary",
        styles["SectionHeading"]
    ))

    frequent_table = [["Word", "Frequency"]]

    for word, count in results["frequent_words"]:
        frequent_table.append([pdf_safe(word), count])

    if len(frequent_table) == 1:
        frequent_table.append(["No results", 0])

    table = Table(frequent_table, colWidths=[80 * mm, 40 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(table)

    story.append(Paragraph(
        "6. Repeated Phrases",
        styles["SectionHeading"]
    ))

    for heading, key in [
        ("Repeated two-word phrases", "repeated_bigrams"),
        ("Repeated three-word phrases", "repeated_trigrams")
    ]:
        story.append(Paragraph(heading, styles["Heading3"]))

        phrase_table = [["Phrase", "Frequency"]]

        for item in results[key]:
            phrase_table.append([
                pdf_safe(item["Phrase"]),
                item["Frequency"]
            ])

        if len(phrase_table) == 1:
            phrase_table.append(["No repeated phrases found", 0])

        table = Table(phrase_table, colWidths=[100 * mm, 40 * mm])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0),
             colors.HexColor("#DCE6F1")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
            ("PADDING", (0, 0), (-1, -1), 6)
        ]))

        story.append(table)
        story.append(Spacer(1, 6))

    story.append(Paragraph(
        "7. Transition-Word Analysis",
        styles["SectionHeading"]
    ))

    transition_table = [["Category", "Transition", "Frequency"]]

    for _, item in results["transitions"].iterrows():
        transition_table.append([
            item["Category"],
            item["Transition"],
            item["Frequency"]
        ])

    if len(transition_table) == 1:
        transition_table.append(["-", "No transitions detected", 0])

    table = Table(
        transition_table,
        colWidths=[55 * mm, 70 * mm, 35 * mm],
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.lightgrey),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(table)

    story.append(Paragraph(
        "8. Longest Sentences",
        styles["SectionHeading"]
    ))

    for index, item in enumerate(
        results["longest_sentences"], start=1
    ):
        sentence = escape(pdf_safe(item["Sentence"]))

        story.append(Paragraph(
            f"{index}. ({item['Words']} words) {sentence}",
            styles["SmallText"]
        ))
        story.append(Spacer(1, 5))

    story.append(Paragraph(
        "9. Methodological Notes and Limitations",
        styles["SectionHeading"]
    ))

    limitations = [
        "Vocabulary diversity is calculated as the ratio of unique "
        "tokens to total tokens. It is sensitive to document length.",

        "Sentence boundaries are estimated using punctuation and "
        "may be inaccurate for abbreviations or unusual formatting.",

        "Repeated phrases and transition words are descriptive "
        "features. Their presence does not establish writing origin.",

        "Readability scores use approximate English syllable counts "
        "and should be interpreted cautiously.",

        "K-Detect does not currently contain a validated AI-text "
        "classification model. No AI authorship probability is "
        "calculated by this version.",

        "These results should not be used as the sole basis for "
        "academic misconduct allegations or disciplinary decisions."
    ]

    for note in limitations:
        story.append(Paragraph(
            "- " + escape(note),
            styles["SmallText"]
        ))
        story.append(Spacer(1, 5))

    story.append(Spacer(1, 12))

    story.append(Paragraph(
        "K-Detect | Open-source research project | "
        "Experimental development version",
        styles["SmallText"]
    ))

    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------
# USER INTERFACE
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an academic document",
    type=["txt", "docx"],
    help="Supported formats: TXT and DOCX."
)

text_input = st.text_area(
    "Or paste your academic writing",
    height=250,
    placeholder="Paste your academic text here..."
)

text = text_input

if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith(".txt"):
            text = uploaded_file.getvalue().decode(
                "utf-8-sig", errors="replace"
            )

        elif uploaded_file.name.lower().endswith(".docx"):
            text = extract_docx(uploaded_file)

    except Exception as error:
        st.error(f"Could not read the document: {error}")


if st.button("Analyze Document", type="primary"):

    if not text.strip():
        st.warning("Please upload or enter some text first.")

    else:
        results = analyze_text(text)

        st.session_state["analysis_results"] = results
        st.session_state["analysis_text"] = text


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "analysis_results" in st.session_state:

    results = st.session_state["analysis_results"]

    st.divider()
    st.header("Academic Writing Analysis")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Words", results["word_count"])
    col2.metric("Sentences", results["sentence_count"])
    col3.metric("Paragraphs", results["paragraph_count"])
    col4.metric("Unique words", results["unique_words"])

    st.subheader("1. Document-Level Linguistic Features")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Vocabulary diversity",
        f'{results["vocabulary_diversity"]:.3f}'
    )

    col2.metric(
        "Average sentence length",
        f'{results["average_sentence_length"]:.2f} words'
    )

    col3.metric(
        "Average paragraph length",
        f'{results["average_paragraph_length"]:.2f} words'
    )

    st.write(
        "Sentence length standard deviation:",
        f'{results["sentence_variation"]:.2f}'
    )

    st.caption(
        "Vocabulary diversity is a simple type-token ratio. "
        "It tends to change with document length."
    )

    st.subheader("2. Sentence-Length Distribution")

    distribution = pd.DataFrame({
        "Sentence category": [
            "Short (≤10 words)",
            "Medium (11–20 words)",
            "Long (>20 words)"
        ],
        "Count": [
            results["short_sentences"],
            results["medium_sentences"],
            results["long_sentences"]
        ]
    })

    st.bar_chart(
        distribution.set_index("Sentence category")
    )

    st.subheader("3. Readability Estimates")

    readability = results["readability"]

    col1, col2 = st.columns(2)

    col1.metric(
        "Flesch Reading Ease",
        readability["flesch_reading_ease"]
        if readability["flesch_reading_ease"] is not None
        else "N/A"
    )

    col2.metric(
        "Gunning Fog Index",
        readability["gunning_fog"]
        if readability["gunning_fog"] is not None
        else "N/A"
    )

    st.caption(
        "Approximate English readability estimates. "
        "They are not measures of writing quality or AI authorship."
    )

    st.subheader("4. Paragraph-Level Analysis")

    paragraph_df = pd.DataFrame(results["paragraph_data"])

    st.dataframe(
        paragraph_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("5. Frequent Vocabulary")

    frequent_df = pd.DataFrame(
        results["frequent_words"],
        columns=["Word", "Frequency"]
    )

    if not frequent_df.empty:
        st.dataframe(
            frequent_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No frequent vocabulary results.")

    st.subheader("6. Repeated Phrases")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Repeated two-word phrases**")
        st.dataframe(
            pd.DataFrame(results["repeated_bigrams"]),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.write("**Repeated three-word phrases**")
        st.dataframe(
            pd.DataFrame(results["repeated_trigrams"]),
            use_container_width=True,
            hide_index=True
        )

    st.subheader("7. Transition-Word Analysis")

    if not results["transitions"].empty:
        st.dataframe(
            results["transitions"],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.write("No listed transition words were detected.")

    st.subheader("8. Longest Sentences")

    for index, item in enumerate(
        results["longest_sentences"], start=1
    ):
        with st.expander(
            f"Sentence {index} — {item['Words']} words"
        ):
            st.write(item["Sentence"])

    st.divider()

    st.subheader("Download Academic Report")

    pdf_data = make_pdf(results)

    st.download_button(
        label="Download Comprehensive PDF Report",
        data=pdf_data,
        file_name="K-Detect_Academic_Report.pdf",
        mime="application/pdf",
        type="primary"
    )

    # Demonstration: highlight the first sentence in yellow
    if (
        uploaded_file is not None
        and uploaded_file.name.lower().endswith(".docx")
    ):
        sentences = split_sentences(
            st.session_state.get("analysis_text", "")
        )

        if sentences:
            uploaded_file.seek(0)

            annotated_data = create_annotated_docx(
                uploaded_file,
                [sentences[0]]
            )

            st.download_button(
                label="Download Annotated DOCX (Demo)",
                data=annotated_data,
                file_name="K-Detect_Annotated_Demo.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

    metrics_data = {
        key: value
        for key, value in results.items()
        if key not in [
            "paragraph_data",
            "frequent_words",
            "repeated_bigrams",
            "repeated_trigrams",
            "transitions",
            "longest_sentences",
            "sentence_lengths",
            "paragraph_lengths"
        ]
    }

    metrics_data["flesch_reading_ease"] = (
        readability["flesch_reading_ease"]
    )
    metrics_data["gunning_fog"] = readability["gunning_fog"]

    metrics_csv = pd.DataFrame([metrics_data]).to_csv(
        index=False
    )

    st.download_button(
        label="Download Summary Data (CSV)",
        data=metrics_csv,
        file_name="K-Detect_Summary.csv",
        mime="text/csv"
    )

    paragraph_csv = paragraph_df.to_csv(index=False)

    st.download_button(
        label="Download Paragraph Analysis (CSV)",
        data=paragraph_csv,
        file_name="K-Detect_Paragraph_Analysis.csv",
        mime="text/csv"
    )

    st.warning(
        "These results describe linguistic characteristics. "
        "They do not establish whether a document was written "
        "by a human or an AI system."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "K-Detect | Open-source research project | "
    "Experimental development version"
)
