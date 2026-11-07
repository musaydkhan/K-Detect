"""PDF report and annotated DOCX. Header on every page: 'K-Detect | ...'."""
import io, re
from xml.sax.saxutils import escape
from docx import Document
from docx.enum.text import WD_COLOR_INDEX
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, PageBreak, KeepTogether
from config import app_config as cfg
from core.pipeline import paragraph_runs

INK, MUTED = colors.HexColor("#14263B"), colors.HexColor("#5B6B7C")
HL = {"generated": "#F6B8AE", "revised": "#FAD98A", "possible": "#E3EBF3"}
SIM_COLOR = "#6C4FD6"
GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "κ": "kappa", "λ": "lambda", "μ": "u", "σ": "sigma",
         "≥": ">=", "≤": "<=", "≈": "~", "→": "->", "–": "-", "—": "-", "−": "-", "‑": "-", "’": "'", "‘": "'",
         "“": '"', "”": '"', "…": "...", "•": "-", "×": "x", "°": " deg", "±": "+/-", "\u200b": ""}


def safe(text: str) -> str:
    """Built-in PDF fonts are Latin-1 only: map common symbols, replace the rest."""
    for k, v in GREEK.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _label(s):
    names = {"generated": "Likely AI-written", "revised": "Likely AI-edited", "possible": "Possible AI patterns"}
    return names.get(s.ai_label, "")


def _page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(INK); canvas.setFont("Helvetica-Bold", 8.5)
    canvas.drawString(18 * mm, A4[1] - 11 * mm, cfg.BRAND)
    canvas.setFont("Helvetica", 8.5); canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm + 42, A4[1] - 11 * mm, "| " + cfg.TAGLINE)
    canvas.setStrokeColor(colors.HexColor("#D5DEE8")); canvas.line(18 * mm, A4[1] - 13 * mm, A4[0] - 18 * mm, A4[1] - 13 * mm)
    canvas.drawString(18 * mm, 9 * mm, "A likelihood, not proof. Do not use as the sole basis for any misconduct decision.")
    canvas.drawRightString(A4[0] - 18 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def make_pdf(a) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=20 * mm,
                            bottomMargin=16 * mm, title="K-Detect report", author=cfg.BRAND)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=21, leading=25, textColor=INK, alignment=TA_LEFT, spaceAfter=4)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12.5, leading=16, textColor=INK, spaceBefore=14, spaceAfter=6)
    B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=14, textColor=INK)
    S = ParagraphStyle("S", parent=B, fontSize=8, leading=11, textColor=MUTED)
    DOC = ParagraphStyle("DOC", parent=B, fontName="Times-Roman", fontSize=10.5, leading=16, spaceAfter=8)
    st = []
    st += [Paragraph("Writing check report", H1),
           Paragraph(safe(f"Generated {a.created}  |  {a.word_count:,} words checked"), S), Spacer(1, 8)]

    # ---- summary
    cells = []
    if a.ai is not None:
        v = a.ai
        big = "Low signal" if v["low_signal"] else f"{v['ai_share']:.0f}%"
        cells.append([Paragraph(f"<font size=20><b>{big}</b></font>", B),
                      Paragraph("<b>AI-writing likelihood</b><br/>"
                                f"Likely AI-written {v['generated']:.0f}%  |  Likely AI-edited {v['revised']:.0f}%  |  "
                                f"Possible {v['possible']:.0f}%", B)])
    if a.sim is not None and not a.sim.available:
        cells.append([Paragraph("<font size=16><b>n/a</b></font>", B),
                      Paragraph("<b>Similarity to open sources</b><br/>No search service could be reached, so there is no similarity result. This is not a clean result.", B)])
    elif a.sim is not None:
        cells.append([Paragraph(f"<font size=20><b>{a.sim.similarity_pct:.0f}%</b></font>", B),
                      Paragraph("<b>Similarity to open sources</b><br/>"
                                f"{len(a.sim.sources)} matching source(s) found. Paywalled papers and other students' work are not searched.", B)])
    if cells:
        t = Table(cells, colWidths=[34 * mm, 140 * mm])
        t.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#C9D5E2")),
                               ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#E0E7EF")),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("PADDING", (0, 0), (-1, -1), 9),
                               ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F2F6FA"))]))
        st.append(t)
    for n in a.notes:
        st.append(Paragraph(safe("Note: " + n), S))
    legend = (f'<font backColor="{HL["generated"]}">&nbsp;Likely AI-written&nbsp;</font>&nbsp; '
              f'<font backColor="{HL["revised"]}">&nbsp;Likely AI-edited&nbsp;</font>&nbsp; '
              f'<font backColor="{HL["possible"]}">&nbsp;Possible AI patterns&nbsp;</font>&nbsp; '
              f'<font color="{SIM_COLOR}"><u>Similar to a source</u></font>')
    st += [Spacer(1, 8), Paragraph(legend, B)]

    # ---- annotated text
    st.append(Paragraph("Your text, annotated", H2))
    for runs in paragraph_runs(a):
        parts = []
        for text, s in runs:
            t = escape(safe(text))
            if s is not None:
                if s.ai_label in HL:
                    t = f'<font backColor="{HL[s.ai_label]}">{t}</font>'
                if s.sim_score:
                    t = f'<font color="{SIM_COLOR}"><u>{t}</u></font>'
            parts.append(t)
        st.append(Paragraph("".join(parts), DOC))

    # ---- sources
    if a.sim is not None:
        st.append(Paragraph("Matching sources", H2))
        if a.sim.sources:
            rows = [["#", "Source", "Where", "Matched words"]]
            for i, s in enumerate(a.sim.sources, 1):
                rows.append([str(i), Paragraph(escape(safe(s.title[:110])) + f"<br/><font size=7 color='#5B6B7C'>{escape(safe(s.url[:90]))}</font>", S),
                             s.origin, str(s.matched_words)])
            t = Table(rows, colWidths=[8 * mm, 112 * mm, 28 * mm, 26 * mm], repeatRows=1)
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                   ("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DEE8")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 5)]))
            st.append(t)
        else:
            st.append(Paragraph("No matching passages were found in the sources that were searched.", B))

    # ---- flagged passages
    if a.ai is not None:
        flagged = sorted([s for s in a.sentences if s.ai_label in ("generated", "revised")], key=lambda s: -s.p_ai)[:12]
        if flagged:
            st.append(Paragraph("Strongest AI-pattern passages", H2))
            for s in flagged:
                st.append(KeepTogether([Paragraph(safe(f"<b>{_label(s)}</b> (score {s.p_ai:.2f}): ") + escape(safe(s.text[:320])), S), Spacer(1, 4)]))

    # ---- writing profile
    p = a.profile
    st.append(Paragraph("Writing profile (descriptive only)", H2))
    rows = [["Measure", "Value"], ["Words / sentences", f"{p['words']:,} / {p['sentences']}"],
            ["Lexical diversity (type-token ratio)", f"{p['lexical_diversity']}"],
            ["Average sentence length", f"{p['avg_sentence_len']} words (SD {p['sentence_len_sd']})"],
            ["Flesch reading ease / Gunning fog", f"{p['flesch']} / {p['fog']}"]]
    t = Table(rows, colWidths=[90 * mm, 84 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), INK), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                           ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#D5DEE8")),
                           ("PADDING", (0, 0), (-1, -1), 5)]))
    st += [t, Paragraph("These measures describe the text. They do not show who or what wrote it.", S)]

    # ---- limits
    st.append(Paragraph("What this report can and cannot tell you", H2))
    for line in [
        "AI-writing scores are statistical likelihoods from a model trained on scientific abstracts. They are not proof of AI use.",
        "Detection is strongest on text resembling the training data and on recent chat-style generators; it is weak on some older or less common generators.",
        "Non-native English writing and heavily edited text can be mis-scored. Treat flags as a prompt to review, not a verdict.",
        "The similarity check covers open scholarly sources and, when enabled, a limited web search. It does not search paywalled journals, books, or other students' submissions.",
        "Do not use this report as the sole basis for an academic-misconduct decision."]:
        st.append(Paragraph("- " + safe(line), B))
    doc.build(st, onFirstPage=_page, onLaterPages=_page)
    return buf.getvalue()


def make_docx(a) -> bytes:
    d = Document()
    d.styles["Normal"].font.name = "Times New Roman"; d.styles["Normal"].font.size = Pt(11)
    sec = d.sections[0]
    sec.header.paragraphs[0].text = cfg.HEADER_LINE
    sec.footer.paragraphs[0].text = "A likelihood, not proof. Do not use as the sole basis for any misconduct decision."
    d.add_heading("Writing check: annotated text", level=1)
    legend = d.add_paragraph()
    for label, color in (("Likely AI-written", WD_COLOR_INDEX.PINK), ("Likely AI-edited", WD_COLOR_INDEX.YELLOW),
                         ("Possible AI patterns", WD_COLOR_INDEX.GRAY_25), ("Similar to a source", WD_COLOR_INDEX.TURQUOISE)):
        r = legend.add_run(f" {label} "); r.font.highlight_color = color; legend.add_run("  ")
    if a.ai is not None:
        d.add_paragraph(f"AI-writing likelihood: {'low signal' if a.ai['ai_share'] and a.ai['low_signal'] else str(round(a.ai['ai_share'])) + '%'}")
    if a.sim is not None:
        d.add_paragraph(f"Similarity to open sources: {a.sim.similarity_pct:.0f}%")
    for runs in paragraph_runs(a):
        p = d.add_paragraph()
        for text, s in runs:
            r = p.add_run(text)
            if s is not None:
                if s.sim_score:
                    r.font.highlight_color = WD_COLOR_INDEX.TURQUOISE
                elif s.ai_label == "generated":
                    r.font.highlight_color = WD_COLOR_INDEX.PINK
                elif s.ai_label == "revised":
                    r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                elif s.ai_label == "possible":
                    r.font.highlight_color = WD_COLOR_INDEX.GRAY_25
    if a.sim is not None and a.sim.sources:
        d.add_heading("Matching sources", level=2)
        for i, s in enumerate(a.sim.sources, 1):
            d.add_paragraph(f"{i}. {s.title} ({s.origin}) {s.url}")
    out = io.BytesIO(); d.save(out)
    return out.getvalue()
