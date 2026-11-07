"""K-Detect | free AI-writing and similarity checker for students."""
import base64, csv, io, os
from pathlib import Path
import streamlit as st
from config import app_config as cfg
from core.extract import ExtractionError, extract_text
from core.pipeline import InputError, Options, run_analysis
from core.report import make_docx, make_pdf
from core import similarity
from ui import components as ui
from ui.styles import build_css

ROOT = Path(__file__).parent
st.set_page_config(page_title=cfg.HEADER_LINE, page_icon=str(ROOT / "assets" / "kdetect_icon.png"), layout="wide")

for key in ("BRAVE_API_KEY", "OPENALEX_API_KEY"):         # Streamlit secrets -> env (HF Spaces already uses env)
    try:
        if key in st.secrets and not os.environ.get(key):
            os.environ[key] = st.secrets[key]
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def get_detector():
    from core.detector import Detector
    return Detector()


@st.cache_resource(show_spinner=False)
def hero_b64():
    return base64.b64encode((ROOT / "assets" / "kdetect_hero.jpg").read_bytes()).decode()


def sentence_csv(a) -> bytes:
    out = io.StringIO(); w = csv.writer(out)
    w.writerow(["sentence_no", "ai_label", "ai_score", "similarity_overlap", "text"])
    for s in a.sentences:
        w.writerow([s.idx + 1, s.ai_label if s.scored else "skipped", round(s.p_ai, 3), s.sim_score, s.text])
    return out.getvalue().encode("utf-8")


# ---- top bar + theme -----------------------------------------------------------
left, right = st.columns([5, 1], vertical_alignment="center")
with right:
    light = st.toggle("Light mode", key="light_mode")
with left:
    st.markdown(ui.header(), unsafe_allow_html=True)
st.markdown(build_css("light" if light else "dark", hero_b64()), unsafe_allow_html=True)

analysis = st.session_state.get("analysis")
st.markdown(ui.hero(compact=analysis is not None), unsafe_allow_html=True)

# ---- input -----------------------------------------------------------------------
go = False
if analysis is None:
    tab_paste, tab_file = st.tabs(["Paste text", "Upload a file"])
    with tab_paste:
        pasted = st.text_area("Your text", height=230, label_visibility="collapsed",
                              placeholder=f"Paste your writing here ({cfg.MIN_WORDS} to {cfg.MAX_WORDS:,} words).")
    with tab_file:
        upload = st.file_uploader("DOCX or PDF", type=list(cfg.ALLOWED_EXTENSIONS), label_visibility="collapsed",
                                  help=f"Up to {cfg.MAX_FILE_MB} MB. Scanned PDFs without a text layer cannot be read.")
        st.caption(f"DOCX or PDF, up to {cfg.MAX_FILE_MB} MB, {cfg.MAX_WORDS:,} words at most. If a file is uploaded, it is used instead of pasted text.")

    with st.expander("Options"):
        c1, c2 = st.columns(2)
        check_ai = c1.checkbox("Look for AI-written text", True)
        check_sim = c1.checkbox("Compare with published sources", True)
        has_key = bool(os.environ.get("BRAVE_API_KEY"))
        use_web = c1.checkbox("Also search the web", False, disabled=not (has_key and check_sim),
                              help="Uses a limited daily web-search allowance shared by all users." if has_key else "Not enabled on this server.")
        if has_key:
            c1.caption(f"Web checks left today: about {similarity.WebBudget.remaining() // max(cfg.SIM_QUERIES_WEB, 1)}")
        skip_refs = c2.checkbox("Skip the reference list", True)
        skip_quotes = c2.checkbox("Skip quoted text", False)

    go = st.button("Check my writing", type="primary", disabled=not (check_ai or check_sim))

    if go:
        try:
            raw = extract_text(upload.name, upload.getvalue(), cfg.MAX_FILE_MB) if upload else pasted
            if not raw.strip():
                raise InputError("Paste some text or upload a DOCX or PDF file first.")
            bar = st.progress(0.02, text="Starting")
            detector = get_detector() if check_ai else None
            result = run_analysis(raw, Options(check_ai, check_sim, use_web and check_sim, skip_refs, skip_quotes), detector,
                                  progress=lambda f, m="Working": bar.progress(min(f, 1.0), text=m))
            result.downloads = {"pdf": make_pdf(result), "docx": make_docx(result), "csv": sentence_csv(result)}
            st.session_state["analysis"] = result
            bar.empty(); st.rerun()
        except (InputError, ExtractionError) as err:
            st.error(str(err))
        except Exception:
            st.error("Something went wrong while checking this text. Try again, or try a shorter section.")

# ---- results ---------------------------------------------------------------------
if analysis is not None:
    if st.button("Check another text"):
        st.session_state.pop("analysis"); st.rerun()
    st.markdown(ui.cards(analysis), unsafe_allow_html=True)
    for note in analysis.notes:
        st.markdown(f'<div class="kd-note">{ui.e(note)}</div>', unsafe_allow_html=True)
    t_doc, t_pass, t_src, t_prof, t_dl = st.tabs(["Marked text", "Flagged passages", "Sources", "Writing profile", "Download"])
    with t_doc:
        mode = st.radio("Show", ["All marks", "AI patterns only", "Similarity only"], horizontal=True, label_visibility="collapsed")
        st.markdown(ui.legend(analysis), unsafe_allow_html=True)
        st.markdown(ui.document(analysis, {"All marks": "all", "AI patterns only": "ai", "Similarity only": "sim"}[mode]), unsafe_allow_html=True)
        st.caption("Hover over a marked sentence to see its score.")
    with t_pass:
        st.markdown(ui.passages(analysis), unsafe_allow_html=True)
    with t_src:
        st.markdown(ui.sources(analysis), unsafe_allow_html=True)
    with t_prof:
        st.markdown(ui.profile(analysis), unsafe_allow_html=True)
    with t_dl:
        d = analysis.downloads
        st.markdown("Your report is built in your browser session and is not saved anywhere.")
        b1, b2, b3 = st.columns(3)
        b1.download_button("PDF report", d["pdf"], "k-detect-report.pdf", "application/pdf", type="primary", use_container_width=True)
        b2.download_button("Annotated Word file", d["docx"], "k-detect-annotated.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        b3.download_button("Scores as CSV", d["csv"], "k-detect-scores.csv", "text/csv", use_container_width=True)

# ---- about -----------------------------------------------------------------------
st.markdown("### About K-Detect")
with st.expander("How accurate is it? Read this before trusting a score"):
    st.markdown(ui.accuracy(), unsafe_allow_html=True)
with st.expander("How it works"):
    st.markdown(
        "**AI patterns.** Your text is cut into passages of about 80 words. A statistical model trained on scientific abstracts "
        "scores each passage, and scores are shared out to sentences. Only confident scores are highlighted.\n\n"
        "**Similarity.** K-Detect picks distinctive phrases from your text and searches OpenAlex, Europe PMC and arXiv "
        "(and the web, if switched on). It then compares your sentences with what those sources return. "
        "It cannot see paywalled papers, books, or other students' work.")
with st.expander("Privacy"):
    st.markdown("No accounts. Your text is processed in memory to produce the report and is not saved. "
                "When the similarity check runs, short phrases (about nine words each) from your text are sent to the search services named above.")
with st.expander("Responsible use"):
    st.markdown("Scores are likelihoods, not proof. Never use them as the only basis for an academic-misconduct decision. "
                "Non-native English writers are at higher risk of false flags.")
st.markdown(f'<div class="kd-foot">{cfg.HEADER_LINE}. A likelihood, not proof.</div>', unsafe_allow_html=True)
