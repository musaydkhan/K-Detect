"""Run with:  pytest -q   (from the project root). No network needed."""
import io, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docx import Document
from reportlab.pdfgen import canvas
from config import app_config as cfg
from core import similarity as sim
from core.extract import ExtractionError, extract_text
from core.pipeline import InputError, Options, run_analysis
from core.report import make_docx, make_pdf
from core.segment import segment

HUMAN = ("We collected samples from fifty wards over three weeks. Several swabs were lost in transit, which we only noticed later, "
         "and so the second round had to be repeated by hand. Honestly the weather did not help. ") * 6
AI = ("This study comprehensively examines the multifaceted role of probiotics in mitigating chemotherapy-induced oral mucositis. "
      "The findings underscore the importance of context in the design of effective supportive care strategies. "
      "Furthermore, this research contributes to broader discussions on the interplay between efficacy, safety, and patient outcomes. ") * 3


ORIG = {n: getattr(sim, n) for n in ("openalex", "europepmc", "arxiv", "brave")}


@pytest.fixture(scope="module")
def detector():
    from core.detector import Detector
    return Detector()


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Never touch the network in tests."""
    for name in ("openalex", "europepmc", "arxiv", "brave"):
        monkeypatch.setattr(sim, name, lambda ph: [])


def test_segmentation_keeps_offsets_and_abbreviations():
    text, paras, sents = segment("Smith et al. showed this in Fig. 2 of the paper. A second sentence follows here. Another one!")
    assert len(sents) == 3
    assert all(text[s.start:s.end] == s.text for s in sents)


def test_references_and_headings_are_not_scored():
    body = "Introduction\n\n" + " ".join(["This is a body sentence with enough words in it."] * 40) + "\n\nReferences\n\nSmith J. 2020. A title. Journal 1:1-2."
    _, _, sents = segment(body)
    kinds = {s.kind for s in sents}
    assert "heading" in kinds and "reference" in kinds
    assert not any(s.scored for s in sents if s.kind in ("heading", "reference"))


def test_docx_and_pdf_extraction():
    d = Document(); d.add_paragraph("Hello docx world. " * 5); buf = io.BytesIO(); d.save(buf)
    assert "Hello docx world" in extract_text("a.docx", buf.getvalue())
    pdf = io.BytesIO(); c = canvas.Canvas(pdf)
    for i in range(12):
        c.drawString(72, 800 - i * 20, "This sentence is real selectable text in a PDF file number %d." % i)
    c.save()
    assert "selectable text" in extract_text("a.pdf", pdf.getvalue())


def test_scanned_pdf_gives_clear_error():
    pdf = io.BytesIO(); c = canvas.Canvas(pdf); c.rect(10, 10, 100, 100); c.save()
    with pytest.raises(ExtractionError, match="scanned"):
        extract_text("scan.pdf", pdf.getvalue())


def test_bad_extension_and_size():
    with pytest.raises(ExtractionError):
        extract_text("a.txt", b"x")
    with pytest.raises(ExtractionError):
        extract_text("a.pdf", b"x" * (11 * 1024 * 1024))


def test_length_limits(detector):
    with pytest.raises(InputError, match="at least"):
        run_analysis("Too short.", Options(), detector)
    with pytest.raises(InputError, match="limit"):
        run_analysis("word " * (cfg.MAX_WORDS + 50), Options(), detector)


def test_detector_ranks_ai_style_above_human(detector):
    h = run_analysis(HUMAN, Options(check_similarity=False), detector).ai
    g = run_analysis(AI, Options(check_similarity=False), detector).ai
    assert g["ai_share"] > h["ai_share"]


def test_similarity_flags_copied_sentence_only():
    copied = "Probiotic supplementation has been widely investigated for the prevention of chemotherapy-induced oral mucositis in cancer patients."
    text = copied + " " + "Our own unrelated field notes describe the monsoon commute in Dhaka and the long queues at the market every single week. " * 6
    _, _, sents = segment(text)
    src = sim.Source("Review", "https://example.org/1", copied + " Results varied.", "Test")
    res = sim.check(sents, scholarly=[lambda ph: [src]])
    assert sents[0].sim_score >= 0.9 and sum(1 for s in sents if s.sim_score) == 1
    assert res.available and res.sources[0].matched_words > 0


def test_unreachable_services_are_not_reported_as_clean(detector, monkeypatch):
    import requests
    for name, fn in ORIG.items():                       # use the real providers...
        monkeypatch.setattr(sim, name, fn)
    def boom(*a, **k):                                    # ...but make every request fail
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(sim.requests, "get", boom)
    a = run_analysis(HUMAN, Options(check_ai=False), detector)
    assert a.sim.available is False and a.sim.similarity_pct == 0
    assert any("not a clean result" in n for n in a.notes)


def test_web_daily_cap_blocks_when_exhausted(monkeypatch):
    monkeypatch.setattr(cfg, "WEB_DAILY_CAP", 3)
    sim.WebBudget._day, sim.WebBudget._used = None, 0
    assert sim.WebBudget.take(3) and not sim.WebBudget.take(1)


def test_reports_build(detector):
    a = run_analysis(HUMAN + AI, Options(), detector)
    pdf, docx = make_pdf(a), make_docx(a)
    assert pdf[:4] == b"%PDF" and docx[:2] == b"PK"


def test_special_characters_do_not_break_pdf(detector):
    a = run_analysis(("The α-helix and β-sheet content was 5 µM ± 2 \u2265 3, <b>bold</b> & more. " * 20), Options(check_similarity=False), detector)
    assert make_pdf(a)[:4] == b"%PDF"


def test_arxiv_only_gets_a_few_queries(monkeypatch):
    calls = {"arxiv": 0, "other": 0}
    def fake_arxiv(ph): calls["arxiv"] += 1; return []
    def fake_other(ph): calls["other"] += 1; return []
    monkeypatch.setattr(sim, "arxiv", fake_arxiv)
    text = "\n\n".join(f"Sentence number {i} discusses unusual bacterial resistance mechanisms across hospital environments in considerable detail." for i in range(30))
    _, _, sents = segment(text)
    sim.check(sents, scholarly=[fake_other, sim.arxiv])
    assert calls["arxiv"] <= cfg.SIM_QUERIES_ARXIV and calls["other"] > calls["arxiv"]
