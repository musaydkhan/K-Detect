"""One entry point that runs every check and returns a single Analysis object."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from config import app_config as cfg
from core import detector as det_mod, linguistics, similarity
from core.segment import segment, count_words


@dataclass
class Options:
    check_ai: bool = True
    check_similarity: bool = True
    use_web: bool = False
    exclude_references: bool = True
    exclude_quotes: bool = False


@dataclass
class Analysis:
    text: str
    paragraphs: list
    sentences: list
    options: Options
    word_count: int
    ai: dict | None = None
    sim: similarity.SimilarityResult | None = None
    profile: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.now().strftime("%d %b %Y, %H:%M"))


class InputError(ValueError):
    pass


def run_analysis(raw_text: str, options: Options, detector=None, progress=None) -> Analysis:
    text, paragraphs, sentences = segment(raw_text, options.exclude_references, options.exclude_quotes)
    wc = count_words(text)
    if wc < cfg.MIN_WORDS:
        raise InputError(f"Add at least {cfg.MIN_WORDS} words. Your text has {wc}; shorter text gives unreliable results.")
    if wc > cfg.MAX_WORDS:
        raise InputError(f"This text has {wc:,} words. The limit is {cfg.MAX_WORDS:,} words per check; "
                         "split it into sections and check them one at a time.")
    a = Analysis(text, paragraphs, sentences, options, wc)
    if options.check_ai:
        if progress: progress(0.15, "Reading your writing for AI patterns")
        windows = detector.score(sentences)
        a.ai = det_mod.summarize(sentences)
        if not windows:
            a.notes.append("Too little scorable text for an AI check (headings and references are skipped).")
    if options.check_similarity:
        def sub(frac): progress and progress(0.35 + 0.55 * frac, "Comparing with open sources")
        if progress: progress(0.35, "Comparing with open sources")
        a.sim = similarity.check(sentences, use_web=options.use_web, skip_quotes=options.exclude_quotes, progress=sub)
        a.notes += a.sim.notes
    if progress: progress(0.95, "Building your report")
    a.profile = linguistics.profile(text, sentences)
    return a


def paragraph_runs(a: Analysis):
    """Yield, per paragraph, a list of (text, Sentence|None) so renderers can highlight."""
    by_para = {}
    for s in a.sentences:
        by_para.setdefault(s.para, []).append(s)
    for pi, (ps, pe) in enumerate(a.paragraphs):
        runs, cur = [], ps
        for s in by_para.get(pi, []):
            if s.start > cur:
                runs.append((a.text[cur:s.start], None))
            runs.append((a.text[s.start:s.end], s)); cur = s.end
        if cur < pe:
            runs.append((a.text[cur:pe], None))
        yield runs
