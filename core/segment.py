"""Text normalisation and sentence segmentation that keeps character offsets."""
import re
from dataclasses import dataclass

ABBREVIATIONS = {
    "e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "vs", "etc", "dr", "mr", "mrs", "ms",
    "prof", "no", "vol", "pp", "p", "approx", "ca", "cf", "ref", "refs", "sp", "spp", "st", "inc",
    "ltd", "jr", "sr", "e.coli", "c", "var", "subsp", "resp", "viz", "sec", "ch",
}
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
REF_HEADING_RE = re.compile(r"^\s*(references|bibliography|works cited|literature cited|reference list)\s*:?\s*$", re.I)
END_RE = re.compile(r"[.!?][\"”’')\]]*\s+")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[ ]{2,}", " ", text)
    # un-wrap hard-wrapped lines (text copied from PDFs or web pages): a long line that does not end a sentence
    # continues on the next line
    text = re.sub(r"(?<=[^\n]{40})(?<![.!?:;])\n(?=[^\n])", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def words_in(text: str):
    return WORD_RE.findall(text)


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


@dataclass
class Sentence:
    idx: int
    para: int
    start: int
    end: int
    text: str
    words: int
    kind: str = "body"       # body | heading | reference | quote
    scored: bool = True      # included in AI scoring
    # filled later
    p_human: float = 0.0
    p_revised: float = 0.0
    p_generated: float = 0.0
    p_ai: float = 0.0
    ai_label: str = "none"   # none | human | possible | revised | generated
    sim_score: float = 0.0
    sim_sources: tuple = ()  # indices into the similarity source list


def _split_paragraph(par: str, offset: int):
    """Yield (start, end) spans of sentences inside a paragraph."""
    spans, last = [], 0
    for m in END_RE.finditer(par):
        before = par[last:m.start() + 1]
        tail = re.findall(r"[A-Za-z.]+", before)[-1:] or [""]
        token = tail[0].rstrip(".").lower()
        nxt = par[m.end():m.end() + 1]
        if token in ABBREVIATIONS or (nxt and nxt.islower()):
            continue
        spans.append((offset + last, offset + m.start() + len(m.group(0).rstrip())))
        last = m.end()
    if par[last:].strip():
        spans.append((offset + last, offset + len(par.rstrip())))
    return spans


def segment(text: str, exclude_references: bool = True, exclude_quotes: bool = False):
    """Return (clean_text, paragraphs[(start,end)], sentences[Sentence])."""
    text = normalize(text)
    sentences, paragraphs = [], []
    in_refs = False
    pos = 0
    for pi, par in enumerate(re.split(r"\n\s*\n|\n", text)):
        start = text.find(par, pos)
        if start < 0:
            continue
        pos = start + len(par)
        if not par.strip():
            continue
        paragraphs.append((start, start + len(par)))
        wc = count_words(par)
        is_heading = wc <= 8 and not re.search(r"[.!?]$", par.strip())
        if exclude_references and REF_HEADING_RE.match(par) and start > len(text) * 0.3:
            in_refs = True
        for s, e in _split_paragraph(par, start):
            seg = text[s:e]
            if not seg.strip():
                continue
            sent = Sentence(len(sentences), len(paragraphs) - 1, s, e, seg, count_words(seg))
            if in_refs:
                sent.kind, sent.scored = "reference", False
            elif is_heading:
                sent.kind, sent.scored = "heading", False
            elif exclude_quotes and re.match(r"^[\"“].*[\"”][.,;]?$", seg.strip()):
                sent.kind = "quote"
            sentences.append(sent)
    return text, paragraphs, sentences