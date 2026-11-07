"""Descriptive writing statistics (not evidence of authorship)."""
import re
from collections import Counter
from statistics import pstdev
from core.segment import words_in

STOP = set("""a an the and or but if while with without of at by for from to in on into onto is am are was were be been
being have has had do does did this that these those it its as not no nor so than then too very can could will would
should may might must shall i we you he she they them their our your my me him her us who whom which what when where
why how all any both each few more most other some such only own same also just about over under again further once
here there between during after before""".split())

TRANSITIONS = {
    "Addition": ["additionally", "furthermore", "moreover", "also", "in addition", "besides"],
    "Contrast": ["however", "nevertheless", "nonetheless", "although", "whereas", "conversely", "on the other hand"],
    "Cause and effect": ["therefore", "consequently", "thus", "hence", "as a result", "accordingly"],
    "Examples": ["for example", "for instance", "such as", "namely"],
    "Conclusion": ["in conclusion", "to conclude", "overall", "in summary", "in sum"],
}


def _syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r"e$", "", w)
    return max(1, len(re.findall(r"[aeiouy]+", w)))


def profile(text: str, sentences) -> dict:
    words = [w.lower() for w in words_in(text)]
    n = len(words)
    body = [s for s in sentences if s.kind == "body"] or list(sentences)
    lengths = [s.words for s in body]
    avg = sum(lengths) / len(lengths) if lengths else 0
    syl = sum(_syllables(w) for w in words)
    complex_w = sum(1 for w in words if _syllables(w) >= 3)
    flesch = 206.835 - 1.015 * avg - 84.6 * (syl / n) if n and avg else None
    fog = 0.4 * (avg + 100 * complex_w / n) if n and avg else None
    low = text.lower()
    trans = []
    for cat, phrases in TRANSITIONS.items():
        for ph in phrases:
            c = len(re.findall(r"\b" + re.escape(ph) + r"\b", low))
            if c:
                trans.append((cat, ph, c))
    trans.sort(key=lambda t: -t[2])
    content = [w for w in words if w not in STOP and len(w) > 2]
    bigrams = Counter(" ".join(words[i:i + 2]) for i in range(len(words) - 1))
    trigrams = Counter(" ".join(words[i:i + 3]) for i in range(len(words) - 2))
    return {
        "words": n, "sentences": len(body), "paragraphs": len({s.para for s in body}),
        "unique_words": len(set(words)),
        "lexical_diversity": round(len(set(words)) / n, 3) if n else 0,
        "avg_sentence_len": round(avg, 1),
        "sentence_len_sd": round(pstdev(lengths), 1) if len(lengths) > 1 else 0,
        "short": sum(1 for l in lengths if l <= 10), "medium": sum(1 for l in lengths if 10 < l <= 20),
        "long": sum(1 for l in lengths if l > 20),
        "flesch": None if flesch is None else round(flesch, 1), "fog": None if fog is None else round(fog, 1),
        "top_words": Counter(content).most_common(12),
        "repeated_phrases": [(p, c) for p, c in trigrams.most_common(8) if c > 1] or
                            [(p, c) for p, c in bigrams.most_common(8) if c > 2],
        "transitions": trans,
        "longest": sorted(((s.words, s.text) for s in body), reverse=True)[:3],
    }
