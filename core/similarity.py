"""Open-source similarity check: scholarly APIs + optional Brave web search.

The check finds overlap with sources it can *reach* (open abstracts, open-access
full text, web snippets). It cannot see paywalled papers or other students' work.
"""
from __future__ import annotations
import difflib, html, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from datetime import date
from typing import Callable, Iterable
import requests
from config import app_config as cfg

UA = {"User-Agent": "K-Detect/1.0 (free student writing checker)"}
TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*")
STOP = set("a an the and or but of at by for from to in on is are was were be been it its as that this with which than then".split())


@dataclass
class Source:
    title: str
    url: str
    text: str            # text available to compare against (abstract / full text / snippet)
    origin: str          # OpenAlex | Europe PMC | arXiv | Web
    matched_words: int = 0
    sentences: list = field(default_factory=list)


def tokens(text: str):
    return TOKEN_RE.findall(text.lower().replace("’", "'"))


# --------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------
def overlap(sent_tokens, src_tokens):
    """Return (coverage, longest_run) of sentence words found in the source in order."""
    if len(sent_tokens) < 6 or len(src_tokens) < 6:
        return 0.0, 0
    sm = difflib.SequenceMatcher(None, sent_tokens, src_tokens, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 4]
    covered = sum(b.size for b in blocks)
    longest = max((b.size for b in blocks), default=0)
    return covered / len(sent_tokens), longest


def match_sentences(sentences, sources: list[Source], skip_quotes: bool = True):
    """Assign each sentence its best source. Mutates Sentence.sim_* and Source.matched_words."""
    src_tok = [tokens(s.text) for s in sources]
    for sent in sentences:
        if not sent.scored or sent.words < 8 or (skip_quotes and sent.kind == "quote"):
            continue
        st = tokens(sent.text)
        best_i, best_cov, best_run = -1, 0.0, 0
        for i, tk in enumerate(src_tok):
            cov, run = overlap(st, tk)
            if cov > best_cov:
                best_i, best_cov, best_run = i, cov, run
        if best_i >= 0 and (best_cov >= cfg.SIM_MATCH_COVERAGE or best_run >= cfg.SIM_MATCH_RUN):
            sent.sim_score = round(best_cov, 2)
            sent.sim_sources = (best_i,)
            sources[best_i].matched_words += sent.words
            sources[best_i].sentences.append(sent.idx)


# --------------------------------------------------------------------------
# choosing what to search for
# --------------------------------------------------------------------------
def _distinct(tk):
    content = [t for t in tk if t not in STOP]
    return len({t for t in content if len(t) >= 6}) / max(len(tk), 1)


def pick_queries(sentences, n: int, phrase_words: int = 9):
    """Pick up to n distinctive phrases, spread across the document."""
    cands = []
    for s in sentences:
        if s.scored and s.kind != "quote" and s.words >= 8:
            tk = s.text.split()
            # most distinctive window of `phrase_words` consecutive words
            best, best_score = None, -1
            for i in range(0, max(1, len(tk) - phrase_words + 1)):
                w = tk[i:i + phrase_words]
                score = _distinct([t.lower() for t in w])
                if score > best_score:
                    best, best_score = w, score
            phrase = re.sub(r'["“”]', "", " ".join(best)).strip(" ,.;:()[]")
            cands.append((_distinct(tokens(s.text)), s.para, s.idx, phrase))
    cands.sort(reverse=True)
    chosen, per_para = [], {}
    for score, para, idx, phrase in cands:           # at most 2 per paragraph first
        if per_para.get(para, 0) < 2 and len(chosen) < n:
            chosen.append((idx, phrase)); per_para[para] = per_para.get(para, 0) + 1
    seen = {c[0] for c in chosen}
    for score, para, idx, phrase in cands:           # not enough paragraphs: fill up from the best remaining
        if len(chosen) >= n:
            break
        if idx not in seen:
            chosen.append((idx, phrase)); seen.add(idx)
    return chosen


# --------------------------------------------------------------------------
# providers: each takes a phrase and returns list[Source]
# --------------------------------------------------------------------------
_FAILS: dict = {}
_OK: dict = {}      # services that answered at least once in this check
_fail_lock = threading.Lock()


def _get(url, name="", **kw):
    """GET with a per-service circuit breaker: after repeated failures the service is skipped."""
    with _fail_lock:
        if _FAILS.get(name, 0) >= cfg.SIM_MAX_FAILS:
            return None
    headers = {**UA, **kw.pop("headers", {})}
    for attempt in (0, 1):
        try:
            r = requests.get(url, headers=headers, timeout=cfg.HTTP_TIMEOUT, **kw)
        except requests.RequestException:
            break
        if r.status_code == 429 and attempt == 0:
            time.sleep(1.2); continue
        with _fail_lock:
            _FAILS[name] = 0 if r.status_code < 500 and r.status_code != 429 else _FAILS.get(name, 0) + 1
            if r.status_code < 500 and r.status_code != 429:
                _OK[name] = _OK.get(name, 0) + 1
        return r
    with _fail_lock:
        _FAILS[name] = _FAILS.get(name, 0) + 1
    return None


def unreachable_services():
    with _fail_lock:
        return sorted(k for k, v in _FAILS.items() if v > 0 and not _OK.get(k))


def reset_failures():
    with _fail_lock:
        _FAILS.clear()
        _OK.clear()


def openalex(phrase: str):
    key = os.environ.get("OPENALEX_API_KEY", "")
    params = {"search": phrase, "per-page": 5, "select": "title,doi,id,abstract_inverted_index"}
    if key:
        params["api_key"] = key
    r = _get("https://api.openalex.org/works", "OpenAlex", params=params)
    if not r or r.status_code != 200:
        return []
    out = []
    for w in r.json().get("results", []):
        inv = w.get("abstract_inverted_index") or {}
        if not inv:
            continue
        words = sorted((pos, word) for word, ps in inv.items() for pos in ps)
        text = " ".join(word for _, word in words)
        url = w.get("doi") or w.get("id") or ""
        out.append(Source(w.get("title") or "Untitled", url, text, "OpenAlex"))
    return out


def europepmc(phrase: str):
    r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", "Europe PMC",
             params={"query": f'"{phrase}"', "format": "json", "resultType": "core", "pageSize": 5})
    if not r or r.status_code != 200:
        return []
    out = []
    for x in r.json().get("resultList", {}).get("result", []):
        text = html.unescape(re.sub(r"<[^>]+>", " ", x.get("abstractText") or ""))
        if not text.strip():
            continue
        pmid, pmcid = x.get("pmid"), x.get("pmcid")
        url = f"https://europepmc.org/article/{x.get('source', 'MED')}/{pmid or pmcid or x.get('id', '')}"
        out.append(Source(x.get("title") or "Untitled", url, text, "Europe PMC"))
    return out


_arxiv_lock = threading.Lock()
_arxiv_last = [0.0]


def arxiv(phrase: str):
    # arXiv's API terms: one request at a time, at most one every 3 seconds.
    with _arxiv_lock:
        wait = 3.0 - (time.time() - _arxiv_last[0])
        if wait > 0:
            time.sleep(wait)
        _arxiv_last[0] = time.time()
        r = _get("http://export.arxiv.org/api/query", "arXiv", params={"search_query": f'all:"{phrase}"', "max_results": 3})
    if not r or r.status_code != 200:
        return []
    out = []
    for entry in re.findall(r"<entry>(.*?)</entry>", r.text, re.S):
        g = lambda tag: html.unescape(re.sub(r"\s+", " ", (re.search(rf"<{tag}>(.*?)</{tag}>", entry, re.S) or [None, ""])[1])).strip()
        out.append(Source(g("title") or "Untitled", g("id"), g("summary"), "arXiv"))
    return out


class WebBudget:
    """Daily cap for web searches so the free monthly credit is never exceeded."""
    _lock = threading.Lock()
    _day, _used = None, 0

    @classmethod
    def take(cls, n: int) -> bool:
        with cls._lock:
            today = date.today()
            if cls._day != today:
                cls._day, cls._used = today, 0
            if cls._used + n > cfg.WEB_DAILY_CAP:
                return False
            cls._used += n
            return True

    @classmethod
    def remaining(cls) -> int:
        with cls._lock:
            return cfg.WEB_DAILY_CAP if cls._day != date.today() else max(0, cfg.WEB_DAILY_CAP - cls._used)


def brave(phrase: str):
    key = os.environ.get("BRAVE_API_KEY", "")
    if not key:
        return []
    r = _get("https://api.search.brave.com/res/v1/web/search", "Web", params={"q": f'"{phrase}"', "count": 5},
             headers={"X-Subscription-Token": key, "Accept": "application/json"})
    if not r or r.status_code != 200:
        return []
    out = []
    for x in r.json().get("web", {}).get("results", []):
        text = " ".join([x.get("description", "")] + list(x.get("extra_snippets") or []))
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        if text.strip():
            out.append(Source(x.get("title") or x.get("url", ""), x.get("url", ""), text, "Web"))
    return out


# --------------------------------------------------------------------------
# orchestration
# --------------------------------------------------------------------------
@dataclass
class SimilarityResult:
    sources: list
    similarity_pct: float
    scored_words: int
    queries_run: int
    notes: list = field(default_factory=list)
    available: bool = True     # False when no search service could be reached


def check(sentences, use_web: bool = False, scholarly: Iterable[Callable] | None = None,
          web: Callable | None = None, skip_quotes: bool = True, progress=None) -> SimilarityResult:
    scholarly = list(scholarly) if scholarly is not None else [openalex, europepmc, arxiv]
    web = web or brave
    notes, jobs = [], []
    for n_phrase, (idx, phrase) in enumerate(pick_queries(sentences, cfg.SIM_QUERIES_SCHOLARLY)):
        jobs += [(fn, phrase) for fn in scholarly if not (fn is arxiv and n_phrase >= cfg.SIM_QUERIES_ARXIV)]
    if use_web:
        wq = pick_queries(sentences, cfg.SIM_QUERIES_WEB)
        if not os.environ.get("BRAVE_API_KEY") and web is brave:
            notes.append("Web check skipped: no search key is configured on this server.")
        elif WebBudget.take(len(wq)) or web is not brave:
            jobs += [(web, phrase) for _, phrase in wq]
        else:
            notes.append("Web check skipped: today's free web-search allowance is used up. Scholarly sources were still checked.")
    found, done = [], 0
    reset_failures()
    pool = ThreadPoolExecutor(max_workers=4)
    futures = [pool.submit(fn, phrase) for fn, phrase in jobs]
    try:
        for f in as_completed(futures, timeout=cfg.SIM_TIME_BUDGET):
            try:
                found += f.result()
            except Exception:
                pass
            done += 1
            if progress:
                progress(done / max(len(futures), 1))
    except FutTimeout:
        notes.append("The similarity check reached its time limit, so some searches were skipped. Results may be incomplete.")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    down = unreachable_services()
    available = True
    if down and not found:
        available = False
        notes.append("No search service could be reached from this server, so there is no similarity result. "
                     "This is not a clean result. Try again in a few minutes.")
    elif down:
        notes.append("Could not reach: " + ", ".join(down) + ". Results from the other sources are shown.")
    seen, sources = set(), []
    for s in found:                                  # de-duplicate by URL/title
        k = (s.url or s.title).lower()
        if k not in seen:
            seen.add(k); sources.append(s)
    match_sentences(sentences, sources, skip_quotes)
    sources = sorted([s for s in sources if s.matched_words], key=lambda s: -s.matched_words)
    for sent in sentences:                           # re-index sim_sources after filtering/sorting
        sent.sim_sources = tuple(i for i, s in enumerate(sources) if sent.idx in s.sentences) if sent.sim_score else ()
    total = sum(s.words for s in sentences if s.scored) or 1
    flagged = sum(s.words for s in sentences if s.sim_score)
    if not jobs:
        notes.append("No searches could be run: the text has no sentence long enough to search for. "
                     "If you pasted from a PDF or web page, check that sentences are not split across many short lines.")
    return SimilarityResult(sources, round(100 * flagged / total, 1), total, len(jobs), notes, available)