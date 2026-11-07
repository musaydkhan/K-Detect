"""HTML builders for the interface. All user text is escaped."""
import html, json
from pathlib import Path
from config import app_config as cfg
from core.pipeline import paragraph_runs

ASSETS = Path(__file__).resolve().parents[1] / "assets"
e = html.escape
AI_NAMES = {"generated": "Likely AI-written", "revised": "Likely AI-edited", "possible": "Possible AI patterns"}
CLS = {"generated": "gen", "revised": "rev", "possible": "poss"}


def header() -> str:
    svg = (ASSETS / "kdetect_logo.svg").read_text()
    return (f'<div class="kd-bar">{svg}<div><span class="kd-name">{cfg.BRAND}</span>'
            f'<span class="kd-tag"> <span class="sep">|</span> {cfg.TAGLINE}</span></div></div>')


def hero(compact: bool) -> str:
    cls = "kd-hero compact" if compact else "kd-hero"
    return (f'<div class="{cls}"><h1>Check your writing before you submit it</h1>'
            '<p>Find passages that read like AI, and text that overlaps with published work. '
            'Every flag comes with a plain explanation of what to do next.</p>'
            '<div class="kd-chips"><span class="kd-chip">Free for every student</span>'
            '<span class="kd-chip">No account</span><span class="kd-chip">Nothing is stored</span>'
            '<span class="kd-chip">English only</span></div></div>')


def _ring(pct, color, text, small=False):
    return (f'<div class="kd-ring{" small" if small else ""}" style="--p:{pct};--c:{color}"><b>{text}</b></div>')


def cards(a) -> str:
    out = []
    if a.ai is not None:
        v = a.ai
        if not v["scored_words"]:
            body, ring = "Not enough scorable text for an AI check.", _ring(0, "#E8604C", "n/a", True)
        elif v["low_signal"]:
            body = ("Low signal. Under 20% of your text shows AI-like patterns, and small amounts like this are often "
                    "false alarms. Review the highlighted passages only if you used AI help.")
            ring = _ring(v["ai_share"], "#E8604C", "Low<br>signal", True)
        elif v["ai_share"] == 0:
            body, ring = "No passage crossed the flag threshold.", _ring(0, "#E8604C", "0%")
        else:
            body = (f"About {v['ai_share']:.0f}% of the text shows patterns typical of AI-written or AI-edited writing. "
                    "This is a likelihood, not proof.")
            ring = _ring(v["ai_share"], "#E8604C", f"{v['ai_share']:.0f}%")
        split = (f'<div class="kd-split"><span><i style="background:#E8604C"></i>AI-written {v["generated"]:.0f}%</span>'
                 f'<span><i style="background:#F2B33D"></i>AI-edited {v["revised"]:.0f}%</span>'
                 f'<span><i style="background:#8FA6BC"></i>Possible {v["possible"]:.0f}%</span></div>') if v["scored_words"] else ""
        out.append(f'<div class="kd-card"><h4>AI-writing likelihood</h4><div class="row">{ring}<p>{body}</p></div>{split}</div>')
    if a.sim is not None:
        s = a.sim
        n = len(s.sources)
        if not s.available:
            out.append('<div class="kd-card"><h4>Similarity to open sources</h4><div class="row">'
                       f'{_ring(0, "#8A74F0", "n/a", True)}<p>Could not reach the search services, so there is no result. '
                       'This is not a clean result.</p></div></div>')
            s = None
    if a.sim is not None and s is not None:
        body = (f"{n} matching source{'s' if n != 1 else ''} found among open scholarly sources"
                f"{' and the web' if a.options.use_web else ''}. " if n else
                "No overlap found in the open sources searched. ") + "Paywalled papers are not searched."
        out.append(f'<div class="kd-card"><h4>Similarity to open sources</h4><div class="row">'
                   f'{_ring(s.similarity_pct, "#8A74F0", f"{s.similarity_pct:.0f}%")}<p>{body}</p></div></div>')
    skipped = sum(x.words for x in a.sentences if not x.scored)
    out.append(f'<div class="kd-card"><h4>What was checked</h4><div class="row">{_ring(100, "#2CC8F0", f"{a.word_count:,}<br>words", True)}'
               f'<p>{len(a.sentences)} sentences in {len(a.paragraphs)} paragraphs.'
               f'{f" {skipped} words (headings, references) were skipped." if skipped else ""}</p></div></div>')
    return f'<div class="kd-cards">{"".join(out)}</div>'


def legend(a) -> str:
    items = []
    if a.ai is not None:
        items += [f'<span><span class="sw" style="background:var(--gen)">Likely AI-written</span></span>',
                  f'<span><span class="sw" style="background:var(--rev)">Likely AI-edited</span></span>',
                  f'<span><span class="sw" style="background:var(--poss)">Possible AI patterns</span></span>']
    if a.sim is not None:
        items.append('<span><span class="sw" style="text-decoration:underline;text-decoration-color:var(--sim);'
                     'text-decoration-thickness:2px;text-underline-offset:4px">Similar to a source</span></span>')
    return f'<div class="kd-legend">{"".join(items)}</div>'


def _tip(s):
    bits = []
    if s.ai_label in AI_NAMES:
        bits.append(f"{AI_NAMES[s.ai_label]} (score {s.p_ai:.2f})")
    if s.sim_score:
        bits.append(f"{round(s.sim_score * 100)}% overlap with a source")
    return " | ".join(bits)


def document(a, mode="all") -> str:
    paras = []
    for runs in paragraph_runs(a):
        parts = []
        for text, s in runs:
            t = e(text)
            if s is None:
                parts.append(t); continue
            cls = []
            if mode in ("all", "ai") and s.ai_label in CLS:
                cls.append(CLS[s.ai_label])
            if mode in ("all", "sim") and s.sim_score:
                cls.append("sim")
            if not s.scored:
                cls.append("skip")
            parts.append(f'<span class="s {" ".join(cls)}" title="{e(_tip(s))}">{t}</span>' if cls else t)
        paras.append(f"<p>{''.join(parts).replace(chr(10), '<br>')}</p>")
    return f'<div class="kd-doc">{"".join(paras)}</div>'


ADVICE_AI = ("If you wrote this yourself, keep your notes and drafts as evidence of your process. If you used AI help, "
             "follow your institution's rules on disclosure, and rewrite it in your own words.")
ADVICE_SIM = ("If this is not a quotation, rewrite it in your own words and cite the source. "
              "If it is a quotation, add quotation marks and a citation.")


def passages(a, limit=40) -> str:
    items = []
    for s in a.sentences:
        ai = s.ai_label in ("generated", "revised")
        if not (ai or s.sim_score):
            continue
        badges, why, cls = "", [], "sim"
        if ai:
            cls = CLS[s.ai_label]
            badges += f'<span class="kd-badge {cls}">{AI_NAMES[s.ai_label]}</span>'
            why.append(f"The wording and sentence patterns resemble AI-generated text (score {s.p_ai:.2f} out of 1). {ADVICE_AI}")
        if s.sim_score:
            badges += '<span class="kd-badge sim">Similar to a source</span>'
            src = a.sim.sources[s.sim_sources[0]] if s.sim_sources else None
            if src:
                link = f'<a href="{e(src.url)}" target="_blank" rel="noopener noreferrer">{e(src.title[:90])}</a>' if src.url.startswith("http") else e(src.title[:90])
                why.append(f"About {round(s.sim_score * 100)}% of this sentence's words appear in {link} ({e(src.origin)}). {ADVICE_SIM}")
        items.append(f'<div class="kd-item {cls}"><div class="meta">{badges}Sentence {s.idx + 1}</div>'
                     f'<div class="txt">{e(s.text)}</div><div class="why">{"<br>".join(why)}</div></div>')
    if not items:
        return '<div class="kd-note">Nothing was flagged. That is a good result, but it is not a guarantee of anything.</div>'
    more = f'<div class="kd-note">Showing the first {limit} of {len(items)} flagged passages. The PDF report lists the strongest ones.</div>' if len(items) > limit else ""
    return "".join(items[:limit]) + more


def sources(a) -> str:
    if a.sim is None:
        return '<div class="kd-note">The similarity check was switched off for this run.</div>'
    if not a.sim.sources:
        return ('<div class="kd-note">No matching passages were found in the sources searched. This tool cannot see paywalled '
                'journals, books, or other students\' work, so an empty list does not prove the text is original.</div>')
    top = max(s.matched_words for s in a.sim.sources) or 1
    rows = ""
    for i, s in enumerate(a.sim.sources, 1):
        link = f'<a href="{e(s.url)}" target="_blank" rel="noopener noreferrer">{e(s.title[:110])}</a>' if s.url.startswith("http") else e(s.title[:110])
        rows += (f'<tr><td>{i}</td><td>{link}<br><small>{e(s.url[:90])}</small></td><td>{e(s.origin)}</td>'
                 f'<td>{s.matched_words}<div class="kd-bar2"><i style="width:{100 * s.matched_words // top}%"></i></div></td></tr>')
    return f'<table class="kd-table"><tr><th>#</th><th>Source</th><th>Found in</th><th>Matched words</th></tr>{rows}</table>'


def profile(a) -> str:
    p = a.profile
    kv = [("Words", f"{p['words']:,}"), ("Sentences", p["sentences"]), ("Lexical diversity", p["lexical_diversity"]),
          ("Average sentence length", f"{p['avg_sentence_len']} words"), ("Sentence length spread (SD)", p["sentence_len_sd"]),
          ("Short / medium / long sentences", f"{p['short']} / {p['medium']} / {p['long']}"),
          ("Flesch reading ease", p["flesch"]), ("Gunning fog index", p["fog"])]
    t1 = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in kv)
    top = ", ".join(f"{e(w)} ({c})" for w, c in p["top_words"]) or "none"
    rep = ", ".join(f"&ldquo;{e(w)}&rdquo; &times;{c}" for w, c in p["repeated_phrases"]) or "none found"
    tr = ", ".join(f"{e(ph)} &times;{c}" for _, ph, c in p["transitions"][:8]) or "none found"
    return (f'<table class="kd-table"><tr><th>Measure</th><th>Value</th></tr>{t1}</table>'
            f'<div class="kd-note"><b>Most used words:</b> {top}<br><b>Repeated phrases:</b> {rep}<br><b>Transition words:</b> {tr}</div>'
            '<div class="kd-note">These measures describe the text. They do not show who or what wrote it.</div>')


def accuracy() -> str:
    data = json.loads((ASSETS / "raid_per_generator.json").read_text())
    rows = "".join(f"<tr><td>{e(r['generator'])}</td><td>{round(r['recall'] * 100)}%</td></tr>" for r in data["rows"])
    return (
        "<p>K-Detect was trained on <b>SciHRA-Detect</b>, a dataset of scientific abstracts that are human-written, AI-edited, "
        "or AI-generated. On held-out abstracts from that same dataset it separates human from AI text very well "
        "(AUC 0.99 on roughly 80-word passages, flagging about 2% of human passages at the threshold used here).</p>"
        "<p><b>That number is the best case.</b> We also tested it on text from AI systems it never saw in training "
        "(a subset of the RAID benchmark). It caught some generators well and others barely at all:</p>"
        f'<table class="kd-table"><tr><th>AI system in the test</th><th>AI text detected</th></tr>{rows}</table>'
        f"<p><small>Measured on whole abstracts at the model's default threshold, with about {round(data['false_positive_rate'] * 100)}% of "
        "human abstracts wrongly flagged. The app's stricter passage threshold flags fewer human passages and also catches somewhat less AI text. "
        "Newer systems such as Claude and Gemini were not tested.</small></p>"
        "<p>The practical meaning: a flag is a reason to look closer, and a clean result does not prove the text is human. "
        "Non-native English writing and heavily edited text can be scored wrongly in either direction.</p>")
