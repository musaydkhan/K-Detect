"""Theme CSS. Every colour is a variable, so light and dark are complete, not partial."""

TOKENS = {
    "dark": dict(bg="#08131F", surface="#0F2236", surface2="#15304A", text="#E8F1FA", muted="#9FB5CA", line="#274560",
                 accent="#2CC8F0", accent_ink="#04283A", hero1="rgba(6,16,28,.55)", hero2="rgba(6,16,28,.92)",
                 gen="rgba(240,106,86,.42)", rev="rgba(244,185,66,.38)", poss="rgba(159,181,202,.20)", sim="#B3A3FF",
                 ring_bg="#1B3752", shadow="0 1px 0 rgba(255,255,255,.04) inset", scheme="dark"),
    "light": dict(bg="#F3F6FA", surface="#FFFFFF", surface2="#E9EFF6", text="#10243A", muted="#52667C", line="#D0DBE7",
                  accent="#0A7BA6", accent_ink="#FFFFFF", hero1="rgba(8,24,44,.50)", hero2="rgba(8,24,44,.86)",
                  gen="#F8C6BC", rev="#FBE0A0", poss="#E2EAF2", sim="#6A4CD4",
                  ring_bg="#DDE6EF", shadow="0 1px 2px rgba(16,36,58,.06)", scheme="light"),
}

BASE = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');
:root{--bg:%(bg)s;--surface:%(surface)s;--surface2:%(surface2)s;--text:%(text)s;--muted:%(muted)s;--line:%(line)s;
--accent:%(accent)s;--accent-ink:%(accent_ink)s;--gen:%(gen)s;--rev:%(rev)s;--poss:%(poss)s;--sim:%(sim)s;--ring-bg:%(ring_bg)s;
color-scheme:%(scheme)s}
html,body,.stApp,[data-testid="stAppViewContainer"]{background:var(--bg)!important;color:var(--text)!important;
font-family:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif!important}
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],footer,#MainMenu{display:none!important}
.block-container{max-width:1060px!important;padding:1.1rem 1.2rem 4rem!important}
.stApp p,.stApp li,.stApp label,.stApp span,.stApp div[data-testid="stMarkdownContainer"],.stApp h1,.stApp h2,.stApp h3{color:var(--text)}
.stApp a{color:var(--accent)}
:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}

/* radio / checkbox / switch: draw them from the tokens so both themes match */
[data-testid="stCheckbox"] label:not(:has(input[role="switch"]))>div:first-of-type{background:var(--surface)!important;border:2px solid var(--muted)!important;box-shadow:none!important}
[data-testid="stCheckbox"] label:has(input:checked):not(:has(input[role="switch"]))>div:first-of-type{background:var(--accent)!important;border-color:var(--accent)!important}
label:has(input[role="switch"])>div:first-of-type{background:var(--line)!important}
label:has(input[role="switch"]:checked)>div:first-of-type{background:var(--accent)!important}
[data-testid="stRadioOption"]>div>div:first-child{background:var(--surface)!important;border:2px solid var(--muted)!important;box-shadow:none!important}
[data-testid="stRadioOption"][data-selected="true"]>div>div:first-child{border-color:var(--accent)!important}
[data-testid="stRadioOption"][data-selected="true"]>div>div:first-child>div{background:var(--accent)!important}

/* top bar */
.kd-bar{display:flex;align-items:center;gap:.8rem;padding:.2rem 0 .9rem}
.kd-bar img,.kd-bar svg{width:38px;height:38px;flex:none}
.kd-name{font-weight:600;font-size:1.15rem;letter-spacing:-.01em}
.kd-tag{color:var(--muted);font-size:.92rem}
.kd-bar .sep{color:var(--line);margin:0 .1rem}

/* hero */
.kd-hero{position:relative;border-radius:18px;overflow:hidden;padding:2.6rem 2.2rem 2.4rem;margin-bottom:1.4rem;
background:linear-gradient(100deg,%(hero1)s,%(hero2)s),var(--hero-img) center/cover no-repeat;border:1px solid var(--line)}
.stApp .kd-hero h1,.kd-hero h1{font-family:'Source Serif 4',Georgia,serif;font-weight:600;font-size:2.25rem;line-height:1.15;letter-spacing:-.015em;
margin:0 0 .6rem;max-width:20ch;color:#F2F8FF!important}
.stApp .kd-hero h1,.stApp .kd-hero h1 *{color:#F2F8FF!important}
.stApp .kd-hero p,.stApp .kd-hero p *{color:#C5D6E6!important}
.stApp .kd-hero .kd-chip{color:#E4EFFA!important}
.stApp .kd-badge{color:#111!important}
.kd-hero p{margin:0;max-width:52ch;color:#C5D6E6!important;font-size:1.02rem;line-height:1.55}
.kd-hero.compact{padding:1.1rem 1.4rem}.kd-hero.compact h1{font-size:1.25rem;margin:0}.kd-hero.compact p{display:none}
.kd-chips{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.1rem}
.kd-chip{border:1px solid rgba(255,255,255,.28);color:#E4EFFA!important;border-radius:999px;padding:.22rem .7rem;font-size:.82rem}
.kd-hero.compact .kd-chips{display:none}

/* inputs */
[data-testid="stTextArea"] textarea{background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--line)!important;
border-radius:12px!important;font-family:'Source Serif 4',Georgia,serif!important;font-size:1.02rem!important;line-height:1.6!important}
[data-testid="stTextArea"] textarea::placeholder{color:var(--muted)!important;opacity:.85}
[data-testid="stFileUploaderDropzone"]{background:var(--surface)!important;border:1.5px dashed var(--line)!important;border-radius:12px!important}
[data-testid="stFileUploaderDropzone"] *{color:var(--muted)!important}
[data-testid="stFileUploaderDropzone"] button{background:var(--surface2)!important;color:var(--text)!important;border:1px solid var(--line)!important}
[data-testid="stFileUploader"] small{color:var(--muted)!important}
[data-testid="stExpander"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary *{color:var(--text)!important}
[data-testid="stCheckbox"] label,[data-testid="stToggle"] label,[data-testid="stRadio"] label{color:var(--text)!important}
[data-testid="stCaptionContainer"],.stApp small{color:var(--muted)!important}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:10px!important;font-weight:600!important;border:1px solid var(--line)!important;
background:var(--surface)!important;color:var(--text)!important;padding:.55rem 1.1rem!important}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{background:var(--accent)!important;color:var(--accent-ink)!important;border-color:var(--accent)!important}
.stButton>button:hover,.stDownloadButton>button:hover{filter:brightness(1.07)}

/* tabs */
[data-baseweb="tab-list"]{gap:.25rem;border-bottom:1px solid var(--line)!important;background:transparent!important}
button[data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;font-weight:500!important;padding:.55rem .9rem!important}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--text)!important}
[data-baseweb="tab-highlight"]{background:var(--accent)!important}
[data-baseweb="tab-border"]{background:transparent!important}
[data-testid="stAlert"]{background:var(--surface2)!important;border:1px solid var(--line)!important;border-radius:12px!important}
[data-testid="stAlert"] *{color:var(--text)!important}
[data-testid="stProgress"] div[role="progressbar"]>div{background:var(--accent)!important}
[data-testid="stStatusWidget"],[data-testid="stStatus"]{background:var(--surface)!important;border:1px solid var(--line)!important;border-radius:12px!important}

/* summary cards */
.kd-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:.9rem;margin:1rem 0 1.2rem}
.kd-card{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1rem 1.1rem;box-shadow:%(shadow)s}
.kd-card h4{margin:0 0 .6rem;font-size:.95rem;font-weight:600;color:var(--text)}
.kd-card .row{display:flex;gap:1rem;align-items:center}
.kd-ring{--p:0;--c:var(--accent);width:84px;height:84px;border-radius:50%%;flex:none;display:grid;place-items:center;
background:conic-gradient(var(--c) calc(var(--p)*1%%),var(--ring-bg) 0)}
.kd-ring b{width:62px;height:62px;border-radius:50%%;background:var(--surface);display:grid;place-items:center;font-size:1.18rem;font-weight:600}
.kd-ring.small b{font-size:.78rem;text-align:center;line-height:1.1}
.kd-card p{margin:0;color:var(--muted);font-size:.9rem;line-height:1.45}
.kd-split{display:flex;gap:.9rem;margin-top:.6rem;font-size:.82rem;color:var(--muted);flex-wrap:wrap}
.kd-split i{display:inline-block;width:.7rem;height:.7rem;border-radius:3px;margin-right:.3rem;vertical-align:-1px}

/* marked-up document */
.kd-legend{display:flex;flex-wrap:wrap;gap:.9rem;font-size:.85rem;color:var(--muted);margin:.4rem 0 .9rem}
.kd-legend span.sw{display:inline-block;padding:0 .55rem;border-radius:4px;color:var(--text)}
.kd-doc{background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:1.6rem 1.8rem;
font-family:'Source Serif 4',Georgia,serif;font-size:1.04rem;line-height:1.78;max-width:860px}
.kd-doc p{margin:0 0 1.05rem;color:var(--text)!important}
.kd-doc .s.gen{background:var(--gen)}.kd-doc .s.rev{background:var(--rev)}.kd-doc .s.poss{background:var(--poss)}
.kd-doc .s.sim{text-decoration:underline;text-decoration-color:var(--sim);text-decoration-thickness:2px;text-underline-offset:4px}
.kd-doc .s{border-radius:3px;box-decoration-break:clone;-webkit-box-decoration-break:clone;padding:.05em 0}
.kd-doc .skip{opacity:.62}

/* passages + sources */
.kd-item{background:var(--surface);border:1px solid var(--line);border-left:4px solid var(--line);border-radius:10px;padding:.8rem 1rem;margin:.6rem 0}
.kd-item.gen{border-left-color:#E8604C}.kd-item.rev{border-left-color:#F2B33D}.kd-item.sim{border-left-color:var(--sim)}
.kd-item .meta{font-size:.82rem;color:var(--muted);margin-bottom:.35rem}
.kd-item .txt{font-family:'Source Serif 4',Georgia,serif;font-size:.98rem;line-height:1.6;color:var(--text)}
.kd-item .why{font-size:.86rem;color:var(--muted);margin-top:.45rem;line-height:1.5}
.kd-badge{display:inline-block;padding:.05rem .5rem;border-radius:999px;font-size:.76rem;font-weight:600;margin-right:.4rem;color:#111}
.kd-badge.gen{background:#F4A99E}.kd-badge.rev{background:#F7D178}.kd-badge.sim{background:#CFC5FF}
.kd-table{width:100%%;border-collapse:collapse;font-size:.9rem;background:var(--surface);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.kd-table th{text-align:left;background:var(--surface2);padding:.5rem .7rem;font-weight:600;color:var(--text)}
.kd-table td{padding:.5rem .7rem;border-top:1px solid var(--line);color:var(--text);vertical-align:top}
.kd-bar2{height:8px;border-radius:4px;background:var(--ring-bg);position:relative;min-width:90px}
.kd-bar2>i{position:absolute;inset:0 auto 0 0;border-radius:4px;background:var(--accent)}
.kd-note{border:1px solid var(--line);background:var(--surface2);border-radius:10px;padding:.7rem .95rem;font-size:.9rem;color:var(--text);margin:.6rem 0}
.kd-foot{color:var(--muted);font-size:.82rem;margin-top:2rem;border-top:1px solid var(--line);padding-top:.9rem}
.kd-grid2{display:grid;grid-template-columns:1fr 1fr;gap:.9rem}

@media (max-width:760px){
 .kd-cards,.kd-grid2{grid-template-columns:1fr}
 .kd-hero{padding:1.6rem 1.2rem}.kd-hero h1{font-size:1.65rem}
 .kd-doc{padding:1rem 1.05rem}.kd-tag{display:none}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
"""


def build_css(theme: str, hero_b64: str) -> str:
    t = TOKENS["light" if theme == "light" else "dark"]
    css = BASE % t
    return f"<style>{css}\n.kd-hero{{--hero-img:url(data:image/jpeg;base64,{hero_b64})}}</style>"
