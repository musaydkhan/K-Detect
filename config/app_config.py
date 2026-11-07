"""Central settings for the K-Detect web app (not for model training)."""
import os

BRAND = "K-Detect"
TAGLINE = "free AI-writing and similarity checker for students"
HEADER_LINE = f"{BRAND} | {TAGLINE}"

# ---- Input limits -------------------------------------------------------
MIN_WORDS = 100          # below this the detector is not reliable
MAX_WORDS = 5000         # keeps CPU time reasonable on a free Hugging Face Space
MAX_FILE_MB = 10
ALLOWED_EXTENSIONS = ("docx", "pdf")

# ---- AI-writing detector ------------------------------------------------
# Blend of the two trained models. External RAID tests showed the word model
# alone generalises better than a 50/50 blend (see evaluation/), so the
# default is word-only. Set CHAR_WEIGHT > 0 to bring the character SVM back.
WORD_WEIGHT = float(os.environ.get("KDETECT_WORD_WEIGHT", 1.0))
CHAR_WEIGHT = round(1.0 - WORD_WEIGHT, 4)

WINDOW_TARGET_WORDS = 80   # passage size scored at once (models were trained on abstracts)
WINDOW_MIN_WORDS = 56
FLAG_THRESHOLD = 0.65      # P(AI) at/above this -> highlighted as likely AI
POSSIBLE_THRESHOLD = 0.50  # between this and FLAG -> "possible", pale highlight
LOW_SIGNAL_SHARE = 20.0    # document AI share below this % is reported as "low signal"

# ---- Similarity (plagiarism) check -----------------------------------------
SIM_MATCH_COVERAGE = 0.60  # share of a sentence's words found in one source
SIM_MATCH_RUN = 10         # ...or a single shared run of at least this many words
SIM_QUERIES_SCHOLARLY = 14 # searches per document against scholarly sources
SIM_QUERIES_ARXIV = 3      # arXiv asks for at most one request per 3 seconds, so it only gets a few
SIM_QUERIES_WEB = 5        # searches per document against the web (Brave)
WEB_DAILY_CAP = int(os.environ.get("KDETECT_WEB_DAILY_CAP", 30))  # protects the free credit
HTTP_TIMEOUT = 6
SIM_TIME_BUDGET = 25      # seconds; the similarity stage never runs longer than this
SIM_MAX_FAILS = 3         # stop calling a service after this many failed requests in a row

# Optional extra detector (Hugging Face model id). Off unless set. EXPERIMENTAL / untested.
HF_MODEL_ID = os.environ.get("KDETECT_HF_MODEL", "")
