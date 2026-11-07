"""Passage-level AI-writing scoring with the SciHRA word/char models."""
from __future__ import annotations
import sys
from pathlib import Path
import joblib
import numpy as np
from config import app_config as cfg

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "runtime"
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:                       # pickled models import core.model_preprocessor
    sys.path.insert(0, _ROOT)


class Detector:
    def __init__(self):
        self.word = joblib.load(MODEL_DIR / "word_logistic.joblib")
        self.char = joblib.load(MODEL_DIR / "char_svm_calibrated.joblib") if cfg.CHAR_WEIGHT > 0 else None
        self.extra = _load_extra(cfg.HF_MODEL_ID)

    # ---- model call ----------------------------------------------------
    def _proba(self, texts):
        """Return (n,3) probabilities: human, AI-revised, AI-generated."""
        p = cfg.WORD_WEIGHT * self.word.predict_proba(texts)
        if self.char is not None:
            p = p + cfg.CHAR_WEIGHT * self.char.predict_proba(texts)
        if self.extra is not None:                # experimental: average P(AI) with an HF model
            q = self.extra(texts)                 # P(AI) per text
            ai = 1 - p[:, 0]
            new_ai = 0.5 * ai + 0.5 * q
            scale = np.where(ai > 0, new_ai / np.maximum(ai, 1e-9), 1.0)
            p[:, 1:] = p[:, 1:] * scale[:, None]
            p[:, 0] = 1 - p[:, 1:].sum(1)
        return p

    # ---- windows ---------------------------------------------------------
    @staticmethod
    def _windows(sents):
        """Overlapping runs of consecutive scored sentences of ~WINDOW_TARGET words."""
        n, out, i = len(sents), [], 0
        target, half = cfg.WINDOW_TARGET_WORDS, cfg.WINDOW_TARGET_WORDS // 2
        while i < n:
            j, w = i, 0
            while j < n and w < target:
                w += sents[j].words; j += 1
            if w < cfg.WINDOW_MIN_WORDS:             # tail too short: extend backwards
                k = i
                while k > 0 and w < cfg.WINDOW_MIN_WORDS:
                    k -= 1; w += sents[k].words
                i = k
            out.append((i, j))
            if j >= n:
                break
            step, acc = i, 0
            while step < j - 1 and acc < half:
                acc += sents[step].words; step += 1
            i = max(step, i + 1)
        return out

    def score(self, sentences):
        """Fill p_* and ai_label on each scored Sentence. Returns the number of windows used."""
        sc = [s for s in sentences if s.scored]
        if sum(s.words for s in sc) < cfg.WINDOW_MIN_WORDS:
            return 0
        wins = self._windows(sc)
        probs = self._proba([" ".join(s.text for s in sc[a:b]) for a, b in wins])
        acc = np.zeros((len(sc), 3)); cnt = np.zeros(len(sc))
        for (a, b), p in zip(wins, probs):
            acc[a:b] += p; cnt[a:b] += 1
        for s, a, c in zip(sc, acc, cnt):
            a = a / max(c, 1)
            s.p_human, s.p_revised, s.p_generated = map(float, a)
            s.p_ai = 1.0 - s.p_human
            if s.p_ai >= cfg.FLAG_THRESHOLD:
                s.ai_label = "generated" if s.p_generated >= s.p_revised else "revised"
            elif s.p_ai >= cfg.POSSIBLE_THRESHOLD:
                s.ai_label = "possible"
            else:
                s.ai_label = "human"
        return len(wins)


def summarize(sentences) -> dict:
    """Word-weighted document summary."""
    sc = [s for s in sentences if s.scored and s.ai_label != "none"]
    total = sum(s.words for s in sc) or 1
    share = lambda lab: 100.0 * sum(s.words for s in sc if s.ai_label == lab) / total
    gen, rev, pos = share("generated"), share("revised"), share("possible")
    ai = gen + rev
    return {
        "scored_words": sum(s.words for s in sc), "generated": round(gen, 1), "revised": round(rev, 1),
        "possible": round(pos, 1), "human": round(share("human"), 1), "ai_share": round(ai, 1),
        "low_signal": 0 < ai < cfg.LOW_SIGNAL_SHARE,
    }


def _load_extra(model_id: str):
    """Optional Hugging Face text-classification model (EXPERIMENTAL, off by default)."""
    if not model_id:
        return None
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model=model_id, truncation=True, max_length=256, top_k=None)
    except Exception:
        return None

    def run(texts):
        out = []
        for res in clf(list(texts)):
            ai = 0.0
            for r in res:
                if any(k in r["label"].lower() for k in ("ai", "fake", "gpt", "machine", "generated", "label_1")):
                    ai = max(ai, r["score"])
            out.append(ai)
        return np.array(out)
    return run
