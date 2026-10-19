"""Candidate generation and heuristic highlight scoring.

The signals are the ones commercial auto-clippers (Restream, Opus Clip, Twitch
clip suggestions...) are built on: loudness spikes over the local baseline,
chat-rate spikes, hype words, speech density, a strong first few seconds, and
visual cuts. Each candidate window gets a weighted sum of z-scored features.
"""

import re
from dataclasses import dataclass, field

import numpy as np

HYPE_WORDS = [
    # en
    "oh my god", "omg", "no way", "what", "wtf", "holy", "insane", "crazy", "let's go",
    "lets go", "clutch", "hahaha", "haha", "lol", "wow", "yes", "noo", "bro", "dude",
    "unbelievable", "oh no", "watch this", "look at this",
    # cs / sk
    "ty vole", "kurva", "cože", "co to", "do prdele", "ježiš", "neee", "jo jo", "pecka",
    "šílený", "brutální", "to snad ne", "fakt", "hustý", "pozor", "kámo", "ty jo",
]
HYPE_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in HYPE_WORDS) + r")\b", re.I)

WEIGHTS = {
    "spike": 1.4,      # loudness peak above the rolling baseline (reactions, screams)
    "energy": 0.6,     # sustained loudness
    "chat": 1.5,       # chat messages/sec (lagged), strongest signal when present
    "hype": 1.0,       # hype words per second
    "speech": 0.5,     # words per second
    "hook": 0.9,       # energy + speech in the first 3 seconds
    "cuts": 0.3,       # scene changes per second
    "silence": -0.8,   # share of dead air
}


@dataclass
class Clip:
    start: float
    end: float
    text: str = ""
    features: dict = field(default_factory=dict)
    score: float = 0.0
    title: str = ""
    reason: str = ""

    @property
    def duration(self):
        return self.end - self.start


def _rolling_mean(x, n):
    if n <= 1 or len(x) == 0:
        return x
    c = np.cumsum(np.insert(x, 0, 0.0))
    half = n // 2
    idx = np.arange(len(x))
    lo = np.clip(idx - half, 0, len(x))
    hi = np.clip(idx + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def candidates_from_transcript(segments, duration, min_len, max_len, pad=0.25):
    out = []
    for i, first in enumerate(segments):
        emitted = 0
        for j in range(i, len(segments)):
            span = segments[j]["end"] - first["start"]
            if span > max_len:
                break
            if span >= min_len:
                start = max(0.0, first["start"] - pad)
                end = min(duration, segments[j]["end"] + pad)
                text = " ".join(s["text"] for s in segments[i : j + 1])
                out.append(Clip(start, end, text))
                emitted += 1
                if emitted >= 3:
                    break
    return out


def candidates_from_audio(env, hop, duration, min_len, max_len, step=2.0):
    """No transcript: fixed windows whose edges snap to the quietest nearby point."""
    def snap(t):
        i0 = max(0, int((t - 1.5) / hop))
        i1 = min(len(env), int((t + 1.5) / hop) + 1)
        if i1 <= i0:
            return t
        return (i0 + int(np.argmin(env[i0:i1]))) * hop

    lengths = sorted({min_len, (min_len + max_len) / 2, max_len})
    out = []
    t = 0.0
    while t + min_len <= duration:
        for length in lengths:
            if t + length > duration:
                continue
            start, end = snap(t), snap(t + length)
            if min_len * 0.8 <= end - start <= max_len * 1.1:
                out.append(Clip(start, end))
        t += step
    return out


def score(clips, *, env, hop, segments=None, chat_rate=None, cuts=None, chat_lag=6.0):
    if not clips:
        return clips
    baseline = np.median(env)
    spike_curve = env - _rolling_mean(env, int(60 / hop))
    smooth = _rolling_mean(env, int(1 / hop))
    silence_floor = baseline - 15

    words = []
    for s in segments or []:
        if s["words"]:
            words.extend((w["start"], w["word"]) for w in s["words"])
        else:  # segment-level timing only: spread words evenly
            toks = s["text"].split()
            for k, tok in enumerate(toks):
                words.append((s["start"] + (s["end"] - s["start"]) * k / max(1, len(toks)), tok))
    word_t = np.array([w[0] for w in words]) if words else np.zeros(0)
    cuts = np.array(cuts or [])

    for c in clips:
        a, b = int(c.start / hop), max(int(c.start / hop) + 1, int(c.end / hop))
        win = smooth[a:b]
        f = {}
        f["energy"] = float(np.mean(win) - baseline)
        f["spike"] = float(np.max(spike_curve[a:b]))
        f["silence"] = float(np.mean(env[a:b] < silence_floor))
        h = min(b, a + int(3 / hop))
        f["hook"] = float(np.mean(smooth[a:h]) - baseline)

        n_words = int(np.sum((word_t >= c.start) & (word_t < c.end))) if len(word_t) else 0
        f["speech"] = n_words / c.duration
        f["hook"] += 2.0 * (int(np.sum((word_t >= c.start) & (word_t < c.start + 3))) if len(word_t) else 0) / 3
        f["hype"] = len(HYPE_RE.findall(c.text)) / c.duration if c.text else 0.0

        if chat_rate is not None and len(chat_rate):
            ca = int((c.start + chat_lag * 0.5) / hop)
            cb = int((c.end + chat_lag) / hop)
            f["chat"] = float(np.mean(chat_rate[ca:cb])) if cb > ca and ca < len(chat_rate) else 0.0
        else:
            f["chat"] = 0.0
        f["cuts"] = float(np.sum((cuts >= c.start) & (cuts < c.end))) / c.duration if len(cuts) else 0.0
        c.features = f

    keys = list(WEIGHTS)
    mat = np.array([[c.features[k] for k in keys] for c in clips], dtype=np.float64)
    std = mat.std(axis=0)
    z = np.where(std > 1e-9, (mat - mat.mean(axis=0)) / np.where(std > 1e-9, std, 1), 0.0)
    w = np.array([WEIGHTS[k] for k in keys])
    for c, row in zip(clips, z @ w):
        c.score = float(row)
    return clips


def select(clips, count, max_overlap=0.0):
    """Greedy non-max suppression: best clips first, drop ones that overlap picked ones."""
    picked = []
    for c in sorted(clips, key=lambda c: c.score, reverse=True):
        ok = True
        for p in picked:
            inter = min(c.end, p.end) - max(c.start, p.start)
            if inter > max_overlap * min(c.duration, p.duration):
                ok = False
                break
        if ok:
            picked.append(c)
            if len(picked) >= count:
                break
    return picked
