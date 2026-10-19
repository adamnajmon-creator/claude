"""Live-chat logs as a highlight signal (chat explodes when something happens).

Accepted formats:
  - [{"t": 12.3, "text": "..."}]                      (seconds from VOD start)
  - TwitchDownloader JSON: {"comments": [{"content_offset_seconds": 12.3, ...}]}
  - yt-dlp live_chat.json (one JSON object per line, videoOffsetTimeMsec)
"""

import json
from pathlib import Path

import numpy as np


def load_times(path):
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        times = []
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            ms = obj.get("videoOffsetTimeMsec") or obj.get("replayChatItemAction", {}).get("videoOffsetTimeMsec")
            if ms is not None:
                times.append(int(ms) / 1000)
        return times
    if isinstance(data, dict) and "comments" in data:
        return [c["content_offset_seconds"] for c in data["comments"]]
    return [m["t"] for m in data]


def density(times, duration, hop):
    bins = int(np.ceil(duration / hop))
    hist, _ = np.histogram(times, bins=bins, range=(0, bins * hop))
    return hist.astype(np.float32) / hop
