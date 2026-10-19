"""Speech-to-text with word timestamps.

Transcript format (also accepted via --transcript):
    {"segments": [{"start": 1.2, "end": 4.0, "text": "...",
                   "words": [{"start": 1.2, "end": 1.5, "word": "hello"}, ...]}]}
"""

import json
from pathlib import Path


def load(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    segments = data["segments"] if isinstance(data, dict) else data
    for seg in segments:
        seg.setdefault("words", [])
    return segments


def save(segments, path):
    Path(path).write_text(
        json.dumps({"segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def transcribe(wav_path, model_size="small", language=None):
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise SystemExit(
            "faster-whisper is not installed. `pip install faster-whisper`, "
            "or pass --transcript / --no-transcript."
        ) from e

    model = WhisperModel(model_size, device="auto", compute_type="auto")
    raw, _ = model.transcribe(
        str(wav_path), language=language, word_timestamps=True, vad_filter=True
    )
    segments = []
    for s in raw:
        segments.append({
            "start": s.start,
            "end": s.end,
            "text": s.text.strip(),
            "words": [
                {"start": w.start, "end": w.end, "word": w.word.strip()}
                for w in (s.words or [])
            ],
        })
    return segments
