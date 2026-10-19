"""ffmpeg/ffprobe helpers: probing, audio extraction, loudness envelope, scene cuts."""

import json
import re
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def probe(path):
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]).stdout
    info = json.loads(out)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    return {
        "duration": float(info["format"]["duration"]),
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
    }


def extract_audio(path, out_wav):
    run([
        "ffmpeg", "-y", "-v", "error", "-i", str(path),
        "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(out_wav),
    ])
    return Path(out_wav)


def load_pcm(wav_path):
    raw = Path(wav_path).read_bytes()
    # Skip the 44-byte canonical WAV header written by ffmpeg.
    return np.frombuffer(raw[44:], dtype=np.int16).astype(np.float32) / 32768.0


def loudness_envelope(samples, hop=0.25):
    """RMS loudness in dBFS per `hop` seconds."""
    n = int(SAMPLE_RATE * hop)
    frames = len(samples) // n
    if frames == 0:
        return np.zeros(0), hop
    block = samples[: frames * n].reshape(frames, n)
    rms = np.sqrt(np.mean(block ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms), hop


def scene_changes(path, threshold=0.3):
    """Timestamps (s) where the picture changes hard (cuts, camera switches)."""
    proc = subprocess.run([
        "ffmpeg", "-v", "info", "-i", str(path), "-an",
        "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-",
    ], capture_output=True, text=True)
    return [float(t) for t in re.findall(r"pts_time:([0-9.]+)", proc.stderr)]
