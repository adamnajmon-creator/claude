"""Cutting, vertical reframing and burned-in word-by-word captions."""

import subprocess
from pathlib import Path

OUT_W, OUT_H = 1080, 1920


def face_center_x(video, start, end, src_w):
    """Median horizontal position of the largest face, sampled every 0.5 s."""
    try:
        import cv2
    except ImportError:
        return None
    if not hasattr(cv2, "CascadeClassifier"):  # OpenCV 5 removed Haar cascades
        print("  face layout needs opencv-python-headless<5; using center crop")
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(video))
    xs = []
    t = start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        t += 0.5
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(gray.shape[0] // 12,) * 2)
        if len(faces):
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            xs.append((x + w / 2) * src_w / frame.shape[1])
    cap.release()
    if not xs:
        return None
    xs.sort()
    return xs[len(xs) // 2]


def video_filter(layout, src_w, src_h, center_x=None):
    if layout == "original":
        return "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if layout == "blur":
        return (
            f"split[a][b];"
            f"[a]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W}:{OUT_H},gblur=sigma=30,eq=brightness=-0.08[bg];"
            f"[b]scale={OUT_W}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    # crop / face: 9:16 window out of the source, positioned on center_x.
    crop_w = min(src_w, int(src_h * 9 / 16) // 2 * 2)
    cx = center_x if center_x is not None else src_w / 2
    x = int(min(max(cx - crop_w / 2, 0), src_w - crop_w))
    return f"crop={crop_w}:{src_h}:{x}:0,scale={OUT_W}:{OUT_H}"


def _ass_time(t):
    t = max(0.0, t)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def _ass_escape(text):
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")")


def write_ass(words, clip_start, clip_end, path, play_w, play_h, group=3):
    """One subtitle event per spoken word: the chunk is shown, the current word pops."""
    font_size = int(play_h * 0.045)
    margin_v = int(play_h * 0.22)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_w}
PlayResY: {play_h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,{font_size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,{max(2, font_size // 12)},2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    clip_words = [w for w in words if w["end"] > clip_start and w["start"] < clip_end and w["word"]]
    lines = []
    for g in range(0, len(clip_words), group):
        chunk = clip_words[g : g + group]
        for k, w in enumerate(chunk):
            start = w["start"] - clip_start
            nxt = chunk[k + 1]["start"] if k + 1 < len(chunk) else w["end"]
            end = max(nxt, w["end"]) - clip_start
            parts = []
            for j, other in enumerate(chunk):
                word = _ass_escape(other["word"].upper())
                if j == k:
                    parts.append(r"{\c&H00E5FF&\fscx112\fscy112}" + word + r"{\r}")
                else:
                    parts.append(word)
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,{' '.join(parts)}")
    Path(path).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def render(video, clip, out_path, *, layout, src_w, src_h, words=None, captions=True):
    out_path = Path(out_path)
    center_x = None
    if layout == "face":
        center_x = face_center_x(video, clip.start, clip.end, src_w)
    vf = video_filter(layout, src_w, src_h, center_x)

    if captions and words:
        ass = out_path.with_suffix(".ass")
        if layout == "original":
            play_w, play_h = src_w, src_h
        else:
            play_w, play_h = OUT_W, OUT_H
        write_ass(words, clip.start, clip.end, ass, play_w, play_h)
        vf += f",ass={ass.name}"

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-ss", f"{clip.start:.3f}", "-i", str(Path(video).resolve()), "-t", f"{clip.duration:.3f}",
        "-filter_complex", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
        out_path.name,
    ]
    # Run inside the output dir so the .ass path needs no filter escaping.
    subprocess.run(cmd, check=True, cwd=out_path.parent)
    return out_path
