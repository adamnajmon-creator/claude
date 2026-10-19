"""autoclip: turn a long stream/VOD into ranked vertical short clips."""

import argparse
import json
import tempfile
from pathlib import Path

from . import chat, highlights, media, render, transcribe


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="autoclip", description=__doc__)
    p.add_argument("video", help="input video file (download VODs with yt-dlp first)")
    p.add_argument("-o", "--out", default="clips", help="output directory")
    p.add_argument("-n", "--count", type=int, default=5, help="number of clips")
    p.add_argument("--min", dest="min_len", type=float, default=15, help="min clip length (s)")
    p.add_argument("--max", dest="max_len", type=float, default=60, help="max clip length (s)")
    p.add_argument("--layout", choices=["face", "crop", "blur", "original"], default="face",
                   help="face: 9:16 crop centered on the speaker; crop: center 9:16; "
                        "blur: full frame over blurred fill; original: keep aspect")
    p.add_argument("--transcript", help="existing transcript JSON (skips Whisper)")
    p.add_argument("--no-transcript", action="store_true", help="audio/chat signals only")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--language", help="force transcription language, e.g. cs, en")
    p.add_argument("--chat", help="chat log (TwitchDownloader JSON, yt-dlp live_chat.json, or [{t,text}])")
    p.add_argument("--no-captions", action="store_true")
    p.add_argument("--no-scenes", action="store_true", help="skip scene-cut detection (faster)")
    p.add_argument("--llm", action="store_true", help="re-rank and title clips with Claude")
    p.add_argument("--llm-pool", type=int, default=25, help="candidates sent to Claude")
    p.add_argument("--dry-run", action="store_true", help="rank only, don't render")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    video = Path(args.video)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = media.probe(video)
    print(f"[1/5] {video.name}: {info['duration']:.0f}s, {info['width']}x{info['height']}")

    with tempfile.TemporaryDirectory() as tmp:
        wav = media.extract_audio(video, Path(tmp) / "audio.wav")
        env, hop = media.loudness_envelope(media.load_pcm(wav))

        segments = []
        if args.transcript:
            segments = transcribe.load(args.transcript)
        elif not args.no_transcript:
            print(f"[2/5] transcribing (whisper {args.whisper_model})...")
            segments = transcribe.transcribe(wav, args.whisper_model, args.language)
            transcribe.save(segments, out / "transcript.json")
    print(f"[2/5] transcript: {len(segments)} segments")

    chat_rate = None
    if args.chat:
        times = chat.load_times(args.chat)
        chat_rate = chat.density(times, info["duration"], hop)
        print(f"      chat: {len(times)} messages")

    cuts = [] if args.no_scenes or not info["width"] else media.scene_changes(video)

    if segments:
        clips = highlights.candidates_from_transcript(segments, info["duration"], args.min_len, args.max_len)
    else:
        clips = highlights.candidates_from_audio(env, hop, info["duration"], args.min_len, args.max_len)
    highlights.score(clips, env=env, hop=hop, segments=segments, chat_rate=chat_rate, cuts=cuts)
    print(f"[3/5] scored {len(clips)} candidate windows")

    if args.llm and segments:
        from . import llm
        pool = highlights.select(clips, args.llm_pool, max_overlap=0.3)
        print(f"[4/5] Claude re-ranking {len(pool)} candidates...")
        clips = llm.rerank(pool)
    best = highlights.select(clips, args.count)
    best.sort(key=lambda c: c.start)

    words = [w for s in segments for w in s["words"]]
    manifest = []
    for i, c in enumerate(best, 1):
        name = f"clip_{i:02d}_{int(c.start // 60):02d}m{int(c.start % 60):02d}s.mp4"
        print(f"[5/5] {name}  {c.duration:4.1f}s  score={c.score:+.2f}  {c.title or c.text[:60]}")
        if not args.dry_run:
            render.render(video, c, out / name, layout=args.layout,
                          src_w=info["width"], src_h=info["height"],
                          words=words, captions=not args.no_captions)
        manifest.append({
            "file": name, "start": round(c.start, 2), "end": round(c.end, 2),
            "score": round(c.score, 3), "title": c.title, "reason": c.reason,
            "transcript": c.text, "features": {k: round(v, 3) for k, v in c.features.items()},
        })
    (out / "clips.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done -> {out}/")


if __name__ == "__main__":
    main()
