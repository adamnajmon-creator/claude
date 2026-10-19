# autoclip

Turns a long stream or VOD into ranked vertical short clips with burned-in
word-by-word captions. It works like Restream's or Opus Clip's auto-clipping.

## How it works

1. **Audio**: ffmpeg extracts the audio, then computes a loudness curve (RMS in dB every 0.25 s).
2. **Transcript**: faster-whisper produces word timestamps. You can also pass your own transcript JSON.
3. **Candidates**: windows of 15–60 s that start and end on sentence boundaries (or on quiet points when there is no transcript).
4. **Scoring**: a weighted sum of z-scored signals:
   - loudness spike over the 60 s baseline (reactions, shouting)
   - chat rate, shifted by the ~6 s chat delay
   - hype words in the transcript (CZ/SK/EN)
   - speech density and the first 3 s (the hook)
   - scene cuts; dead air counts against a window
5. **Claude (optional, `--llm`)**: re-ranks the top 25 for standalone viral potential and writes titles.
6. **Selection**: non-max suppression, so the chosen clips don't overlap.
7. **Render**: 9:16 1080×1920 output with one of these layouts:
   - `face`: crop centered on the face
   - `crop`: center crop
   - `blur`: the whole frame over a blurred background
   - `original`: keep the source aspect ratio

   Captions are rendered with ASS, highlighting the current word.

## Usage

```bash
pip install -r requirements.txt          # ffmpeg must be on PATH
yt-dlp -o vod.mp4 <stream URL>           # download a VOD

python -m autoclip vod.mp4 -n 5                         # 5 clips, face crop, captions
python -m autoclip vod.mp4 --chat chat.json --llm       # + chat signal + Claude titles
python -m autoclip vod.mp4 --layout blur --language cs
python -m autoclip vod.mp4 --no-transcript --dry-run    # audio only, ranking without rendering
```

Output: `clips/clip_NN_XXmYYs.mp4` plus `clips.json`, which holds each clip's
times, score, title and individual signals. With Whisper it also writes `transcript.json`.

Chat logs: TwitchDownloader JSON (`TwitchDownloaderCLI chatdownload`), yt-dlp
`live_chat.json` (`--write-subs --sub-langs live_chat`), or `[{"t": sec, "text": ...}]`.

`--llm` needs `ANTHROPIC_API_KEY`. Tune the signal weights in `WEIGHTS` in
`autoclip/highlights.py`.
