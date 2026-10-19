"""Optional Claude re-ranking: judges which candidates work as standalone short clips
and writes a title for each. Needs ANTHROPIC_API_KEY (or `ant auth login`)."""

import json

MODEL = "claude-opus-5-5"

SYSTEM = """You are the editor of a short-form video channel (TikTok, Reels, Shorts).
You get transcript excerpts of candidate clips cut from a long livestream or video.
Judge each one as a standalone short: does it have a hook in the first seconds, a
payoff, emotion or humor, and does it make sense without the rest of the stream?
Score 0-100. Write a punchy title (max 60 chars) in the transcript's language and a
one-sentence reason. Score every candidate you are given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "title", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def rerank(clips, weight=0.65, effort="medium"):
    """Blend Claude's 0-100 score with the heuristic score (rank-normalized)."""
    import anthropic

    payload = [
        {"id": i, "start": round(c.start, 1), "duration": round(c.duration, 1), "transcript": c.text}
        for i, c in enumerate(clips)
    ]
    client = anthropic.Anthropic()
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
    )
    if response.stop_reason == "refusal":
        print("  Claude declined to score these clips; keeping heuristic ranking.")
        return clips
    if response.stop_reason == "max_tokens":
        print("  Claude's answer was cut off; keeping heuristic ranking.")
        return clips

    text = next(b.text for b in response.content if b.type == "text")
    verdicts = {v["id"]: v for v in json.loads(text)["clips"]}

    order = sorted(range(len(clips)), key=lambda i: clips[i].score)
    heur_rank = {i: r / max(1, len(clips) - 1) for r, i in enumerate(order)}
    for i, c in enumerate(clips):
        v = verdicts.get(i)
        if not v:
            c.score = (1 - weight) * heur_rank[i]
            continue
        c.title, c.reason = v["title"], v["reason"]
        c.features["llm"] = v["score"]
        c.score = weight * (v["score"] / 100) + (1 - weight) * heur_rank[i]
    return clips
