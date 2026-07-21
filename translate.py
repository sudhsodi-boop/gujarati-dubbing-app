"""Stage 3 – translate Gujarati segments to the target language with Gemini.

Translation is duration-aware: each segment carries its original speaking
time so the model keeps the translated line speakable in roughly the same
window (keeps the later lip-sync tight).
"""

from __future__ import annotations

from .asr import _extract_json_array
from .keys import KeyRotator

TRANSLATE_MODEL = "gemini-2.5-flash"

_PROMPT = """You are localizing a video. Translate each segment below from {src} into NATURAL, SPOKEN {dst}.

Input JSON: [{{"id": <int>, "seconds": <float>, "text": "<src text>"}}, ...]

Rules:
- The translation of each segment must be speakable aloud in about the same number of seconds given (this is a dub — brevity matters).
- Use simple, conversational language (how a person would actually say it), not literal word-for-word translation.
- Keep names, brands, numbers and facts accurate.
- Return ONLY a JSON array: [{{"id": <int>, "text": "<translation>"}}, ...] with one entry per input id, in order.
- No markdown, no commentary.
"""


def translate_segments(
    segments: list[dict],
    target_language: str,
    rotator: KeyRotator,
    model: str = TRANSLATE_MODEL,
    source_language: str = "Gujarati",
    batch_size: int = 40,
    progress_cb=None,
) -> list[dict]:
    from google import genai
    from google.genai import types

    payload_all = [
        {
            "id": i,
            "seconds": round(seg["end"] - seg["start"], 2),
            "text": seg["text"],
        }
        for i, seg in enumerate(segments)
    ]

    def _translate_batch(batch: list[dict]) -> dict[int, str]:
        prompt = _PROMPT.format(src=source_language, dst=target_language) + (
            "\nInput JSON:\n" + _json_dumps(batch)
        )

        def _call(api_key: str):
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.3,
                ),
            )
            return _extract_json_array(resp.text or "[]")

        arr = rotator.execute(_call)
        out: dict[int, str] = {}
        for item in arr:
            try:
                out[int(item["id"])] = str(item["text"]).strip()
            except (KeyError, TypeError, ValueError):
                continue
        return out

    translations: dict[int, str] = {}
    n_batches = (len(payload_all) + batch_size - 1) // batch_size
    for b in range(0, len(payload_all), batch_size):
        translations.update(_translate_batch(payload_all[b : b + batch_size]))
        if progress_cb:
            done = min((b // batch_size) + 1, n_batches)
            progress_cb(done / n_batches, f"Translated batch {done}/{n_batches}")

    result = []
    for i, seg in enumerate(segments):
        row = dict(seg)
        row["translated"] = translations.get(i, seg["text"])
        result.append(row)
    return result


def _json_dumps(obj) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)
