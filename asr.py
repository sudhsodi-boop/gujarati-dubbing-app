"""Stage 2 – Gujarati speech-to-text using Gemini, with word/segment timestamps.

The audio is split into silence-bounded chunks (keeps the Gemini free tier happy
and makes timestamps more reliable), each chunk is transcribed to JSON segments,
and timestamps are shifted by the chunk offset.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pydub import AudioSegment, silence

from .keys import KeyRotator

ASR_MODEL = "gemini-2.5-flash"

_ASR_PROMPT = """You are transcribing Gujarati (ગુજરાતી) speech for a video dubbing pipeline.

Listen to the audio clip and return ONLY a JSON array of segments:
[{"start": <float seconds>, "end": <float seconds>, "text": "<Gujarati text>"}]

Rules:
- All times are relative to the START of THIS clip.
- Use natural sentence/phrase boundaries (about 2-12 seconds each).
- Write the transcription in Gujarati script, exactly as spoken (including code-switched English words in Latin script).
- Merge short utterances; do not create hundreds of tiny segments.
- If there is no speech in the clip, return [].
- Return pure JSON only — no markdown, no comments.
"""


# ---------------------------------------------------------------------- chunk
def chunk_audio(
    wav_path: str | Path,
    out_dir: str | Path,
    max_chunk_s: float = 75.0,
    min_chunk_s: float = 8.0,
    min_silence_ms: int = 600,
) -> list[tuple[Path, float]]:
    """Split a WAV into chunks bounded by silence. Returns [(chunk_path, offset_s)]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio = AudioSegment.from_wav(str(wav_path))
    total_ms = len(audio)
    dbfs = audio.dBFS
    thresh = (dbfs - 16) if dbfs != float("-inf") else -45

    sils = silence.detect_silence(
        audio, min_silence_len=min_silence_ms, silence_thresh=thresh
    )
    # mid-points of silent regions = candidate split points (ms)
    splits = sorted((a + b) // 2 for a, b in sils)

    chunks: list[tuple[Path, float]] = []
    start = 0
    idx = 0
    max_ms = int(max_chunk_s * 1000)
    min_ms = int(min_chunk_s * 1000)
    while start < total_ms:
        hard_end = min(start + max_ms, total_ms)
        # prefer the latest split point inside [start+min, hard_end]
        cut = None
        for sp in splits:
            if start + min_ms <= sp <= hard_end:
                cut = sp
            elif sp > hard_end:
                break
        end = cut if cut else hard_end
        if end <= start:
            end = hard_end
        seg = audio[start:end]
        # skip near-silent leftovers
        if len(seg) > 500:
            path = out_dir / f"chunk_{idx:04d}.wav"
            seg.export(str(path), format="wav")
            chunks.append((path, start / 1000.0))
            idx += 1
        start = end
    return chunks


# ------------------------------------------------------------------ gemini call
def _extract_json_array(text: str) -> list:
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array in model output: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def transcribe_chunk(
    chunk_path: str | Path,
    rotator: KeyRotator,
    model: str = ASR_MODEL,
    source_language: str = "Gujarati",
) -> list[dict]:
    from google import genai  # imported lazily so the app still boots without it
    from google.genai import types

    prompt = _ASR_PROMPT.replace("Gujarati", source_language)

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        uploaded = client.files.upload(file=str(chunk_path))
        # wait until the file is ACTIVE (usually instant for small clips)
        for _ in range(20):
            state = getattr(uploaded, "state", None)
            name = getattr(state, "name", state)
            if name in (None, "ACTIVE"):
                break
            if name == "FAILED":
                raise RuntimeError("Gemini file upload failed.")
            time.sleep(1.5)
            uploaded = client.files.get(name=uploaded.name)
        resp = client.models.generate_content(
            model=model,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        return _extract_json_array(resp.text or "[]")

    return _normalize_segments(rotator.execute(_call))
    # Note: files uploaded to the Gemini Files API expire automatically (~48 h),
    # so we skip explicit cleanup here.


def _normalize_segments(raw: list) -> list[dict]:
    out = []
    for item in raw:
        try:
            s = float(item["start"])
            e = float(item["end"])
            t = str(item.get("text", "")).strip()
        except (KeyError, TypeError, ValueError):
            continue
        if t and e > s:
            out.append({"start": s, "end": e, "text": t})
    out.sort(key=lambda d: d["start"])
    return out


# ------------------------------------------------------------------- orchestrate
def transcribe_full(
    wav_path: str | Path,
    tmp_dir: str | Path,
    rotator: KeyRotator,
    model: str = ASR_MODEL,
    source_language: str = "Gujarati",
    progress_cb=None,
    max_chunk_s: float = 75.0,
) -> list[dict]:
    chunks = chunk_audio(wav_path, Path(tmp_dir) / "chunks", max_chunk_s=max_chunk_s)
    all_segments: list[dict] = []
    for i, (chunk_path, offset) in enumerate(chunks):
        segs = transcribe_chunk(
            chunk_path, rotator, model=model, source_language=source_language
        )
        for s in segs:
            s["start"] = round(s["start"] + offset, 3)
            s["end"] = round(s["end"] + offset, 3)
        all_segments.extend(segs)
        if progress_cb:
            progress_cb((i + 1) / len(chunks), f"Chunk {i + 1}/{len(chunks)} transcribed")
    all_segments.sort(key=lambda d: d["start"])
    return all_segments
