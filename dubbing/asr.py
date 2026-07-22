"""Stage 2 – Gujarati speech-to-text using Gemini, with song/music detection.

The audio is split into silence-bounded chunks (keeps the Gemini free tier happy
and makes timestamps more reliable), each chunk is transcribed to JSON segments,
and timestamps are shifted by the chunk offset.

Songs / bhajans / kirtans / music are NOT transcribed (when skip_songs=True):
Gemini marks them as {"kind": "song"} segments, the app keeps the ORIGINAL
audio for those stretches in the final dub (no dead air, no weird lyric dub).

Robustness (never skip content):
  bad JSON -> repair -> salvage valid objects -> retry with ANOTHER KEY ->
  split the chunk in half and retry each half.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from pydub import AudioSegment, silence

from .keys import KeyRotator

ASR_MODEL = "gemini-2.5-flash"

_ASR_PROMPT = """You are transcribing Gujarati (ગુજરાતી) speech for a video dubbing pipeline.

Listen to the audio clip and return ONLY a JSON array of segments:
[{"start": <float seconds>, "end": <float seconds>, "text": "<Gujarati text>", "kind": "speech"}]

Rules:
- All times are relative to the START of THIS clip.
- Use natural sentence/phrase boundaries (about 2-12 seconds each).
- Write the transcription in Gujarati script, exactly as spoken (including code-switched English words in Latin script).
- Merge short utterances; do not create hundreds of tiny segments.
- If there is no speech in the clip, return [].
- Return pure JSON only — no markdown, no comments.
- The JSON must be COMPLETE and VALID — never stop writing in the middle of the array.
{song_rule}
"""

_SONG_RULE_SKIP = """- SONGS / MUSIC: any section that is a song, bhajan, kirtan, dhun, aarti,
  chant with music, or singing must NOT be transcribed — do NOT write lyrics.
  Instead output ONE segment covering that whole section:
  {"start": <float>, "end": <float>, "text": "", "kind": "song"}
  Pure instrumental music: same thing, mark it "song"."""

_SONG_RULE_KEEP = """- SONGS / MUSIC: if a section is a song, bhajan, kirtan, dhun, aarti or
  chant with singing, transcribe the lyrics like normal speech but set
  "kind": "song" on those segments."""


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


# ------------------------------------------------------- bulletproof JSON parse
def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _repair_array(candidate: str) -> str:
    """Best-effort repair of a broken JSON array of objects."""
    t = candidate
    # trailing commas before } or ]  (", }" -> "}")
    t = re.sub(r",(\s*[}\]])", r"\1", t)
    if not t.rstrip().endswith("]"):
        # truncated output: cut after the last complete object, close the array
        cut = t.rfind("}")
        if cut != -1:
            t = t[: cut + 1]
            t = re.sub(r",\s*$", "", t)
            t += "]"
    return t


def _salvage_objects(text: str) -> list:
    """String-aware scan that recovers every complete {...} object, even if the
    overall array is malformed (never lose transcribed lines)."""
    objs, depth, start, in_str, esc = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    frag = text[start : i + 1]
                    try:
                        obj = json.loads(frag)
                    except Exception:
                        # maybe a trailing comma inside; one last nudge
                        try:
                            obj = json.loads(re.sub(r",(\s*})", r"\1", frag))
                        except Exception:
                            obj = None
                    if isinstance(obj, dict):
                        objs.append(obj)
                    start = None
    return objs


def _extract_json_array(text: str) -> list:
    """Parse the model's JSON array; repair/salvage if needed.

    Raises JSONDecodeError/ValueError only when truly hopeless — the key
    rotator treats those as retryable (another key re-attempts the chunk).
    """
    cleaned = _strip_fences(text)
    start = cleaned.find("[")
    if start == -1:
        raise ValueError(f"No JSON array in model output: {text[:200]!r}")
    end = cleaned.rfind("]")
    candidate = cleaned[start : end + 1] if end > start else cleaned[start:]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_repair_array(candidate))
    except json.JSONDecodeError:
        pass
    salvaged = _salvage_objects(candidate)
    if salvaged:
        return salvaged
    return json.loads(candidate)  # raise the ORIGINAL error


_JSON_FAIL_MARKERS = (
    "JSONDecodeError", "Expecting", "No JSON array",
    "Unterminated", "Extra data", "Invalid control character",
)


def _json_failure(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    return any(m in msg for m in _JSON_FAIL_MARKERS)


# ------------------------------------------------------------------ gemini call
def transcribe_chunk(
    chunk_path: str | Path,
    rotator: KeyRotator,
    model: str = ASR_MODEL,
    source_language: str = "Gujarati",
    skip_songs: bool = True,
    max_attempts: int | None = None,
) -> list[dict]:
    from google import genai  # imported lazily so the app still boots without it
    from google.genai import types

    prompt = _ASR_PROMPT.replace("Gujarati", source_language).replace(
        "{song_rule}", _SONG_RULE_SKIP if skip_songs else _SONG_RULE_KEEP
    )

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

    attempts = max_attempts or max(4 * len(rotator), 8)
    raw = rotator.execute(_call, max_attempts=attempts)
    return _normalize_segments(raw)
    # Note: files uploaded to the Gemini Files API expire automatically (~48 h),
    # so we skip explicit cleanup here.


def _normalize_segments(raw: list) -> list[dict]:
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            s = float(item["start"])
            e = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if e <= s:
            continue
        kind = str(item.get("kind", "") or "").strip().lower()
        t = str(item.get("text", "") or "").strip()
        music_placeholder = bool(t) and any(
            m in t.lower() for m in ("(music)", "[music]", "♪", "(song)")
        )
        if kind == "song" or music_placeholder:
            out.append({"start": s, "end": e, "text": "", "kind": "song"})
        elif not t:
            # empty text = the model says "no speech here" -> treat as music/song
            out.append({"start": s, "end": e, "text": "", "kind": "song"})
        else:
            out.append({"start": s, "end": e, "text": t, "kind": "speech"})
    out.sort(key=lambda d: d["start"])
    return out


# ------------------------------------------------- split-retry (never skip)
def _transcribe_chunk_safe(
    chunk_path: str | Path,
    rotator: KeyRotator,
    model: str,
    source_language: str,
    skip_songs: bool,
    depth: int = 0,
) -> list[dict]:
    try:
        return transcribe_chunk(
            chunk_path, rotator, model=model,
            source_language=source_language, skip_songs=skip_songs,
        )
    except Exception as exc:  # noqa: BLE001 - deliberate: we retry, never drop
        if depth >= 2 or not _json_failure(exc):
            raise
    # The model kept producing broken JSON for this long chunk ->
    # split the chunk in half and transcribe each half (content is preserved).
    chunk_path = Path(chunk_path)
    audio = AudioSegment.from_wav(str(chunk_path))
    half = len(audio) // 2
    if half < 3000:
        raise RuntimeError(f"Chunk too small to split but JSON keeps failing: {chunk_path.name}")
    out: list[dict] = []
    for j, (a, b) in enumerate(((0, half), (half, len(audio)))):
        part = chunk_path.with_name(f"{chunk_path.stem}_part{j}.wav")
        audio[a:b].export(str(part), format="wav")
        sub = _transcribe_chunk_safe(
            part, rotator, model, source_language, skip_songs, depth + 1
        )
        off = a / 1000.0
        for s in sub:
            s["start"] += off
            s["end"] += off
        out.extend(sub)
    return out


def _merge_song_ranges(songs: list[dict], gap_s: float = 3.0) -> list[dict]:
    """Merge overlapping / near-adjacent song ranges (songs split across chunks)."""
    merged: list[dict] = []
    for r in sorted(songs, key=lambda x: x["start"]):
        if merged and r["start"] <= merged[-1]["end"] + gap_s:
            merged[-1]["end"] = max(merged[-1]["end"], r["end"])
        else:
            merged.append({"start": r["start"], "end": r["end"]})
    return merged


# ------------------------------------------------------------------- orchestrate
def transcribe_full(
    wav_path: str | Path,
    tmp_dir: str | Path,
    rotator: KeyRotator,
    model: str = ASR_MODEL,
    source_language: str = "Gujarati",
    progress_cb=None,
    max_chunk_s: float = 75.0,
    skip_songs: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Transcribe the whole file. Returns (speech_segments, song_ranges)."""
    chunks = chunk_audio(wav_path, Path(tmp_dir) / "chunks", max_chunk_s=max_chunk_s)
    speech: list[dict] = []
    songs: list[dict] = []
    for i, (chunk_path, offset) in enumerate(chunks):
        segs = _transcribe_chunk_safe(
            chunk_path, rotator, model, source_language, skip_songs
        )
        for s in segs:
            s["start"] = round(s["start"] + offset, 3)
            s["end"] = round(s["end"] + offset, 3)
            if s.get("kind") == "song":
                songs.append({"start": s["start"], "end": s["end"]})
                if not skip_songs:
                    s["text"] = s.get("text") or "🎵 (song)"
                    speech.append(s)
            else:
                s.pop("kind", None)
                speech.append(s)
        if progress_cb:
            usage = " · ".join(f"{k}:{v}" for k, v in rotator.stats().items())
            progress_cb(
                (i + 1) / len(chunks),
                f"Chunk {i + 1}/{len(chunks)} · 🔑 {usage}"
                + (f" · 🎵 {len(songs)} song part(s) so far" if songs else ""),
            )
    speech.sort(key=lambda d: d["start"])
    songs = _merge_song_ranges(songs) if skip_songs else []
    return speech, songs
