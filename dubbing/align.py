"""Stage 4b – fit synthesized segments into the timeline and assemble the track.

Two pacing modes:

* "strict" – every segment starts at its exact original start time, stretched
  to fit its slot (best for the lip-sync step, can create speed jumps + gaps).
* "flow"   – CONTINUOUS speech: segments are placed back-to-back following the
  audio, artificial silences capped at ~1.2s, gentle even pacing. Use for
  sermons / Q&A / monologue-style videos where speech should never stall.

Speed changes use ffmpeg's `atempo` (pitch-preserving).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

MIN_SLOT_MS = 900

# ---- strict mode tuning ----
MAX_SPEEDUP = 1.45   # don't sound chipmunk-y
MAX_SLOWDOWN = 0.80  # don't sound drunk
OVERHANG = 1.12      # allow 12% bleed past the slot after max speedup

# ---- flow mode tuning ----
MAX_GAP_MS = 1200          # cap artificial silence between segments (kills awkward pauses)
FLOW_SPEED_TRIGGER = 1.25  # only speed up if audio exceeds 125% of its slot
FLOW_MAX_SPEEDUP = 1.50    # gentler than strict: keeps speech natural
FLOW_MAX_LEN = 1.60        # hard cap: audio may be at most 160% of slot


def _atempo_filter(factor: float) -> str:
    """Chain atempo filters (each must be within [0.5, 2.0])."""
    factors = []
    f = factor
    while f > 2.0:
        factors.append(2.0)
        f /= 2.0
    while f < 0.5:
        factors.append(0.5)
        f /= 0.5
    factors.append(f)
    return ",".join(f"atempo={x:.4f}" for x in factors)


def apply_atempo(audio: AudioSegment, factor: float, tmp_dir: str | Path) -> AudioSegment:
    if abs(factor - 1.0) < 0.01:
        return audio
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    src = tmp_dir / "stretch_in.wav"
    dst = tmp_dir / "stretch_out.wav"
    audio.export(str(src), format="wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-filter:a", _atempo_filter(factor),
        "-ar", str(audio.frame_rate), "-ac", str(audio.channels),
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return audio  # safest fallback: unstretched rather than crash
    return AudioSegment.from_wav(str(dst))


def stretch_to_fit(audio: AudioSegment, slot_ms: int, tmp_dir: str | Path) -> AudioSegment:
    """Strict mode: force the audio into its slot (speed up / mild slow down)."""
    slot_ms = max(int(slot_ms), MIN_SLOT_MS)
    length = len(audio)
    if length > slot_ms:
        factor = min(length / slot_ms, MAX_SPEEDUP)
        audio = apply_atempo(audio, factor, tmp_dir)
        if len(audio) > slot_ms * OVERHANG:
            audio = audio[: int(slot_ms * OVERHANG)].fade_out(60)
        return audio
    if length >= slot_ms * MAX_SLOWDOWN:
        return apply_atempo(audio, length / slot_ms, tmp_dir)
    return audio  # very short → leave natural, silence pads the gap


def _flow_fit(audio: AudioSegment, slot_ms: int, tmp_dir: str | Path) -> AudioSegment:
    """Flow mode: only tame overly-long lines; never pad or slow."""
    slot_ms = max(int(slot_ms), MIN_SLOT_MS)
    length = len(audio)
    if length > slot_ms * FLOW_SPEED_TRIGGER:
        factor = min(length / slot_ms, FLOW_MAX_SPEEDUP)
        audio = apply_atempo(audio, factor, tmp_dir)
        if len(audio) > slot_ms * FLOW_MAX_LEN:
            audio = audio[: int(slot_ms * FLOW_MAX_LEN)].fade_out(60)
    return audio


def _ensure_length(track: AudioSegment, need_ms: int) -> AudioSegment:
    if need_ms > len(track):
        track = track + AudioSegment.silent(duration=need_ms - len(track))
    return track


def build_track(
    segments: list[dict],
    total_s: float,
    out_wav: str | Path,
    tmp_dir: str | Path,
    progress_cb=None,
    mode: str = "flow",
) -> Path:
    segs = sorted(segments, key=lambda s: s.get("start", 0.0))
    track = AudioSegment.silent(duration=int((total_s or 30) * 1000) + 300)
    cursor_ms: int | None = None  # flow-mode write head

    placed = 0
    for i, seg in enumerate(segs):
        wav = seg.get("wav")
        if not wav or not Path(wav).exists():
            continue
        audio = AudioSegment.from_wav(wav)
        slot_ms = int((seg["end"] - seg["start"]) * 1000)

        if mode == "strict":
            audio = stretch_to_fit(audio, slot_ms, tmp_dir)
            place_ms = int(seg["start"] * 1000)
        else:  # ---- continuous speech ----
            audio = _flow_fit(audio, slot_ms, tmp_dir)
            ideal = int(seg["start"] * 1000)
            if cursor_ms is None:
                place_ms = ideal
            else:
                gap = ideal - cursor_ms
                if gap > MAX_GAP_MS:
                    place_ms = cursor_ms + MAX_GAP_MS   # pause killer
                elif gap > 0:
                    place_ms = ideal                    # keep short natural gap
                else:
                    place_ms = cursor_ms                # chain back-to-back

        track = _ensure_length(track, place_ms + len(audio))
        track = track.overlay(audio, position=place_ms)
        cursor_ms = place_ms + len(audio)
        placed += 1
        if progress_cb:
            progress_cb((i + 1) / len(segs), f"Aligned {i + 1}/{len(segs)}")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    normalize(track, headroom=1.0).export(str(out_wav), format="wav")
    return out_wav
