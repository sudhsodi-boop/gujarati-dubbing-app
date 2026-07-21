"""Stage 4b – fit each synthesized segment into its original time slot and
assemble the full dubbed audio track.

Speed changes use ffmpeg's `atempo` (pitch-preserving). A segment longer than
its slot is sped up (capped, then hard-trimmed with a fade); a shorter segment
is mildly stretched (down to 0.80x) or padded with silence.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

MIN_SLOT_MS = 900
MAX_SPEEDUP = 1.45   # don't sound chipmunk-y
MAX_SLOWDOWN = 0.80  # don't sound drunk
OVERHANG = 1.12      # allow 12% bleed past the slot after max speedup


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
        # safest fallback: return unstretched rather than crash the pipeline
        return audio
    return AudioSegment.from_wav(str(dst))


def stretch_to_fit(
    audio: AudioSegment, slot_ms: int, tmp_dir: str | Path
) -> AudioSegment:
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


def build_track(
    segments: list[dict],
    total_s: float,
    out_wav: str | Path,
    tmp_dir: str | Path,
    progress_cb=None,
) -> Path:
    total_ms = int(total_s * 1000) + 300
    track = AudioSegment.silent(duration=total_ms)

    placed = 0
    for i, seg in enumerate(segments):
        wav = seg.get("wav")
        if not wav or not Path(wav).exists():
            continue
        audio = AudioSegment.from_wav(wav)
        slot_ms = int((seg["end"] - seg["start"]) * 1000)
        audio = stretch_to_fit(audio, slot_ms, tmp_dir)
        start_ms = min(int(seg["start"] * 1000), total_ms - 1)
        end_ms = start_ms + len(audio)
        if end_ms > total_ms:  # extend canvas instead of truncating end
            track = track + AudioSegment.silent(duration=end_ms - total_ms)
            total_ms = end_ms
        track = track.overlay(audio, position=start_ms)
        placed += 1
        if progress_cb:
            progress_cb((i + 1) / len(segments), f"Aligned {i + 1}/{len(segments)}")

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    normalize(track, headroom=1.0).export(str(out_wav), format="wav")
    return out_wav
