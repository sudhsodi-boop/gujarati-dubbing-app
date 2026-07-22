"""Proofreading helpers: make sure NOTHING is skipped.

1. speech_gaps()  – finds stretches of real speech in the audio that are NOT
   covered by any transcript segment (missed transcription).
2. translation_report() – finds segments with missing/possibly-untranslated text.
"""

from __future__ import annotations

from pathlib import Path

from pydub import AudioSegment, silence


# --------------------------------------------------------------- audio coverage
def speech_gaps(
    audio: str | Path | AudioSegment,
    segments: list[dict],
    min_gap_s: float = 3.0,
    min_silence_ms: int = 900,
    ignore_ranges: list[dict] | None = None,
) -> list[tuple[float, float]]:
    """Return [(start_s, end_s)] of spoken audio not covered by any segment.

    ignore_ranges (e.g. skipped song sections) count as "covered" so they
    are not reported as missed speech.
    """
    if isinstance(audio, (str, Path)):
        audio = AudioSegment.from_wav(str(audio))
    dbfs = audio.dBFS
    thresh = (dbfs - 16) if dbfs != float("-inf") else -45
    nonsilent = silence.detect_nonsilent(
        audio, min_silence_len=min_silence_ms, silence_thresh=thresh
    )
    seg_ints = sorted(
        (int(s["start"] * 1000), int(s["end"] * 1000))
        for s in segments
        if s.get("end", 0) > s.get("start", 0)
    )
    for r in ignore_ranges or []:
        if r.get("end", 0) > r.get("start", 0):
            seg_ints.append((int(r["start"] * 1000), int(r["end"] * 1000)))
    seg_ints.sort()
    gaps = []
    for a, b in nonsilent:
        cur = a
        for s, e in seg_ints:
            if e <= cur or s >= b:
                continue
            if s > cur:
                gaps.append((cur, min(s, b)))
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
    return [(g / 1000, h / 1000) for g, h in gaps if (h - g) >= min_gap_s * 1000]


def coverage_pct(segments: list[dict], total_s: float) -> float:
    if not total_s:
        return 0.0
    covered = sum(s["end"] - s["start"] for s in segments if s.get("end", 0) > s.get("start", 0))
    return round(100.0 * covered / total_s, 1)


# ------------------------------------------------------------ translation audit
def translation_report(segments: list[dict]) -> dict:
    """missing = no translation at all; identical = translated == source (likely skipped)."""
    missing, identical, short = [], [], []
    for i, s in enumerate(segments):
        t = str(s.get("translated") or "").strip()
        src = str(s.get("text") or "").strip()
        if not t:
            missing.append(i)
        elif t == src:
            identical.append(i)
        elif src and len(t) < 0.25 * len(src):
            short.append(i)  # suspiciously short — might be truncated
    return {
        "total": len(segments),
        "translated": len(segments) - len(missing),
        "missing": missing,
        "identical": identical,
        "short": short,
    }
