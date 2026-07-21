"""Small ffmpeg / ffprobe wrappers used by the pipeline."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError(
            "ffmpeg/ffprobe not found. Install it and make sure it's on PATH: "
            "https://ffmpeg.org/download.html  (Windows: `winget install ffmpeg`, "
            "Mac: `brew install ffmpeg`, Ubuntu: `sudo apt install ffmpeg`)"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n" + " ".join(cmd) + "\n\n" + proc.stderr[-2000:]
        )
    return proc


def probe_duration_s(path: str | Path) -> float:
    out = _run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ]
    ).stdout.strip()
    return float(out)


def extract_audio(video_path: str | Path, wav_path: str | Path, sr: int = 16000) -> Path:
    """Mono 16 kHz WAV – ideal for ASR."""
    _run(
        [
            "ffmpeg", "-y", "-i", str(video_path), "-vn",
            "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(wav_path),
        ]
    )
    return Path(wav_path)


def mux_audio_video(
    video_path: str | Path, audio_wav: str | Path, out_mp4: str | Path
) -> Path:
    """Swap the audio track of `video_path` for `audio_wav` (video is copied)."""
    _run(
        [
            "ffmpeg", "-y",
            "-i", str(video_path), "-i", str(audio_wav),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(out_mp4),
        ]
    )
    return Path(out_mp4)


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segments: list[dict], path: str | Path, text_key: str = "text") -> Path:
    lines = []
    for i, seg in enumerate(segments, 1):
        if not seg.get(text_key):
            continue
        lines.append(str(i))
        lines.append(
            f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}"
        )
        lines.append(str(seg[text_key]).strip())
        lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return Path(path)


def ffmpeg_python_hint() -> str:
    return (
        "ffmpeg is required on the machine running this app.\n"
        "Windows: winget install ffmpeg\n"
        "macOS:   brew install ffmpeg\n"
        "Linux:   sudo apt install ffmpeg"
    )


if __name__ == "__main__":
    check_ffmpeg()
    print("ffmpeg OK", file=sys.stderr)
