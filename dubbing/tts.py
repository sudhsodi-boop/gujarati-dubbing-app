"""Stage 4 – text-to-speech.

Two engines, both producing natural (non-robotic) standard voices:
  * "gemini" – Google Gemini 2.5 Flash TTS via your free-tier AI Studio keys
    (model: gemini-2.5-flash-preview-tts; swap to gemini-2.5-flash-tts for GA).
  * "edge"   – Microsoft Edge neural voices via edge-tts (free, no key).

If the Gemini TTS free quota is exhausted mid-run we automatically fall back
to Edge voices, so a dub never dies half-way.
"""

from __future__ import annotations

import asyncio
import wave
from pathlib import Path

from pydub import AudioSegment

from .keys import KeyRotator, NoKeysAvailable

TTS_MODEL = "gemini-2.5-flash-preview-tts"  # change to "gemini-2.5-flash-tts" (GA) if you like
TTS_SAMPLE_RATE = 24_000

# --- Gemini prebuilt voices (subset — full list in Google AI Studio docs) ---
GEMINI_VOICES = {
    "Kore": "Female — firm, clear",
    "Aoede": "Female — breezy, warm",
    "Callirrhoe": "Female — easy-going",
    "Leda": "Female — youthful",
    "Charon": "Male — informative, steady",
    "Fenrir": "Male — excitable, energetic",
    "Puck": "Male — upbeat",
    "Orus": "Male — firm",
}

EDGE_VOICES = {
    "Hindi": {
        "hi-IN-SwaraNeural": "Swara — Hindi female",
        "hi-IN-MadhurNeural": "Madhur — Hindi male",
    },
    "English": {
        "en-IN-NeerjaNeural": "Neerja — Indian English female",
        "en-IN-PrabhatNeural": "Prabhat — Indian English male",
        "en-US-JennyNeural": "Jenny — US English female",
        "en-US-GuyNeural": "Guy — US English male",
        "en-GB-SoniaNeural": "Sonia — UK English female",
    },
}


# ------------------------------------------------------------------ gemini tts
def synth_gemini(
    text: str,
    voice: str,
    rotator: KeyRotator,
    out_wav: str | Path,
    model: str = TTS_MODEL,
    style_hint: str = "Speak at a natural, clear, conversational pace.",
) -> Path:
    from google import genai
    from google.genai import types

    def _call(api_key: str):
        client = genai.Client(api_key=api_key)
        resp = client.models.generate_content(
            model=model,
            contents=f"{style_hint}\n{text}",
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice
                        )
                    )
                ),
            ),
        )
        data = resp.candidates[0].content.parts[0].inline_data.data
        if not data:
            raise RuntimeError("Gemini TTS returned empty audio.")
        return data

    pcm = rotator.execute(_call)
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    # Gemini TTS streams raw 16-bit mono PCM @ 24 kHz → wrap into WAV
    with wave.open(str(out_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(pcm)
    return out_wav


# -------------------------------------------------------------------- edge tts
def synth_edge(
    text: str,
    voice: str,
    out_wav: str | Path,
    rate_pct: int = 0,
) -> Path:
    import edge_tts

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = out_wav.with_suffix(".tmp.mp3")
    rate = f"{rate_pct:+d}%"

    async def _go():
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicate.save(str(tmp_mp3))

    asyncio.run(_go())
    seg = AudioSegment.from_mp3(str(tmp_mp3))
    seg = seg.set_frame_rate(TTS_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    seg.export(str(out_wav), format="wav")
    tmp_mp3.unlink(missing_ok=True)
    return out_wav


# ------------------------------------------------------------------- per segment
def synthesize_track(
    segments: list[dict],
    out_dir: str | Path,
    engine: str = "edge",
    voice: str = "hi-IN-SwaraNeural",
    rotator: KeyRotator | None = None,
    edge_fallback_voice: str | None = None,
    speaker_voices: dict[str, str] | None = None,  # {speaker: edge voice}
    auto_fallback_to_edge: bool = True,
    progress_cb=None,
) -> list[dict]:
    """Synthesize every segment's `translated` text. Returns segments with 'wav'.

    speaker_voices maps speaker labels (e.g. {"Pujyashree": ..., "Questioner": ...})
    to Edge voices — used when engine == "edge" for two-voice dubbing.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gemini_dead = False
    results = []
    for i, seg in enumerate(segments):
        text = str(seg.get("translated") or seg.get("text") or "").strip()
        wav_path = out_dir / f"seg_{i:04d}.wav"
        if not text:
            results.append({**seg, "wav": None, "engine": None})
            continue

        # voice for this segment
        if engine == "edge" and speaker_voices and seg.get("speaker") in speaker_voices:
            seg_voice = speaker_voices[seg["speaker"]]
        else:
            seg_voice = voice

        done = False
        if engine == "gemini" and not gemini_dead:
            try:
                synth_gemini(text, seg_voice, rotator, wav_path)  # type: ignore[arg-type]
                done = True
                results.append({**seg, "wav": str(wav_path), "engine": "gemini"})
            except NoKeysAvailable:
                gemini_dead = True  # quota gone on every key → use edge below
            except Exception:
                gemini_dead = True  # unexpected TTS error → fall back, don't die
        if not done:
            # gemini-fallback uses the dedicated fallback voice; edge uses the user pick
            v = edge_fallback_voice if engine == "gemini" else seg_voice
            synth_edge(text, v, wav_path)
            results.append({**seg, "wav": str(wav_path), "engine": "edge"})

        if progress_cb:
            progress_cb((i + 1) / len(segments), f"Voiced {i + 1}/{len(segments)}")
    return results
