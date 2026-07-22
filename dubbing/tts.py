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
import time
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

EDGE_VOICE_GENDER = {
    "hi-IN-SwaraNeural": "F", "hi-IN-MadhurNeural": "M",
    "en-IN-NeerjaNeural": "F", "en-IN-PrabhatNeural": "M",
    "en-US-JennyNeural": "F", "en-US-GuyNeural": "M",
    "en-GB-SoniaNeural": "F",
}


def pick_edge_fallback(language: str, gender: str = "F") -> str:
    """Edge voice of the requested gender for fallback (keeps dub gender-consistent)."""
    for v in EDGE_VOICES[language]:
        if EDGE_VOICE_GENDER.get(v) == gender:
            return v
    return next(iter(EDGE_VOICES[language]))


def gemini_voice_gender(voice: str) -> str:
    return "F" if GEMINI_VOICES.get(voice, "").startswith("Female") else "M"


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
    exhausted_policy: str = "fallback",   # "wait" | "ask" | "fallback"
    interaction: dict | None = None,       # ask-mode hooks (set_ask, decision, ...)
    progress_cb=None,
) -> list[dict]:
    """Synthesize every segment's `translated` text. NO CONTENT IS EVER SKIPPED.

    exhausted_policy (only for engine="gemini", when all keys run out):
      * "wait"     – block until keys recover; every line keeps the Gemini voice.
      * "ask"      – wait too; if cumulative waiting exceeds threshold, pause and
                     ask the user (via interaction hooks) wait vs standard voice.
      * "fallback" – immediately use the gender-matched Edge backup voice.
    speaker_voices maps speaker labels to Edge voices — used when engine == "edge".
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ask_threshold_s = float((interaction or {}).get("threshold_s", 180))
    hard_cap_wait_s = 1500.0     # absolute safety: never wait more than 25 min total
    ask_timeout_s = 720.0        # user has 12 min to answer; default = standard voice

    gemini_backoff_until = 0.0
    policy = exhausted_policy
    total_wait_s = 0.0
    results = []
    for i, seg in enumerate(segments):
        text = str(seg.get("translated") or seg.get("text") or "").strip()
        wav_path = out_dir / f"seg_{i:04d}.wav"
        if not text:
            results.append({**seg, "wav": None, "engine": None})
            continue

        if engine == "edge" and speaker_voices and seg.get("speaker") in speaker_voices:
            seg_voice = speaker_voices[seg["speaker"]]
        else:
            seg_voice = voice

        used_engine = None
        if engine == "gemini" and time.time() >= gemini_backoff_until:
            while used_engine is None:
                try:
                    synth_gemini(text, seg_voice, rotator, wav_path)  # type: ignore[arg-type]
                    used_engine = "gemini"
                except NoKeysAvailable:
                    # every key cooling down
                    if policy == "fallback" or total_wait_s >= hard_cap_wait_s:
                        break
                    wait_s = int(max(rotator.soonest_ready_in() if rotator else 60, 15))  # type: ignore[union-attr]
                    total_wait_s += wait_s
                    if progress_cb:
                        progress_cb(min((i + 0.5) / len(segments), 0.99),
                                    f"⏳ All keys cooling — resuming in ~{wait_s}s "
                                    f"(no content skipped, total wait {int(total_wait_s)}s)")
                    if policy == "ask" and total_wait_s >= ask_threshold_s and interaction:
                        decision = interaction["decision"]()
                        if decision is None:
                            interaction["set_ask"](True)
                            t0 = time.time()
                            while decision is None and time.time() - t0 < ask_timeout_s:
                                time.sleep(2)
                                decision = interaction["decision"]()
                            interaction["set_ask"](False)
                        if decision != "wait":      # "standard" or timeout → standard voice
                            policy = "fallback"
                            total_wait_s = 0.0
                            break
                        total_wait_s = 0.0          # user chose to keep waiting → reset couner
                    time.sleep(min(wait_s, 60))
                except Exception:
                    # non-quota/non-transient (e.g. content filter): brief Gemini pause,
                    # backup for THIS segment, keep engine alive for the rest
                    gemini_backoff_until = time.time() + 60
                    break
        if used_engine is None:
            v = edge_fallback_voice if engine == "gemini" else seg_voice
            synth_edge(text, v, wav_path)
            used_engine = "edge"
        results.append({**seg, "wav": str(wav_path), "engine": used_engine})

        if progress_cb:
            progress_cb((i + 1) / len(segments), f"Voiced {i + 1}/{len(segments)}")
    return results
