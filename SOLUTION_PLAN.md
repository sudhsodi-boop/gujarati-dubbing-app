# AI Dubbing App: Gujarati → English / Hindi (with Lip-Sync)
**Solution Architecture & Options — July 2026**

---

## 1. How AI Dubbing Works (The 5-Stage Pipeline)

Every dubbing solution — hosted or self-built — follows this pipeline:

```
Gujarati Video
   │
   ├─► Stage 1: Audio Extraction (ffmpeg)
   │
   ├─► Stage 2: ASR — Speech-to-Text with WORD-LEVEL TIMESTAMPS
   │        Whisper large-v3 / IndicWhisper (AI4Bharat)
   │
   ├─► Stage 3: Translation (Gujarati → English / Hindi)
   │        GPT-4-class LLM / IndicTrans2 / NLLB
   │
   ├─► Stage 4: TTS with Voice Cloning + TIME ALIGNMENT
   │        ElevenLabs / XTTS-v2  →  stretch audio to fit each original segment
   │
   └─► Stage 5: Lip-Sync video to new audio
            MuseTalk / LatentSync / Wav2Lip (open source)
            OR Sync Labs / HeyGen (API)
```

**The two "make or break" details:**
- **Word-level timestamps** from ASR — without them, you can't alignTranslated speech to the video.
- **Time-stretching** — translated sentences are longer/shorter than the original; the TTS audio must be retimed to fit each segment before lip-sync.

---

## 2. The Three Possible Solutions

### Option A — Ready-Made Platform (No code, fastest)
Upload your Gujarati video → pick English/Hindi → download dubbed video.

| Platform | Price | Notes |
|---|---|---|
| **HeyGen** | ~$24/mo | Best lip-sync, 175+ languages (Gujarati ✓), voice cloning |
| **Dubverse** | ~$18/mo | Indian company, strong Hindi/Indian-language support |
| **Rask AI** | ~$50/mo | 130+ languages, multi-speaker, lip-sync on higher tier |
| **ElevenLabs Dubbing** | ~$5–6/mo | Best voices, but mostly audio-only (weak lip-sync) |

✅ Best quality-to-effort ratio | ❌ Subscription cost, no customization, not "your app"

---

### Option B — Build YOUR OWN App using APIs (Recommended hybrid)
Your own web app / UI; backend calls paid APIs per video.

| Stage | API Choice | Cost (approx) |
|---|---|---|
| ASR (Gujarati) | OpenAI Whisper API / Groq Whisper | ~$0.006 / min |
| Translation | GPT-4o-mini / Gemini Flash | ~$0.01 / min |
| TTS + Voice Clone | **ElevenLabs** (Hindi ✓, English ✓) | ~$0.10–0.18 / min |
| Lip-Sync | **Sync Labs API** (~$0.08/sec) or HeyGen API | ~$4.80 / min of video |

💡 Lip-sync API is the biggest cost. Alternative: run open-source lip-sync (MuseTalk/Wav2Lip) on a rented GPU (~$0.50/hr on RunPod/Vast.ai) → ~$0.05–0.10/min instead.

✅ It's YOUR product/brand | ✅ Pay only per use | ❌ Some dev work (~1–2 days MVP)

---

### Option C — Fully Open-Source / Self-Hosted (Free per video)
Everything runs on your GPU (or free Colab).

| Stage | Open-Source Model | Notes |
|---|---|---|
| ASR | **Whisper large-v3** or **IndicWhisper** (AI4Bharat) | IndicWhisper is best-in-class for Gujarati (~13% WER vs ~18–21% vanilla Whisper) |
| Translation | **IndicTrans2** (AI4Bharat) or LLM | Purpose-built for Indian languages |
| TTS + Voice Clone | **XTTS-v2** (Coqui) — Hindi ✓ English ✓ | Clones your voice from ~6 sec of the original audio |
| Lip-Sync | **MuseTalk** (balanced) / **LatentSync** (HD) / **Wav2Lip** (lightweight) / **InfiniteTalk** (long videos) | MuseTalk = recommended default |
| Orchestration | Python + ffmpeg | Streamlit/Gradio UI on top |

✅ Zero per-video cost, full privacy | ❌ Needs GPU (16GB+ VRAM ideal) or Colab Pro, more setup work

---

## 3. Gujarati-Specific Considerations

1. **Gujarati ASR is the weakest link** (WER ~13–20%). Add an optional *transcript review/edit screen* in the app before translation — fixing 2–3 misheard words dramatically improves the final dub.
2. **Target languages are well supported**: Hindi + English have excellent TTS/voice-cloning (ElevenLabs, XTTS-v2). Gujarati output is NOT needed, so we're safe.
3. **Lip-sync models are language-agnostic** — they map phonemes→mouth shapes, so Gujarati input is no problem.
4. **Best results**: front-facing single speaker, clear audio, minimal background music. Music/noise needs a source-separation step (Demucs) so the background can be kept.

---

## 4. Recommended Path

**MVP in a day:** Option A (HeyGen/Dubverse) for a quick proof-of-concept.
**Real product:** Option B → C hybrid:
- APIs for ASR + Translation + TTS (cheap, high quality)
- Open-source lip-sync on rented GPU (cuts the biggest cost by 10–50x)

## 5. Rough Cost per 10-Minute Video (Option B)
- Whisper ASR: ~$0.06
- Translation: ~$0.10
- ElevenLabs TTS: ~$1.50
- Lip-sync: ~$48 (Sync Labs API) **or** ~$1 (self-hosted MuseTalk)
- **Total: ~$2.50–$50** depending on lip-sync choice

---

## 6. Confirmed Decisions (2026-07-21) — what we're actually building
- **App form**: Streamlit web app (upload → review transcript → dub → export).
- **AI engine**: User's own **Gemini free-tier API keys with automatic rotation** (429-aware cooldowns) for ASR + translation.
- **Voice**: Standard, natural (non-robotic) voices — Gemini 2.5 TTS + Edge neural voices auto-fallback (user chose "standard voice, not robotic"). Voice cloning moved to roadmap.
- **Compute**: No guaranteed GPU → app does stages 1–4 locally; **lip-sync on Google Colab free T4** via `lipsync_colab.ipynb` (MuseTalk).
- **Total cost per video: $0** (all free tier) — see README.md.
