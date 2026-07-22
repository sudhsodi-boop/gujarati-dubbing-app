# 🎬 Gujarati AI Dubbing App

Dub **Gujarati videos into natural English or Hindi** — with proper lip-sync — using
**only free tools**:

| Stage | Tool | Cost |
|---|---|---|
| Speech-to-text (Gujarati) | Gemini 2.5 Flash (your free AI Studio keys, auto-rotated) | Free |
| Translation | Gemini 2.5 Flash | Free |
| Voice (Hindi / English) | Gemini 2.5 TTS **or** Microsoft Edge neural voices (auto-fallback) | Free |
| Time alignment | ffmpeg `atempo` + pydub | Free |
| Lip-sync | MuseTalk on Google Colab (free T4 GPU) | Free |

```
Gujarati video ─► extract audio ─► transcribe (review!) ─► translate ─►
TTS per segment ─► time-stretch to original timings ─► dubbed video ─►
► MuseTalk (Colab) ─► lip-synced final video
```

---

## 1. Setup (one time, ~5 min)

**A. Install dependencies**
```bash
cd gujarati-dubbing-app
pip install -r requirements.txt
```

**B. Install ffmpeg** (needed for audio/video handling):
- Windows: `winget install ffmpeg`  (or https://ffmpeg.org/download.html)
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

**C. Get free Gemini API keys** — https://aistudio.google.com/app/apikey
Create a few (even from 2–3 Google accounts) and paste them in the app's sidebar,
or save them in a `.env`/environment variable:
```bash
# Windows PowerShell
setx GEMINI_API_KEYS "key1,key2,key3"
# macOS/Linux
export GEMINI_API_KEYS="key1,key2,key3"
```

## 2. Run the app

**Option A — run locally:**
```bash
streamlit run app.py
```
**Option B — host it FREE online** (no local Python/ffmpeg needed): see
**[DEPLOY_STREAMLIT_CLOUD.md](DEPLOY_STREAMLIT_CLOUD.md)** ☁️

Then in the browser:
1. **Upload** your Gujarati video (mp4/mov/mkv).
2. **Transcribe** — Gemini writes the Gujarati transcript. ✅ *Review & fix wrong words* (this is the #1 quality booster).
3. **Translate** to Hindi or English (editable).
4. **Voice & align** — pick a voice; the app generates speech that fits each original time slot.
5. **Export** — you get:
   - the **dubbed video** (audio replaced), and
   - a **`lipsync_package.zip`**.

> ⚠️ The Gemini **TTS** free tier has tighter daily limits than the text models.
> If it runs out mid-dub, the app automatically finishes with Edge neural voices,
> so your dub never fails. You can also select "Edge neural voices" from the start.
> Both are natural, non-robotic standard voices.

## 3. Add lip-sync (Colab notebook)

The Streamlit app makes the *voice* right; `lipsync_colab.ipynb` makes the *mouth* right.

1. In the project folder, open **`lipsync_colab.ipynb`** → upload to [Google Colab](https://colab.research.google.com).
2. Colab: **Runtime ▸ Change runtime type ▸ T4 GPU** (free).
3. Run the cells top-to-bottom. When asked, upload `lipsync_package.zip`.
4. The notebook downloads & runs **MuseTalk** and gives you `final.mp4` with mouth movement synced to the new audio.

## 4. Tips for best quality

- **Front-facing single speaker** works best for lip-sync (side profiles will fail — that's a limitation of all current open lip-sync models).
- **Clean audio** = better transcription. Background music confuses ASR.
- Keep the translated lines about as long as the originals (the translation prompt already asks for this; review before voicing).
- If the mouth looks offset, rerun the notebook with a small `bbox_shift` (+/-5).

## 5. Roadmap ideas (when free isn't enough)

- **Voice cloning** (dub in *your own* voice): ElevenLabs (paid) or XTTS-v2 (free, needs GPU).
- **Multi-speaker** videos: add pyannote diarization → voice per speaker.
- **Background music retention**: Demucs vocal separator before ASR, remix after.
- Fully local lip-sync: run MuseTalk directly in this app if you get a GPU machine.

## Files
```
app.py                  Streamlit UI (5-step wizard)
dubbing/                pipeline modules
  keys.py               API-key rotation (429-aware cooldowns)
  asr.py                silence-aware chunking + Gemini transcription
  translate.py          duration-aware translation
  tts.py                Gemini TTS + Edge neural voices (auto-fallback)
  align.py              ffmpeg atempo time-fitting + track assembly
  videoutils.py         ffmpeg wrappers, SRT writer
lipsync_colab.ipynb     MuseTalk lip-sync notebook (Colab T4)
make_notebook.py        regenerates the notebook
requirements.txt / .env.example
```
