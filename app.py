"""Gujarati → English/Hindi AI dubbing app (Streamlit).

Workflow:
  1. Upload a Gujarati video
  2. Gemini transcribes it (review & fix the transcript)
  3. Gemini translates it (review & edit)
  4. Natural TTS voices it, time-aligned to the original speech
  5. Download the audio-swapped video + a package for the lip-sync Colab notebook

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# make sure the script's own directory is importable (helps on cloud deploys)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import streamlit as st

from dubbing import align, asr, translate, tts, videoutils
from dubbing.keys import KeyRotator, load_keys_from_text

st.set_page_config(page_title="Gujarati AI Dubbing", page_icon="🎬", layout="wide")
st.title("🎬 Gujarati → English / Hindi AI Dubbing")
st.caption(
    "Gemini free-tier ASR + translation · natural TTS voices · time-aligned dubbing "
    "· lip-sync via the companion Colab notebook"
)


# ---------------------------------------------------------------- run folder
def get_run_dir() -> Path:
    if "run_dir" not in st.session_state:
        run_dir = Path("runs") / time.strftime("run_%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        st.session_state.run_dir = str(run_dir)
    return Path(st.session_state.run_dir)


def _default_keys() -> str:
    """Keys from env var, or Streamlit Secrets (for Community Cloud deploys)."""
    if os.environ.get("GEMINI_API_KEYS"):
        return os.environ["GEMINI_API_KEYS"]
    try:
        return st.secrets.get("GEMINI_API_KEYS", "")
    except Exception:
        return ""


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")

    default_keys = _default_keys()
    keys_text = st.text_area(
        "Gemini API keys (one per line or comma-separated)",
        value=default_keys,
        height=110,
        help="Create free keys at https://aistudio.google.com/app/apikey — "
        "paste several; the app rotates automatically when one hits its free limit.",
    )
    keys = load_keys_from_text(keys_text)
    st.caption(f"🔑 {len(keys)} key(s) loaded" if keys else "🔑 No keys — required for transcription & translation")

    target_language = st.radio("Dub into", ["Hindi", "English"], horizontal=True)

    engine = st.radio(
        "Voice engine",
        ["edge", "gemini"],
        format_func=lambda e: {
            "edge": "Edge neural voices (free, no key needed, very reliable)",
            "gemini": "Gemini 2.5 TTS (your keys; auto-falls back to Edge on quota)",
        }[e],
    )
    if engine == "gemini":
        voice = st.selectbox(
            "Voice",
            list(tts.GEMINI_VOICES),
            format_func=lambda v: f"{v} — {tts.GEMINI_VOICES[v]}",
        )
    else:
        eopts = tts.EDGE_VOICES[target_language]
        voice = st.selectbox(
            "Voice", list(eopts), format_func=lambda v: f"{eopts[v]}"
        )
    edge_fallback = list(tts.EDGE_VOICES[target_language])[0]

    with st.expander("Advanced"):
        source_language = st.text_input("Source language", "Gujarati")
        asr_model = st.text_input("ASR model", asr.ASR_MODEL)
        translate_model = st.text_input("Translation model", translate.TRANSLATE_MODEL)
        tts_model = st.text_input("Gemini TTS model", tts.TTS_MODEL)
        max_chunk = st.slider("Max ASR chunk (seconds)", 45, 120, 75)
        show_cols = st.checkbox("Show per-segment audio player", value=False)


# --- friendly key onboarding: paste keys right in the main window ---
if not keys:
    st.warning("🔑 **One step before we start** — add your free Gemini API key(s) below.")
    pasted = st.text_area(
        "Paste API key(s) here — one per line",
        placeholder="AIzaSy....",
        height=90,
        key="main_key_input",
    )
    if pasted.strip():
        keys.extend(load_keys_from_text(pasted))
        st.success(f"✅ {len(keys)} key(s) loaded — you're ready!")
    st.markdown(
        """
**How to get a free key (1 minute):**
1. Open 👉 [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account → **Create API key** → copy it
3. Paste it in the box above (optional: create keys in new Google projects for more free quota — the app rotates them automatically)

🔒 *Keys live only in your browser session — nothing is saved on any server.*
"""
    )
    st.stop()
else:
    st.success(f"🔑 {len(keys)} API key(s) active")


def rotator_or_stop():
    if not keys:
        st.error("Add at least one Gemini API key in the sidebar (aistudio.google.com → Get API key).")
        st.stop()
    return KeyRotator(keys)


# ------------------------------------------------------------------ step 1
st.header("1️⃣ Upload Gujarati video")
uploaded = st.file_uploader("Video file", type=["mp4", "mov", "mkv", "webm", "m4v"])
if uploaded and st.session_state.get("video_name") != uploaded.name:
    run_dir = get_run_dir()
    video_path = run_dir / f"original{Path(uploaded.name).suffix}"
    video_path.write_bytes(uploaded.getbuffer())
    st.session_state.update(
        video_name=uploaded.name, video_path=str(video_path),
        segments=None, voiced=None, track_wav=None, dubbed_video=None,
        audio_wav=None, duration=None,
    )

if st.session_state.get("video_path"):
    video_path = Path(st.session_state.video_path)
    st.video(str(video_path))
    try:
        videoutils.check_ffmpeg()
        if not st.session_state.get("audio_wav"):
            with st.spinner("Extracting audio…"):
                wav = videoutils.extract_audio(video_path, get_run_dir() / "source.wav")
                st.session_state.audio_wav = str(wav)
                st.session_state.duration = videoutils.probe_duration_s(video_path)
        dur = st.session_state.duration
        st.info(f"Duration: **{dur:.1f}s** ({dur/60:.1f} min)")
    except RuntimeError as exc:
        st.error(str(exc))
        st.code(videoutils.ffmpeg_python_hint())
        st.stop()

# ------------------------------------------------------------------ step 2
if st.session_state.get("audio_wav"):
    st.header("2️⃣ Transcribe (review & fix)")
    col_a, col_b = st.columns([1, 4])
    if col_a.button("🗣️ Transcribe with Gemini", type="primary"):
        rot = rotator_or_stop()
        bar = st.progress(0.0, text="Starting…")
        def cb(p, msg): bar.progress(p, text=msg)
        try:
            segs = asr.transcribe_full(
                st.session_state.audio_wav, get_run_dir(), rot,
                model=asr_model, source_language=source_language,
                max_chunk_s=max_chunk, progress_cb=cb,
            )
            st.session_state.segments = segs
            st.session_state.voiced = None
            bar.progress(1.0, text=f"Done — {len(segs)} segments")
        except Exception as exc:
            st.exception(exc)

    segs = st.session_state.get("segments")
    if segs:
        st.write("Review the Gujarati transcript — fixing misheard words here improves the whole dub:")
        df = pd.DataFrame(segs)[["start", "end", "text"]]
        edited = st.data_editor(
            df, num_rows="dynamic", use_container_width=True,
            column_config={
                "start": st.column_config.NumberColumn("Start (s)", format="%.2f", width="small"),
                "end": st.column_config.NumberColumn("End (s)", format="%.2f", width="small"),
                "text": st.column_config.TextColumn(f"{source_language} text", width="large"),
            },
            key="transcript_editor",
        )
        st.session_state.segments = edited.to_dict("records")

# ------------------------------------------------------------------ step 3
if st.session_state.get("segments"):
    st.header("3️⃣ Translate")
    col_a, col_b = st.columns([1, 4])
    if col_a.button(f"🌍 Translate to {target_language}", type="primary"):
        rot = rotator_or_stop()
        bar = st.progress(0.0, text="Translating…")
        try:
            out = translate.translate_segments(
                st.session_state.segments, target_language, rot,
                model=translate_model, source_language=source_language,
                progress_cb=lambda p, m: bar.progress(p, text=m),
            )
            st.session_state.segments = out
            st.session_state.voiced = None
            bar.progress(1.0, text="Translation done")
        except Exception as exc:
            st.exception(exc)

    if any(s.get("translated") for s in st.session_state.segments):
        df = pd.DataFrame(st.session_state.segments)[["text", "translated"]]
        edited = st.data_editor(
            df, use_container_width=True,
            column_config={
                "text": st.column_config.TextColumn(f"Original ({source_language})", width="medium", disabled=True),
                "translated": st.column_config.TextColumn(f"Translation ({target_language}) — editable", width="large"),
            },
            key="translation_editor",
        )
        merged = []
        for orig, row in zip(st.session_state.segments, edited.to_dict("records")):
            orig["translated"] = row["translated"]
            merged.append(orig)
        st.session_state.segments = merged

# ------------------------------------------------------------------ step 4
if any(s.get("translated") for s in st.session_state.get("segments") or []):
    st.header("4️⃣ Generate dubbed audio")
    if st.button("🎙️ Voice & align all segments", type="primary"):
        run_dir = get_run_dir()
        rot = KeyRotator(keys) if keys else None
        bar = st.progress(0.0, text="Voicing…")
        voiced = tts.synthesize_track(
            st.session_state.segments,
            run_dir / "voice_segments",
            engine=engine, voice=voice, rotator=rot,
            edge_fallback_voice=edge_fallback,
            progress_cb=lambda p, m: bar.progress(p, text=m),
        )
        st.session_state.voiced = voiced
        bar.progress(0.99, text="Assembling final track…")
        track = align.build_track(
            voiced, st.session_state.duration, run_dir / "dubbed.wav",
            run_dir / "stretch_tmp",
        )
        st.session_state.track_wav = str(track)
        st.session_state.dubbed_video = None
        bar.progress(1.0, text="Dubbed audio ready 🎉")

    if st.session_state.get("voiced"):
        voiced = st.session_state.voiced
        used_gemini = sum(1 for s in voiced if s.get("engine") == "gemini")
        used_edge = sum(1 for s in voiced if s.get("engine") == "edge")
        st.caption(f"Voices used — Gemini: {used_gemini}, Edge: {used_edge}")
        if show_cols:
            for s in voiced:
                if s.get("wav"):
                    st.write(f"[{s['start']:.1f}s] {s.get('translated','')}")
                    st.audio(s["wav"])

    if st.session_state.get("track_wav"):
        st.subheader("Preview final audio track")
        st.audio(st.session_state.track_wav)

# ------------------------------------------------------------------ step 5
if st.session_state.get("track_wav"):
    st.header("5️⃣ Export & lip-sync")
    if st.button("▶️ Create dubbed video + lip-sync package", type="primary"):
        run_dir = get_run_dir()
        with st.spinner("Muxing video…"):
            dubbed = videoutils.mux_audio_video(
                st.session_state.video_path,
                st.session_state.track_wav,
                run_dir / f"dubbed_{target_language.lower()}.mp4",
            )
        videoutils.write_srt(st.session_state.segments, run_dir / "original.srt")
        videoutils.write_srt(st.session_state.segments, run_dir / "dub.srt", text_key="translated")

        # package everything the Colab notebook needs
        pkg_dir = run_dir / "lipsync_package"
        pkg_dir.mkdir(exist_ok=True)
        shutil.copy(st.session_state.video_path, pkg_dir / "original.mp4")
        shutil.copy(st.session_state.track_wav, pkg_dir / "dub.wav")
        (pkg_dir / "INSTRUCTIONS.txt").write_text(
            "Lip-sync package generated by the Gujarati Dubbing app.\n\n"
            "1. Open lipsync_colab.ipynb in Google Colab (Runtime → GPU T4 or better).\n"
            "2. Upload this zip (or the two media files) when the notebook asks.\n"
            "3. Run all cells → download the final lip-synced mp4.\n",
            encoding="utf-8",
        )
        zip_path = run_dir / "lipsync_package.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in pkg_dir.iterdir():
                zf.write(f, f.name)

        st.session_state.dubbed_video = str(dubbed)
        st.session_state.zip_path = str(zip_path)

    if st.session_state.get("dubbed_video"):
        st.success("Ready! Two outputs:")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Dubbed video (audio swapped)")
            st.video(st.session_state.dubbed_video)
            st.download_button(
                "⬇️ Download dubbed video",
                Path(st.session_state.dubbed_video).read_bytes(),
                file_name=f"dubbed_{target_language.lower()}.mp4",
                mime="video/mp4",
            )
        with c2:
            st.subheader("For lip-sync (Colab)")
            st.download_button(
                "⬇️ Download lip-sync package (.zip)",
                Path(st.session_state.zip_path).read_bytes(),
                file_name="lipsync_package.zip",
                mime="application/zip",
            )
            st.markdown(
                "Lips don't move yet in the left video. To fix: open "
                "**lipsync_colab.ipynb** (in this project folder) in Google Colab "
                "with a free T4 GPU, upload this zip, and run all cells — "
                "MuseTalk will re-animate the mouth to match the new audio."
            )
