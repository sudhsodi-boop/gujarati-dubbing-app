"""Gujarati AI Dubbing Studio (Streamlit).

Inputs:  Video | Audio | Imported transcript (Excel/CSV)
Stages:  transcribe -> proofread -> translate -> voice -> export
Long steps run as BACKGROUND JOBS (safe to switch tabs); runs are persisted
on disk in ./runs so you can resume and download later.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # cloud-import safety

import pandas as pd
import streamlit as st

from dubbing import align, asr, io_import, qa, store, translate, tts, videoutils
from dubbing.keys import KeyRotator, load_keys_from_text, AllKeysQuotaExhausted

st.set_page_config(page_title="Gujarati AI Dubbing Studio", page_icon="🎬", layout="wide")
st.title("🎬 Gujarati AI Dubbing Studio")
ss = st.session_state


# ------------------------------------------------------------------- helpers
def _default_keys() -> str:
    if os.environ.get("GEMINI_API_KEYS"):
        return os.environ["GEMINI_API_KEYS"]
    try:
        return st.secrets.get("GEMINI_API_KEYS", "")
    except Exception:
        return ""


def _safe_editor(df, **kwargs):
    try:
        return st.data_editor(df, width="stretch", **kwargs)
    except TypeError:
        return st.data_editor(df, use_container_width=True, **kwargs)


def fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def get_run_dir() -> Path:
    if "run_id" not in ss:
        ss.run_id = store.new_run()
    return store.run_dir(ss.run_id)


def reset_stage_outputs(keep_transcript=False):
    for k in ("track_wav", "dubbed_video", "subbed_video", "zip_path",
              "upload_sig", "import_video_sig", "audio_input"):
        ss.pop(k, None)
    if not keep_transcript:
        for k in ("segments", "speakers", "imported"):
            ss.pop(k, None)


def merge_editor_rows(records, cols):
    """Merge data_editor output into segments IN PLACE (never loses other fields)."""
    segs = list(ss.segments)
    out = []
    for i, rec in enumerate(records):
        row = dict(segs[i]) if i < len(segs) else {}
        for k in cols:
            if k in rec and rec[k] is not None:
                row[k] = rec[k]
        out.append(row)
    ss.segments = out
    store.save_segments(ss.run_id, out)


def poll_job(kind: str, label: str):
    """Block the page while a background job runs (it continues server-side if you leave)."""
    ph = st.empty()
    while True:
        s = store.status(ss.run_id, kind)
        running = (s.get("status") == "running") or (
            not s and store.job_running(ss.run_id, kind)
        )
        if running:
            ph.progress(min(float(s.get("pct", 0.02) or 0.02), 1.0),
                        text=f"{label}: {s.get('msg', 'working…')}")
            time.sleep(1.2)
            continue
        break
    if s.get("status") == "error":
        ph.empty()
        st.error(f"❌ {label} failed: {s.get('msg')}")
        with st.expander("Error details"):
            st.code(s.get("tb", ""))
        return None
    ph.progress(1.0, text=f"{label}: ✅ done")
    return store.job_result(ss.run_id, kind) or {}


def refresh_from_store():
    """Pull segments/artifacts of the current run from disk into session state."""
    segs = store.load_segments(ss.run_id)
    if segs:
        ss.segments = segs
        speakers = []
        for s in segs:
            sp = (s.get("speaker") or "").strip()
            if sp and sp not in speakers:
                speakers.append(sp)
        ss.speakers = speakers
    for key, pat in (("track_wav", "dubbed.wav"),):
        art = store.artifact(ss.run_id, pat)
        if art:
            ss[key] = art
    ss.songs = store.load_json(ss.run_id, "songs") or []
    orig = sorted(Path(get_run_dir()).glob("original.*"))
    if orig:
        ss.video_path = str(orig[0])


# ---------------------------------------------------------------- run bootstrap
run_dir = get_run_dir()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    keys_text = st.text_area(
        "Gemini API keys (one per line or comma-separated)",
        value=_default_keys(), height=100,
        help="Free keys: https://aistudio.google.com/app/apikey — add several; they rotate automatically.",
    )
    keys = load_keys_from_text(keys_text)
    st.caption(f"🔑 {len(keys)} key(s) loaded" if keys else "🔑 No keys yet")
    if keys:
        st.caption("⚡ Quota multiplies only if keys come from **different Google projects** — in AI Studio use “Create API key in new project”.")

    mode = st.radio("Input type", ["🎬 Video", "🎵 Audio", "📄 Import transcript (Excel/CSV)"], key="mode")

    target_language = st.radio("Dub into", ["Hindi", "English"], horizontal=True)
    engine = st.radio(
        "Voice engine", ["edge", "gemini"],
        format_func=lambda e: {
            "edge": "Edge neural voices (free, reliable)",
            "gemini": "Gemini 2.5 TTS (uses your keys; falls back to Edge)",
        }[e],
    )

    exhausted_policy = "fallback"
    if engine == "gemini":
        exhausted_policy = st.radio(
            "If ALL keys run out mid-dub",
            ["wait", "ask", "fallback"],
            index=0,
            format_func=lambda p: {
                "wait": "⏳ Wait for keys to recover — never change voice (recommended)",
                "ask": "❓ Wait, but ask me if it takes >3 min (standard voice?)",
                "fallback": "⚡ Switch to gender-matched standard voice immediately",
            }[p],
        )
        st.caption("Dubbing never skips a line — it only decides WHO voices it while keys cool.")

    pacing = st.radio(
        "Speech pacing",
        ["Continuous speech (no pauses)", "Strict lip-sync (exact timings)"],
        index=0,
    )
    pacing_mode = "flow" if pacing.startswith("Continuous") else "strict"
    if pacing_mode == "strict":
        st.caption("Strict: every line locked to its exact original time slot — best for the Colab lip-sync step, but long translations may speed up and leave gaps.")
    else:
        st.caption("Continuous: lines flow back-to-back like the original — artificial pauses capped at ~1.2s, gentle even pacing.")

    skip_songs = st.checkbox(
        "🎵 Skip songs & music (bhajan/kirtan)",
        value=True,
        help="Songs are NOT transcribed, translated or voiced — the ORIGINAL song audio is kept as-is in the final dub, so there is no dead air.",
    )
    if skip_songs:
        st.caption("Songs are auto-detected during transcription and kept in the original audio.")

    speakers = ss.get("speakers") or []
    speaker_voices: dict[str, str] = {}
    dual_voices = False
    voice = None
    edge_opts = tts.EDGE_VOICES[target_language]
    edge_voice_list = list(edge_opts)

    if engine == "edge" and len(speakers) >= 2:
        dual_voices = st.checkbox("🎭 Use a different voice per speaker", value=True)
    if dual_voices:
        for idx, sp in enumerate(speakers):
            v = st.selectbox(
                f"Voice — {sp}", edge_voice_list,
                index=min(idx, len(edge_voice_list) - 1),
                format_func=lambda x: f"{edge_opts[x]}",
                key=f"voice_{sp}_{target_language}",
            )
            speaker_voices[sp] = v
        voice = speaker_voices.get(speakers[0], edge_voice_list[0])
    elif engine == "gemini":
        voice = st.selectbox(
            "Voice", list(tts.GEMINI_VOICES),
            format_func=lambda v: f"{v} — {tts.GEMINI_VOICES[v]}",
            key=f"voice_gemini_{target_language}",
        )
        if len(speakers) >= 2:
            st.caption("🎭 Multi-speaker dual voices use the Edge engine.")
    else:
        voice = st.selectbox(
            "Voice", edge_voice_list,
            format_func=lambda x: f"{edge_opts[x]}",
            key=f"voice_edge_{target_language}",
        )
    # gender-matched fallback: if Gemini quota dies mid-dub, the Edge voice
    # that takes over must match the gender of the voice you picked
    if engine == "gemini" and voice:
        gender = "F" if tts.GEMINI_VOICES.get(voice, "").startswith("Female") else "M"
        edge_fallback = tts.pick_edge_fallback(target_language, gender)
    else:
        edge_fallback = edge_voice_list[0]

    with st.expander("Advanced"):
        source_language = st.text_input("Source language", "Gujarati")
        asr_model = st.text_input("ASR model", asr.ASR_MODEL)
        translate_model = st.text_input("Translation model", translate.TRANSLATE_MODEL)
        tts_model = st.text_input("Gemini TTS model", tts.TTS_MODEL)
        max_chunk = st.slider("Max ASR chunk (seconds)", 45, 120, 75)

    st.divider()
    with st.expander("🗂 History / resume a run"):
        if st.button("🆕 Start new project"):
            reset_stage_outputs()
            ss.run_id = store.new_run()
            st.rerun()
        runs = store.list_runs()
        if not runs:
            st.caption("No saved runs yet.")
        for r in runs[:12]:
            rid = r["id"]
            c1, c2, c3 = st.columns([3, 1, 1])
            c1.caption(f"**{r.get('label', rid)}**\n{rid} · {r.get('mode','?')} → {r.get('target','?')}")
            if rid != ss.run_id and c2.button("Load", key=f"load_{rid}"):
                reset_stage_outputs()
                ss.run_id = rid
                refresh_from_store()
                st.rerun()
            if c3.button("🗑", key=f"del_{rid}"):
                store.delete_run(rid)
                if rid == ss.run_id:
                    reset_stage_outputs()
                    ss.run_id = store.new_run()
                st.rerun()

# --- key onboarding (main area) ---
if not keys:
    pasted = st.text_area("Paste API key(s) here — one per line", placeholder="AIzaSy....",
                          height=90, key="main_key_input")
    if pasted.strip():
        keys.extend(load_keys_from_text(pasted))
if not keys:
    st.warning("🔑 **One step before we start** — add your free Gemini API key(s) above.")
    st.markdown(
        """
**How to get a free key (1 minute):**
1. Open 👉 [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in → **Create API key** → copy it
3. Paste above (more keys = more free daily quota — they rotate automatically)

🔒 *Keys live only in your browser session — nothing is saved on any server.*
"""
    )
    st.stop()

st.success(f"🔑 {len(keys)} API key(s) active")
st.info(f"📁 Current project: **{ss.run_id}** — progress is saved automatically; "
        f"you can switch tabs or leave and come back via 🗂 History.", icon="💾")

# ------------------------------------------------- show any running job live
for _kind, _label in (("transcribe", "Transcription"), ("translate", "Translation"),
                      ("voice", "Voicing"), ("subtitle", "Subtitles")):
    if store.job_running(ss.run_id, _kind):
        # Check for transcription quota wait prompt
        _quota_ask = (_kind == "transcribe"
                      and store.get_flag(ss.run_id, "ask_quota_wait")
                      and not store.get_flag(ss.run_id, "quota_decision"))
        _voice_ask = (_kind == "voice"
                      and store.get_flag(ss.run_id, "ask_fallback")
                      and not store.get_flag(ss.run_id, "fallback_decision"))
        
        if _quota_ask:
            qinfo = store.get_flag(ss.run_id, "ask_quota_wait") or {}
            wait_msg = qinfo.get("message", "Quota exhausted — waiting for reset.")
            st.warning(f"🔴 **All {len(keys)} Gemini API keys have hit their daily free quota.**\n\n"
                       f"{wait_msg}\n\n"
                       f"**What would you like to do?**")
            c1, c2 = st.columns(2)
            if c1.button("⏳ Wait for quota reset (could be hours)", key="quota_wait", type="primary"):
                store.set_flag(ss.run_id, "quota_decision", "wait")
                st.rerun()
            if c2.button("❌ Cancel transcription", key="quota_cancel"):
                store.set_flag(ss.run_id, "quota_decision", "cancel")
                st.rerun()
            st.caption("(Transcription is PAUSED — no content is lost. Your choice resumes or cancels.)")
        elif _voice_ask:
            st.warning("⏳ All Gemini keys have been cooling for a while. "
                       "What should I do for the remaining lines?")
            c1, c2 = st.columns(2)
            if c1.button("⏳ Keep waiting for Gemini keys", key="ask_wait"):
                store.set_flag(ss.run_id, "fallback_decision", "wait")
                st.rerun()
            if c2.button("🗣️ Use the standard voice for the rest", key="ask_std"):
                store.set_flag(ss.run_id, "fallback_decision", "standard")
                st.rerun()
            st.caption("(The dub is PAUSED, nothing is lost — your choice resumes it instantly.)")
        else:
            poll_job(_kind, _label)
            refresh_from_store()
            st.rerun()

# ------------------------------------------------------------------ step 1
st.header("1️⃣ Input")

def _handle_new_upload(name: str):
    return ss.get("upload_sig") != name

if mode in ("🎬 Video", "🎵 Audio"):
    is_video = mode == "🎬 Video"
    ftypes = ["mp4", "mov", "mkv", "webm", "m4v"] if is_video else ["mp3", "wav", "m4a", "ogg", "aac", "flac"]
    up = st.file_uploader(("Video file" if is_video else "Audio file"), type=ftypes)
    if up and _handle_new_upload(up.name + up.type):
        reset_stage_outputs()
        ext = Path(up.name).suffix
        media_path = run_dir / f"original{ext}"
        media_path.write_bytes(up.getbuffer())
        ss.upload_sig = up.name + up.type
        ss.video_path = str(media_path) if is_video else None
        ss.audio_input = None if is_video else str(media_path)
        try:
            videoutils.check_ffmpeg()
            with st.spinner("Preparing audio for AI (ffmpeg)…"):
                wav = videoutils.extract_audio(media_path, run_dir / "source.wav")
            ss.audio_wav = str(wav)
            ss.duration = videoutils.probe_duration_s(media_path)
            store.save_meta(ss.run_id, label=up.name, mode="video" if is_video else "audio",
                            target=target_language, duration=ss.duration)
        except RuntimeError as exc:
            st.error(str(exc))
            st.code(videoutils.ffmpeg_python_hint())
            st.stop()

    if ss.get("video_path") and Path(ss.video_path).exists():
        st.video(ss.video_path)
        st.info(f"Duration: **{ss.get('duration', 0):.1f}s** ({ss.get('duration', 0)/60:.1f} min)")
    elif ss.get("audio_input") and Path(ss.audio_input).exists():
        st.audio(ss.audio_input)
        st.info(f"Duration: **{ss.get('duration', 0):.1f}s** ({ss.get('duration', 0)/60:.1f} min)")

else:  # transcript import
    up = st.file_uploader("Excel (.xlsx/.xls) or CSV with columns: Speaker · Time From · Time To · Questions · Matter",
                          type=["xlsx", "xls", "csv"])
    if up and _handle_new_upload("import:" + up.name):
        reset_stage_outputs()
        try:
            segs = io_import.import_transcript(up.getbuffer(), up.name)
        except Exception as exc:
            st.exception(exc)
            st.stop()
        ss.segments = segs
        ss.imported = True
        ss.speakers = []
        for s in segs:
            sp = (s.get("speaker") or "").strip()
            if sp and sp not in ss.speakers:
                ss.speakers.append(sp)
        ss.duration = max(s["end"] for s in segs)
        store.save_segments(ss.run_id, segs)
        store.save_meta(ss.run_id, label=up.name, mode="import", target=target_language,
                        duration=ss.duration)
        ss.upload_sig = "import:" + up.name
        st.success(f"✅ Imported {len(segs)} rows from {up.name}"
                   + (f" · speakers: {', '.join(ss.speakers)}" if ss.speakers else "")
                   + " — Q&A rows on one line were auto-split into Questioner + Pujyashree segments.")

    vup = st.file_uploader(
        "Optional: attach the ORIGINAL video → Step 5 can output a dubbed VIDEO",
        type=["mp4", "mov", "mkv", "webm", "m4v"], key="import_video",
    )
    if vup and ss.get("import_video_sig") != vup.name:
        p = run_dir / f"original{Path(vup.name).suffix}"
        p.write_bytes(vup.getbuffer())
        ss.video_path = str(p)
        ss.import_video_sig = vup.name
        try:
            videoutils.check_ffmpeg()
            ss.duration = max(float(ss.get("duration") or 0.0), videoutils.probe_duration_s(p))
        except RuntimeError as exc:
            st.error(str(exc))
            st.code(videoutils.ffmpeg_python_hint())
            st.stop()
        st.success("🎬 Video attached. Dub timing follows your sheet's Time From/Time To — "
                   "make sure those times match this video's audio.")

ready = bool(ss.get("audio_wav")) or bool(ss.get("imported"))
segs = ss.get("segments")

# ------------------------------------------------------------------ step 2
if ready:
    if ss.get("imported"):
        st.header("2️⃣ Imported transcript (review)")
    else:
        st.header("2️⃣ Transcribe (review & fix)")
        if st.button("🗣️ Start transcription (runs in background)", type="primary"):
            job_args = dict(
                audio_path=str(ss.audio_wav), work_dir=str(run_dir), run_id=ss.run_id,
                keys=list(keys), model=asr_model, src=source_language, max_chunk=max_chunk,
                skip_songs=skip_songs,
            )

            def _transcribe_job(cb, a=job_args):
                cb(0.01, "Splitting audio into chunks…")
                rot = KeyRotator(list(a["keys"]))
                
                # --- RESUME: load existing segments & find where we left off ---
                existing_segments = store.load_segments(a["run_id"]) or []
                existing_songs = store.load_json(a["run_id"], "songs") or []
                existing_count = len(existing_segments)
                
                # Get chunk boundaries to know which chunks are already done
                chunks = asr.chunk_audio(a["audio_path"], Path(a["work_dir"]) / "chunks", 
                                         max_chunk_s=a["max_chunk"])
                
                def _chunk_range(idx):
                    """Return (start, end) of chunk idx in global timeline."""
                    _, offset = chunks[idx]
                    chunk_path = chunks[idx][0]
                    from pydub import AudioSegment
                    dur = len(AudioSegment.from_wav(str(chunk_path))) / 1000.0
                    return offset, offset + dur
                
                # Find first chunk that has NO segments covering its range
                start_chunk = 0
                if existing_count > 0:
                    for idx, (chunk_path, offset) in enumerate(chunks):
                        c_start, c_end = _chunk_range(idx)
                        # Check if any existing segment overlaps this chunk's range significantly
                        covered = any(
                            s["start"] < c_end and s["end"] > c_start
                            for s in existing_segments
                        )
                        if not covered:
                            start_chunk = idx
                            break
                    else:
                        start_chunk = len(chunks)  # all done
                
                # Start with existing segments/songs
                speech = [dict(s) for s in existing_segments]
                songs = [dict(s) for s in existing_songs]
                
                if start_chunk > 0:
                    cb(0.1, f"📂 Resuming from chunk {start_chunk + 1}/{len(chunks)} "
                         f"({existing_count} segments already transcribed)…")
                
                while True:
                    try:
                        # Process remaining chunks
                        for i in range(start_chunk, len(chunks)):
                            chunk_path, offset = chunks[i]
                            segs = asr._transcribe_chunk_safe(
                                chunk_path, rot, a["model"], a["src"], a["skip_songs"]
                            )
                            for s in segs:
                                s["start"] = round(s["start"] + offset, 3)
                                s["end"] = round(s["end"] + offset, 3)
                                if s.get("kind") == "song":
                                    songs.append({"start": s["start"], "end": s["end"]})
                                    if not a["skip_songs"]:
                                        s["text"] = s.get("text") or "🎵 (song)"
                                        speech.append(s)
                                else:
                                    s.pop("kind", None)
                                    speech.append(s)
                            
                            # --- CHECKPOINT SAVE after EACH chunk ---
                            store.save_segments(a["run_id"], speech)
                            store.save_json(a["run_id"], "songs", songs)
                            
                            if progress_cb:
                                usage = " · ".join(f"{k}:{v}" for k, v in rot.stats().items())
                                progress_cb(
                                    (i + 1) / len(chunks),
                                    f"Chunk {i + 1}/{len(chunks)} · 🔑 {usage}"
                                    + (f" · 🎵 {len(songs)} song part(s) so far" if songs else ""),
                                )
                        
                        # All chunks done
                        speech.sort(key=lambda d: d["start"])
                        songs = asr._merge_song_ranges(songs) if a["skip_songs"] else []
                        store.save_segments(a["run_id"], speech)
                        store.save_json(a["run_id"], "songs", songs)
                        return {"count": len(speech), "songs": len(songs),
                                "key_usage": rot.stats()}
                        
                    except AllKeysQuotaExhausted as e:
                        # All keys exhausted — ask user whether to wait
                        retry_after = e.retry_after_s
                        if retry_after is None or retry_after > 86400:
                            wait_msg = "Quota resets at midnight Pacific time (could be many hours)."
                        elif retry_after > 3600:
                            wait_msg = f"Estimated wait: ~{retry_after/3600:.1f} hours."
                        elif retry_after > 60:
                            wait_msg = f"Estimated wait: ~{retry_after/60:.0f} minutes."
                        else:
                            wait_msg = f"Estimated wait: ~{retry_after:.0f} seconds."
                        
                        # Set flag for UI to show prompt
                        store.set_flag(a["run_id"], "ask_quota_wait", {
                            "retry_after": retry_after,
                            "message": wait_msg,
                            "timestamp": time.time()
                        })
                        store.set_flag(a["run_id"], "quota_decision", None)
                        
                        # Update job status to waiting
                        _write_status(a["run_id"], "transcribe", status="waiting", 
                                     pct=0.5, msg=f"All keys quota exhausted. {wait_msg}")
                        
                        # Wait for user decision
                        while True:
                            decision = store.get_flag(a["run_id"], "quota_decision")
                            if decision == "wait":
                                store.set_flag(a["run_id"], "ask_quota_wait", None)
                                store.set_flag(a["run_id"], "quota_decision", None)
                                cb(0.5, "Resuming transcription after quota wait…")
                                break  # retry the transcription (will resume from last checkpoint)
                            elif decision == "cancel":
                                store.set_flag(a["run_id"], "ask_quota_wait", None)
                                store.set_flag(a["run_id"], "quota_decision", None)
                                raise RuntimeError("Transcription cancelled by user — all keys quota exhausted.")
                            time.sleep(2.0)

            store.job_start(ss.run_id, "transcribe", _transcribe_job)
            res = poll_job("transcribe", "Transcribing")
            refresh_from_store()
            if res is None:
                st.stop()
            if int(res.get("count", 0)) == 0:
                st.error("⚠️ Transcription returned NO speech — check the audio actually "
                         "contains speech (try a short loud clip to test first).")
                st.stop()
            st.rerun()
        st.caption("Safe to switch tabs — the job keeps running server-side and saves to this project.")
        _tres = store.job_result(ss.run_id, "transcribe")
        if _tres:
            if _tres.get("key_usage"):
                st.caption("🔑 Transcription key rotation: " + ", ".join(
                    f"{k} → {v} calls" for k, v in _tres["key_usage"].items()))
            if _tres.get("songs"):
                st.success(f"🎵 {_tres['songs']} song/music section(s) were detected and skipped "
                           "— their ORIGINAL audio will be kept in the final dub.")

    segs = ss.get("segments")
    if segs:
        has_speaker = any((s.get("speaker") or "").strip() for s in segs)
        df = pd.DataFrame(
            [{"start": s.get("start", 0.0), "end": s.get("end", 0.0),
              "speaker": s.get("speaker", ""), "text": s.get("text", "")}
             for s in segs]
        )
        cols_cfg = {
            "start": st.column_config.NumberColumn("Start (s)", format="%.2f", width="small"),
            "end": st.column_config.NumberColumn("End (s)", format="%.2f", width="small"),
            "text": st.column_config.TextColumn(f"{source_language} text — edit here if AI misheard", width="large"),
        }
        if has_speaker:
            cols_cfg["speaker"] = st.column_config.TextColumn("Speaker", width="small")
        edited = _safe_editor(df, num_rows="dynamic", column_config=cols_cfg, key="transcript_editor")
        merge_editor_rows(edited.to_dict("records"), ("start", "end", "text", "speaker"))
        segs = ss.segments

# ------------------------------------------------------- proofreading panel
if ready and segs:
    with st.expander("🔎 Proofreading & completeness — verify nothing was skipped", expanded=True):
        rep = qa.translation_report(segs)
        m1, m2, m3 = st.columns(3)
        m1.metric("Segments", rep["total"])
        m2.metric("Translated", rep["translated"])
        m3.metric("Missing translation", len(rep["missing"]) + len(rep["identical"]))
        if ss.get("audio_wav"):
            st.caption(f"Transcript covers speech from {fmt_ts(segs[0]['start'])} to {fmt_ts(segs[-1]['end'])} "
                       f"(segment-time coverage {qa.coverage_pct(segs, ss.get('duration', 0))}% of file duration — "
                       f"the rest is pauses/music, that's normal).")
            _songs = ss.get("songs") or []
            if _songs:
                st.info("🎵 Song/music sections skipped (kept in original audio): "
                        + " · ".join(f"{fmt_ts(r['start'])}–{fmt_ts(r['end'])}" for r in _songs[:10])
                        + (f" … +{len(_songs) - 10} more" if len(_songs) > 10 else ""))
            if st.button("🔍 Scan for possibly missed speech"):
                with st.spinner("Analyzing audio for uncovered speech…"):
                    try:
                        gaps = qa.speech_gaps(ss.audio_wav, segs, ignore_ranges=ss.get("songs"))
                    except Exception as exc:
                        st.error(str(exc))
                        gaps = None
                if gaps is not None:
                    if gaps:
                        st.warning(f"Found {len(gaps)} spot(s) with speech but NO transcript — "
                                   f"listen there and add/extend rows in the table above if needed:")
                        st.dataframe(pd.DataFrame(
                            [{"from": fmt_ts(g), "to": fmt_ts(h), "approx_seconds": round(h - g, 1)}
                             for g, h in gaps][:50]))
                    else:
                        st.success("✅ No uncovered speech detected — nothing was skipped.")
        else:
            st.caption("Imported transcript — audio coverage scan not applicable (no source audio).")
        if rep["identical"]:
            st.info(f"{len(rep['identical'])} row(s) have translation identical to the original "
                    f"(often names/numbers — or untranslated leftovers). Check rows: "
                    + ", ".join(str(i + 1) for i in rep["identical"][:20]))

# ------------------------------------------------------------------ step 3
def _translate_job_factory(only_missing: bool, a: dict):
    def _job(cb):
        rot = KeyRotator(list(a["keys"]))
        segs = [dict(s) for s in (store.load_segments(a["run_id"]) or a["segments"])]
        if only_missing:
            idxs = [i for i, s in enumerate(segs)
                    if not str(s.get("translated") or "").strip()
                    or str(s.get("translated")).strip() == str(s.get("text")).strip()]
            if not idxs:
                return {"translated": 0}
            subset = [segs[i] for i in idxs]
            out = translate.translate_segments(
                subset, a["target"], rot, model=a["model"],
                source_language=a["src"], progress_cb=cb)
            for i, row in zip(idxs, out):
                segs[i]["translated"] = row["translated"]
            translated_n = len(out)
        else:
            segs = translate.translate_segments(
                segs, a["target"], rot, model=a["model"],
                source_language=a["src"], progress_cb=cb)
            translated_n = len(segs)
        store.save_segments(a["run_id"], segs)
        return {"translated": translated_n}
    return _job

if ready and segs:
    st.header("3️⃣ Translate")
    _t_args = dict(run_id=ss.run_id, keys=list(keys), target=target_language,
                   model=translate_model, src=source_language,
                   segments=[dict(s) for s in ss.segments])
    c1, c2 = st.columns([1, 1])
    if c1.button(f"🌍 Translate ALL to {target_language}", type="primary"):
        store.job_start(ss.run_id, "translate", _translate_job_factory(False, _t_args))
        res = poll_job("translate", "Translating")
        refresh_from_store()
        if res is None:
            st.stop()
        st.rerun()
    if c2.button("🩹 Translate only missing/identical rows"):
        store.job_start(ss.run_id, "translate", _translate_job_factory(True, _t_args))
        res = poll_job("translate", "Translating missing rows")
        refresh_from_store()
        if res is None:
            st.stop()
        st.rerun()

    if any(s.get("translated") for s in ss.segments):
        st.caption("ℹ️ Left = original (stays unchanged). Right = translation (editable). "
                   "Speaker labels are never translated — they select the voice.")
        has_speaker = any((s.get("speaker") or "").strip() for s in ss.segments)
        df = pd.DataFrame(
            [{"speaker": s.get("speaker", ""), "text": s.get("text", ""),
              "translated": s.get("translated", "")} for s in ss.segments]
        )
        cols_cfg = {
            "text": st.column_config.TextColumn(f"Original ({source_language})", width="medium", disabled=True),
            "translated": st.column_config.TextColumn(f"Translation ({target_language}) — editable", width="large"),
        }
        if has_speaker:
            cols_cfg["speaker"] = st.column_config.TextColumn("Speaker", width="small", disabled=True)
        edited = _safe_editor(df, column_config=cols_cfg, key="translation_editor")
        merge_editor_rows(edited.to_dict("records"), ("translated", "speaker"))
        segs = ss.segments

# ------------------------------------------------------------------ step 4
if ready and any(s.get("translated") for s in ss.get("segments") or []):
    st.header("4️⃣ Generate dubbed audio")
    if dual_voices:
        st.caption("🎭 Two-voice mode: " + "; ".join(f"{sp} → **{speaker_voices[sp]}**" for sp in speaker_voices))
    st.caption("⏳ ~2–4 min per 5 min of content. Runs in background — safe to leave and come back.")
    if st.button("🎙️ Start voicing & alignment (background)", type="primary"):
        sp_map = dict(speaker_voices) if dual_voices else None
        _v_args = dict(
            run_id=ss.run_id, work_dir=str(run_dir),
            keys=list(keys) if engine == "gemini" else [],
            engine=engine, voice=voice, fallback=edge_fallback, sp_map=sp_map,
            duration=ss.duration, segments=[dict(s) for s in ss.segments],
            pacing=pacing_mode, policy=exhausted_policy,
        )

        def _voice_job(cb, a=_v_args):
            rot = KeyRotator(list(a["keys"])) if a["keys"] else None
            run_p = Path(a["work_dir"])
            store.set_flag(a["run_id"], "ask_fallback", False)
            store.set_flag(a["run_id"], "fallback_decision", None)
            interaction = {
                "threshold_s": 180.0,
                "set_ask": lambda b: store.set_flag(a["run_id"], "ask_fallback", b),
                "decision": lambda: store.get_flag(a["run_id"], "fallback_decision"),
            }
            voiced = tts.synthesize_track(
                a["segments"], run_p / "voice_segments",
                engine=a["engine"], voice=a["voice"], rotator=rot,
                edge_fallback_voice=a["fallback"],
                speaker_voices=a["sp_map"],
                exhausted_policy=a["policy"], interaction=interaction,
                progress_cb=lambda p, m: cb(0.8 * p, m),
            )
            store.set_flag(a["run_id"], "ask_fallback", False)
            store.set_flag(a["run_id"], "fallback_decision", None)
            track = align.build_track(
                voiced, a["duration"], run_p / "dubbed.wav",
                run_p / "stretch_tmp",
                progress_cb=lambda p, m: cb(0.8 + 0.2 * p, m),
                mode=a["pacing"],
            )
            # keep the ORIGINAL audio under skipped song/bhajan sections
            songs = store.load_json(a["run_id"], "songs") or []
            songs_mixed = 0
            if songs:
                src_wav = run_p / "source.wav"
                if not src_wav.exists():
                    orig = next(iter(sorted(run_p.glob("original.*"))), None)
                    if orig:
                        try:
                            src_wav = Path(videoutils.extract_audio(orig, run_p / "song_source.wav"))
                        except Exception:
                            src_wav = None
                if src_wav and src_wav.exists():
                    cb(0.98, f"Mixing {len(songs)} original song section(s) into the dub…")
                    align.overlay_regions(str(track), str(src_wav), songs)
                    songs_mixed = len(songs)
            counts = {"gemini": 0, "edge": 0}
            for r in voiced:
                if r.get("engine") in counts:
                    counts[r["engine"]] += 1
            return {
                "track": str(track), **counts, "songs_mixed": songs_mixed,
                "key_usage": rot.stats() if rot else {},
            }

        store.job_start(ss.run_id, "voice", _voice_job)
        res = poll_job("voice", "Voicing")
        refresh_from_store()
        if res is None:
            st.stop()
        st.rerun()

    _vres = store.job_result(ss.run_id, "voice") or {}
    if _vres.get("key_usage"):
        st.caption("🔑 Key rotation stats: " + ", ".join(
            f"{k} → {v} calls" for k, v in _vres["key_usage"].items()))
    if _vres.get("songs_mixed"):
        st.success(f"🎵 {_vres['songs_mixed']} song/music section(s) kept their ORIGINAL audio in the dub.")
    if engine == "gemini" and _vres.get("edge"):
        st.info(
            f"ℹ️ {_vres['edge']} of {_vres.get('gemini', 0) + _vres.get('edge', 0)} segments "
            f"temporarily used the gender-matched Edge backup while ALL Gemini keys were "
            f"cooling down (rotation then resumed automatically). Add more keys from "
            f"DIFFERENT projects to shrink this."
        )

    if ss.get("track_wav") and Path(ss.track_wav).exists():
        st.subheader("🎧 Preview final audio track")
        st.audio(ss.track_wav)
        st.download_button("⬇️ Download audio (WAV)",
                           Path(ss.track_wav).read_bytes(),
                           file_name=f"dub_{target_language.lower()}.wav", mime="audio/wav")

# ------------------------------------------------------------------ step 5
if ready and ss.get("track_wav") and Path(ss.track_wav).exists():
    st.header("5️⃣ Export")
    videoutils.write_srt(ss.segments, run_dir / "original.srt")
    videoutils.write_srt(ss.segments, run_dir / "dub.srt", text_key="translated")
    st.download_button("⬇️ Subtitles (SRT): original", (run_dir / "original.srt").read_bytes(),
                       file_name="original.srt", key="dl_srt_orig")
    st.download_button(f"⬇️ Subtitles (SRT): {target_language}", (run_dir / "dub.srt").read_bytes(),
                       file_name="dub.srt", key="dl_srt_dub")

    if ss.get("video_path") and Path(ss.video_path).exists():
        if st.button("▶️ Build dubbed video + lip-sync package", type="primary"):
            with st.spinner("Muxing video…"):
                dubbed = videoutils.mux_audio_video(
                    ss.video_path, ss.track_wav,
                    run_dir / f"dubbed_{target_language.lower()}.mp4")
            pkg = run_dir / "lipsync_package"
            pkg.mkdir(exist_ok=True)
            shutil.copy(ss.video_path, pkg / "original.mp4")
            shutil.copy(ss.track_wav, pkg / "dub.wav")
            (pkg / "INSTRUCTIONS.txt").write_text(
                "Open lipsync_colab.ipynb in Colab (T4 GPU) and upload this zip.\n",
                encoding="utf-8")
            zip_path = run_dir / "lipsync_package.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for f in pkg.iterdir():
                    zf.write(f, f.name)
            ss.dubbed_video = str(dubbed)
            ss.zip_path = str(zip_path)

        col_a, col_b = st.columns(2)
        if ss.get("dubbed_video") and Path(ss.dubbed_video).exists():
            with col_a:
                st.subheader("Dubbed video")
                st.video(ss.dubbed_video)
                st.download_button("⬇️ Download dubbed video",
                                   Path(ss.dubbed_video).read_bytes(),
                                   file_name=f"dubbed_{target_language.lower()}.mp4", mime="video/mp4")
            with col_b:
                st.subheader("For lip-sync (Colab)")
                st.download_button("⬇️ Download lip-sync package (.zip)",
                                   Path(ss.zip_path).read_bytes(),
                                   file_name="lipsync_package.zip", mime="application/zip")

        st.divider()
        st.subheader("📝 Optional: subtitled version (no Colab needed)")
        st.caption("Burns translated subtitles into the dubbed video. Good for long videos.")
        if st.button("Create subtitled video (background)"):
            if not ss.get("dubbed_video"):
                st.warning("Build the dubbed video first (button above).")
            else:
                _s_args = dict(video=str(ss.dubbed_video),
                               srt=str(run_dir / "dub.srt"),
                               out=str(run_dir / f"dubbed_{target_language.lower()}_subtitled.mp4"))

                def _subtitle_job(cb, a=_s_args):
                    cb(0.1, "rendering…")
                    out = videoutils.burn_subtitles(a["video"], a["srt"], a["out"])
                    cb(1.0, "done")
                    return {"video": str(out)}

                store.job_start(ss.run_id, "subtitle", _subtitle_job)
                res = poll_job("subtitle", "Subtitles")
                refresh_from_store()
                if res is None:
                    st.stop()
        if store.job_result(ss.run_id, "subtitle"):
            subbed = store.job_result(ss.run_id, "subtitle").get("video")
            if subbed and Path(subbed).exists():
                st.video(subbed)
                st.download_button("⬇️ Download subtitled video", Path(subbed).read_bytes(),
                                   file_name=f"dubbed_{target_language.lower()}_subtitled.mp4",
                                   mime="video/mp4", key="dl_subbed")
    else:
        st.info("🎵 Audio/transcript project — the dubbed audio (WAV) and subtitles above are your outputs. "
                "Lip-sync applies to video input only.")
