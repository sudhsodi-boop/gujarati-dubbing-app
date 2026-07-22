"""Generate lipsync_colab.ipynb programmatically (keeps the JSON valid).

Run:  python make_notebook.py
"""

import json
from pathlib import Path

CELLS = []


def md(text: str):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    })


def code(text: str):
    CELLS.append({
        "cell_type": "code",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
        "execution_count": None,
        "outputs": [],
    })


md("""# 🎬 Lip-Sync Notebook — MuseTalk on Colab (LONG-VIDEO MODE 🆕)

Takes `lipsync_package.zip` from the Streamlit app (**original video** + **dubbed audio**)
and re-animates the speaker's mouth to match the new audio.

**Long videos welcome:** the video is auto-split into chunks (default 15 min), each chunk is
lip-synced separately with **progress saved** — so if Colab disconnects, just re-run the same
cell and it continues where it stopped. ✅

**Before running:** `Runtime ▸ Change runtime type ▸ T4 GPU` (free tier works).

Run cells **top to bottom**. Keep this browser tab open while it runs.""")


code("""# 1️⃣ Check GPU
!nvidia-smi -L""")


code("""# 2️⃣ Upload lipsync_package.zip (or upload original.mp4 + dub.wav separately)
import os, zipfile, glob, shutil

os.makedirs('/content/pack', exist_ok=True)

try:
    from google.colab import files
    uploaded = files.upload()  # pick lipsync_package.zip (be patient: large files take a while)
except ImportError:
    print("Not in Colab — place the files in /content/pack manually.")

for name in os.listdir('/content'):
    if name.endswith('.zip'):
        with zipfile.ZipFile(f'/content/{name}') as z:
            z.extractall('/content/pack')
        print(f"Extracted {name}")
        break

# also allow loose uploads at /content
for f in glob.glob('/content/*.mp4') + glob.glob('/content/*.wav'):
    shutil.copy(f, '/content/pack/')

VIDEO = next(iter(sorted(glob.glob('/content/pack/**/original*.mp4', recursive=True)
              + glob.glob('/content/pack/*.mp4'))), None)
AUDIO = next(iter(sorted(glob.glob('/content/pack/**/dub*.wav', recursive=True)
              + glob.glob('/content/pack/*.wav'))), None)
assert VIDEO and AUDIO, "original.mp4 / dub.wav not found — upload lipsync_package.zip!"
print("VIDEO:", VIDEO)
print("AUDIO:", AUDIO)""")


code("""# 3️⃣ Clone MuseTalk + install dependencies (~3-5 min, one-time per session)
%cd /content
!git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git
%cd /content/MuseTalk
!pip install -q -r requirements.txt
!pip install -q --no-cache-dir -U openmim
!mim install -q mmengine
!mim install -q "mmcv>=2.0.1"
!mim install -q "mmdet>=3.1.0"
!mim install -q "mmpose>=1.1.0"
print("Install finished (ignore resolver warnings). If imports fail later: Runtime ▸ Restart, then re-run from cell 4.")""")


code("""# 4️⃣ Download MuseTalk model weights from HuggingFace (one-time)
%cd /content/MuseTalk
!pip install -q huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download("TMElyralab/MuseTalk", local_dir="/content/MuseTalk/models")
!find models -maxdepth 2 -type d | head -20
# If this fails (repo moved/gated), open the MuseTalk GitHub README "Model Zoo"
# section and place the weights under /content/MuseTalk/models/ manually.""")


code("""# ✂️ 5️⃣ Split video + dub audio into chunks (needed so a long video fits & resumes)
import os, subprocess, math, wave, contextlib

CHUNK_MINUTES = 15          # 15 min/chunk works well on a free T4
CHUNK_S = CHUNK_MINUTES * 60
WORK = '/content/work'
os.makedirs(f'{WORK}/chunks', exist_ok=True)

def _run(cmd, quiet=True):
    if not quiet:
        print('>', ' '.join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('\\n'.join(r.stderr.splitlines()[-25:]))

def duration_of(path):
    out = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration',
                          '-of','default=noprint_wrappers=1:nokey=1', path],
                         capture_output=True, text=True).stdout.strip()
    return float(out)

dur_v = duration_of(VIDEO)
dur_a = duration_of(AUDIO)
total = max(dur_v, dur_a)
n = max(1, math.ceil(total / CHUNK_S))
print(f"Video {dur_v/60:.1f} min | audio {dur_a/60:.1f} min -> {n} chunk(s) of up to {CHUNK_MINUTES} min")
print("(Re-encoding chunks for accuracy — a few minutes for long videos, one-time.)")

jobs = []
for i in range(n):
    start = i * CHUNK_S
    dur = min(CHUNK_S, total - start)
    vpath = f'{WORK}/chunks/v_{i:03d}.mp4'
    apath = f'{WORK}/chunks/a_{i:03d}.wav'
    done  = f'{WORK}/done_{i:03d}'
    jobs.append((i, start, dur, vpath, apath, done))
    print(f'  chunk {i+1}/{n}: {start/60:.1f}–{(start+dur)/60:.1f} min')
    if not os.path.exists(vpath):
        _run(['ffmpeg','-y','-ss',str(start),'-t',str(dur),'-i',VIDEO,
              '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
              '-an', vpath])
    if not os.path.exists(apath):
        _run(['ffmpeg','-y','-ss',str(start),'-t',str(dur),'-i',AUDIO,
              '-ac','1','-ar','16000','-c:a','pcm_s16le', apath])
print('✅ Chunks ready.')""")


code("""# ▶️ 6️⃣ Run MuseTalk chunk-by-chunk (RESUME-SAFE: finished chunks are skipped)
# If Colab disconnects mid-way: reconnect, re-run THIS cell (and cell 5 if the VM was wiped).
BBOX_SHIFT = 0     # raise/lower (e.g. 5 or -5) only if the mouth box looks offset

import os, glob, shutil, subprocess, time
os.makedirs(f'{WORK}/synced', exist_ok=True)
unet = '/content/MuseTalk/models/musetalk/pytorch-model.bin'

for (i, start, dur, vpath, apath, done) in jobs:
    if os.path.exists(done):
        print(f'⏭️  chunk {i+1}/{len(jobs)} already done — skipping')
        continue
    t0 = time.time()
    print(f'--- 🎬 chunk {i+1}/{len(jobs)} ({start/60:.1f}–{(start+dur)/60:.1f} min) ---')
    ypath = f'/content/MuseTalk/configs/inference/chunk_{i:03d}.yaml'
    with open(ypath, 'w') as f:
        f.write(f'task_{i}:\\n  video_path: "{vpath}"\\n  audio_path: "{apath}"\\n  bbox_shift: {BBOX_SHIFT}\\n')
    cmd = ['python', '-m', 'scripts.inference',
           '--inference_config', ypath.replace('/content/MuseTalk/', ''),
           '--result_dir', f'{WORK}/synced/chunk_{i:03d}']
    if os.path.exists(unet):
        cmd += ['--unet_model_path', unet]
    r = subprocess.run(cmd, cwd='/content/MuseTalk')
    if r.returncode != 0:
        raise RuntimeError(f'Chunk {i+1} failed (see logs above). Fix, then just RE-RUN this cell — progress is kept.')
    mp4s = glob.glob(f'{WORK}/synced/chunk_{i:03d}/**/*.mp4', recursive=True)
    assert mp4s, f'Chunk {i+1}: no output video was produced'
    shutil.copy(mp4s[-1], f'{WORK}/synced/final_{i:03d}.mp4')
    open(done, 'w').write('ok')
    print(f'✅ chunk {i+1}/{len(jobs)} done in {(time.time()-t0)/60:.1f} min')
print('🎉 All chunks lip-synced! Run the next cell to merge.')""")


code("""# 🧩 7️⃣ Merge chunks + attach the full dubbed audio  ->  /content/final.mp4
import os, glob, subprocess

parts = sorted(glob.glob(f'{WORK}/synced/final_*.mp4'))
assert len(parts) == len(jobs), f'Not all chunks finished yet: {len(parts)}/{len(jobs)} — re-run the previous cell.'

open(f'{WORK}/concat.txt', 'w').write(''.join(f"file '{p}'\\n" for p in parts))
subprocess.run(['ffmpeg','-y','-f','concat','-safe','0',
                '-i', f'{WORK}/concat.txt', '-c','copy', f'{WORK}/merged.mp4'],
               check=True, capture_output=True)
# attach the full-length dub audio so the soundtrack is perfect end-to-end
subprocess.run(['ffmpeg','-y','-i', f'{WORK}/merged.mp4','-i', AUDIO,
                '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','192k',
                '-shortest','/content/final.mp4'],
               check=True, capture_output=True)

sz = os.path.getsize('/content/final.mp4') / 1e6
print(f'✅ final.mp4 ready ({sz:.0f} MB)')

from IPython.display import Video, display
if total < 240:
    display(Video('/content/final.mp4', embed=True, width=720))
else:
    print('(Video is long — preview skipped.)')

try:
    from google.colab import files
    files.download('/content/final.mp4')
except Exception:
    pass
print("💡 Large file? Left toolbar → 📁 Files → find final.mp4 → ⋮ → Download")""")


md("""## 🛠 Troubleshooting

- **Colab disconnected mid-run** → reconnect and **re-run cell 6** (finished chunks are
  skipped). If the whole VM was reset, re-run cells 3→7 (cell 5 re-splits quickly if files survived).
- **Face not detected** → speaker must be mostly front-facing; cut away side-profile scenes first.
- **Mouth area looks shifted** → set `BBOX_SHIFT = 5` or `-5` in cell 6 and re-run.
- **`mmcv` install fails** → run `mim install "mmcv==2.1.0"`, then `Runtime ▸ Restart`, continue.
- **Script/arg mismatch** (repo changed) → check https://github.com/TMElyralab/MuseTalk

### 🔁 Reusing for another video in the same session
Re-run cell 2 with the new zip, then cells 5 → 7. **Tip:** delete old markers first:
`!rm -f /content/work/done_* /content/work/chunks/*`
""")

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": CELLS,
}

out = Path(__file__).parent / "lipsync_colab.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {out} ({len(CELLS)} cells)")
