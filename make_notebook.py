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


md("""# 🎬 Lip-Sync Notebook — MuseTalk on Colab

Takes `lipsync_package.zip` from the Streamlit app (**original video** + **dubbed audio**)
and re-animates the speaker's mouth to match the new audio.

**Before running:** `Runtime ▸ Change runtime type ▸ T4 GPU` (free tier works).

Run cells **top to bottom**.""")


code("""# 1️⃣ Check GPU
!nvidia-smi -L""")


code("""# 2️⃣ Upload lipsync_package.zip (or upload original.mp4 + dub.wav separately)
import os, zipfile, glob, shutil

os.makedirs('/content/pack', exist_ok=True)

try:
    from google.colab import files
    uploaded = files.upload()  # pick lipsync_package.zip
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


code("""# 3️⃣ Clone MuseTalk + install dependencies (~3-5 min)
%cd /content
!git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git
%cd /content/MuseTalk
!pip install -q -r requirements.txt
!pip install -q --no-cache-dir -U openmim
!mim install -q mmengine
!mim install -q "mmcv>=2.0.1"
!mim install -q "mmdet>=3.1.0"
!mim install -q "mmpose>=1.1.0"
print("Install finished. If you see import errors below, use: Runtime ▸ Restart and run cell 4 onward.")""")


code("""# 4️⃣ Download MuseTalk model weights from HuggingFace
%cd /content/MuseTalk
!pip install -q huggingface_hub
from huggingface_hub import snapshot_download
snapshot_download("TMElyralab/MuseTalk", local_dir="/content/MuseTalk/models")
!find models -maxdepth 2 -type d | head -20
# If this fails (repo moved/gated), open the MuseTalk GitHub README "Model Zoo"
# section and place the weights under /content/MuseTalk/models/ manually.""")


code("""# 5️⃣ Run MuseTalk lip-sync
# bbox_shift: move the lip-sync box up(+)/down(-) if the mouth region looks offset.
BBOX_SHIFT = 0

import os, subprocess
VIDEO_PATH = VIDEO  # absolute path from cell 2
AUDIO_PATH = AUDIO

yaml_text = f'''task_0:
  video_path: "{VIDEO_PATH}"
  audio_path: "{AUDIO_PATH}"
  bbox_shift: {BBOX_SHIFT}
'''
os.makedirs('/content/MuseTalk/configs/inference', exist_ok=True)
with open('/content/MuseTalk/configs/inference/ours.yaml', 'w') as f:
    f.write(yaml_text)

cmd = ["python", "-m", "scripts.inference",
       "--inference_config", "configs/inference/ours.yaml",
       "--result_dir", "/content/results"]
unet = "/content/MuseTalk/models/musetalk/pytorch-model.bin"
if os.path.exists(unet):
    cmd += ["--unet_model_path", unet]

print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd="/content/MuseTalk")

!find /content/results -name "*.mp4" -ls""")


code("""# 6️⃣ Mux to guarantee the dubbed audio track + preview + download
import glob, os

outs = sorted(glob.glob('/content/results/**/*.mp4', recursive=True),
              key=os.path.getmtime)
assert outs, "No output mp4 found — check the MuseTalk logs above."
synced = outs[-1]
print("MuseTalk output:", synced)

!ffmpeg -y -i "$synced" -i "$AUDIO" -map 0:v:0 -map 1:a:0 \\
    -c:v copy -c:a aac -shortest /content/final.mp4 -loglevel error

from IPython.display import Video, display
display(Video('/content/final.mp4', embed=True, width=720))

try:
    from google.colab import files
    files.download('/content/final.mp4')
except ImportError:
    pass""")


md("""## 🛠 Troubleshooting

- **Face not detected / no output** → the speaker must be mostly **front-facing**.
  Crop the video so the face is clearly visible and not too small.
- **Mouth area looks shifted/blurry box** → rerun cell 5 with `BBOX_SHIFT = 5` or `-5`.
- **`mmcv` install fails** → run `mim install "mmcv==2.1.0"`, restart runtime, continue.
- **Script/arg mismatch** (repo changed) → check the current README:
  https://github.com/TMElyralab/MuseTalk

### Fallback: Wav2Lip (lower quality, very reliable)
If MuseTalk refuses to run:
```
%cd /content
!git clone https://github.com/Rudrabha/Wav2Lip.git && cd Wav2Lip
!pip install -q -r requirements.txt
# download wav2lip_gan.pth into Wav2Lip/checkpoints/ (see repo README for mirrors)
!python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth \\
    --face "$VIDEO" --audio "$AUDIO" --outfile /content/final.mp4 --pads 0 10 0 0
```
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
