# ☁️ Deploy to Streamlit Community Cloud (FREE hosting)

No local Python, no local ffmpeg — the app runs online at a URL like
`https://your-app.streamlit.app`.

**What stays on Colab:** the *lip-sync* step still needs a GPU → keep using
`lipsync_colab.ipynb` in Google Colab (free T4). The cloud app does everything else:
upload → transcribe → translate → voice → produces `lipsync_package.zip`.

---

## Step 1: Create a GitHub account (free)
1. Go to https://github.com → **Sign up** (just email + password).
2. Verify your email.

## Step 2: Put the project on GitHub (no coding, browser only)
1. After signing in, click the **＋** (top-right) → **New repository**.
2. Name: `gujarati-dubbing-app` → Public (required for free Streamlit Cloud) → **Create**.
3. On the new repo page, click **"uploading an existing file"**.
4. Drag ALL files from your unzipped `gujarati-dubbing-app` folder into the page:
   - `app.py`, `README.md`, `requirements.txt`, `packages.txt`
   - the `dubbing` → just drag the **whole folder** in (GitHub keeps the structure)
   - the `.streamlit` folder (contains `config.toml`)
   - optional: `SOLUTION_PLAN.md`, `lipsync_colab.ipynb`
5. Click **Commit changes**.

## Step 3: Deploy
1. Go to https://share.streamlit.io → **Sign in with GitHub**.
2. Click **"New app"** / **"Create app"**.
3. Pick:
   - Repository: `yourname/gujarati-dubbing-app`
   - Branch: `main`
   - Main file path: `app.py`
4. **Advanced settings → Secrets** (recommended): paste your keys like this:
   ```toml
   GEMINI_API_KEYS = "key1,key2,key3"
   ```
   (Or just leave secrets empty and paste keys in the app sidebar each session.)
5. Click **Deploy** and wait ~2–4 minutes. Done! 🎉

Your app is now online 24/7 at `https://<name>.streamlit.app` — shareable with anyone.

---

## Good to know
- **ffmpeg** is auto-installed from `packages.txt` on every deploy. ✅
- Free tier: ~1 GB RAM, app **sleeps after ~15 min of no traffic** (wakes in ~30 s on visit).
- Video upload limit set to 1 GB (`.streamlit/config.toml`). Very long videos are
  still better handled on your own machine.
- To update the app later: edit files on GitHub (web editor) → Streamlit redeploys automatically.
- **Lip-sync reminder:** download `lipsync_package.zip` from the app → run `lipsync_colab.ipynb` on Colab (free T4 GPU) → get the final lip-synced video. Colab is unavoidable for the GPU step.

## Alternative: Hugging Face Spaces
Also free and works with this exact project (Streamlit SDK + `packages.txt`):
https://huggingface.co/new-space → SDK: Streamlit → upload the same files.
Takes ffmpeg from `packages.txt` automatically.
