# SETUP_WINDOWS.md
# Hikari Shinro AI — Windows Setup Guide
# ==========================================
# Tested on: Windows 10/11, Python 3.10/3.11

---

## STEP 0 — Prerequisites (install once)

### Python 3.10 or 3.11
Download from: https://www.python.org/downloads/
✅ During install: CHECK "Add Python to PATH"
✅ During install: CHECK "Add Python to environment variables"

Verify in Command Prompt:
```
python --version
```
Should show: Python 3.10.x or 3.11.x


### Git
Download from: https://git-scm.com/download/win
(needed to clone the repo)


### Microsoft C++ Build Tools (needed for PyAudio)
Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- Click "Download Build Tools"
- In installer, check: "Desktop development with C++"
- Click Install (takes ~5-10 mins)


### FFmpeg (needed for Whisper audio processing)
1. Download from: https://www.gyan.dev/ffmpeg/builds/
   → Click "ffmpeg-release-essentials.zip"
2. Extract to: C:\ffmpeg\
3. Add to PATH:
   - Open Start → search "Environment Variables"
   - Click "Environment Variables"
   - Under "System variables", click "Path" → "Edit"
   - Click "New" → type: C:\ffmpeg\bin
   - Click OK on all windows
4. Verify: open new Command Prompt → type: ffmpeg -version

---

## STEP 1 — Clone the project

Open Command Prompt (Win+R → cmd → Enter):
```cmd
cd C:\
git clone https://github.com/YOUR_TEAM/hikari-shinro-ai
cd hikari-shinro-ai
```

---

## STEP 2 — Create virtual environment

```cmd
python -m venv venv
venv\Scripts\activate
```

You should see (venv) at the start of your prompt.

---

## STEP 3 — Install PyTorch (CPU version — works on any laptop)

```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

If your laptop has an NVIDIA GPU (optional, faster):
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## STEP 4 — Install PyAudio (Windows-specific method)

PyAudio needs a pre-compiled wheel on Windows. Use pipwin:

```cmd
pip install pipwin
pipwin install pyaudio
```

If pipwin fails, download the wheel directly:
1. Go to: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
2. Download: PyAudio‑0.2.14‑cp310‑cp310‑win_amd64.whl
   (use cp310 for Python 3.10, cp311 for Python 3.11)
3. Install it:
```cmd
pip install PyAudio-0.2.14-cp310-cp310-win_amd64.whl
```

---

## STEP 5 — Install all other dependencies

```cmd
pip install -r requirements.txt
```

---

## STEP 6 — Configure environment

```cmd
copy .env.example .env
notepad .env
```

In Notepad, replace "your_groq_api_key_here" with your actual key.
Get a FREE key at: https://console.groq.com
(Sign up → API Keys → Create API Key → Copy)

Save and close Notepad.

---

## STEP 7 — First run (downloads models ~500MB)

```cmd
cd app
python app.py
```

On first run, it downloads:
- Whisper base model (~145MB)
- YOLO-World-S model (~50MB)
- MiDaS Small model (~83MB)

This takes 5-10 minutes with good internet. Subsequent runs are instant.

---

## STEP 8 — Open the HUD

Open your browser: http://localhost:5000

You should see the Hikari Shinro AI anime HUD.

---

## TROUBLESHOOTING

### "No module named 'cv2'"
```cmd
pip install opencv-python
```

### "Camera not found" / device error
Try changing the camera index in .env:
```
CAMERA_DEVICE=1
```
(0 = built-in webcam, 1 = first external USB camera)

### "GROQ_API_KEY not set"
Make sure your .env file has the key and you're running from the
hikari-shinro-ai/ folder (not inside app/).

### "DLL load failed" for PyAudio
Install Microsoft Visual C++ Redistributable:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### PyTorch "RuntimeError: CUDA not available"
That's fine — the system runs on CPU automatically. No GPU needed.

### Whisper "FileNotFoundError: ffmpeg"
FFmpeg not in PATH. Re-do Step 0 → FFmpeg section.
Then close and reopen Command Prompt.

### "Port 5000 already in use"
Change the port in app.py line at the bottom:
```python
socketio.run(app, host="0.0.0.0", port=5001, ...)
```
Then open: http://localhost:5001

### pyttsx3 no audio
Ensure Windows audio is not muted.
Test in Command Prompt:
```python
python -c "import pyttsx3; e=pyttsx3.init(); e.say('hello'); e.runAndWait()"
```

---

## FULL VERIFIED INSTALL SEQUENCE (copy-paste this)

```cmd
:: Run this in Command Prompt after installing Python, C++ Build Tools, FFmpeg

cd C:\hikari-shinro-ai
python -m venv venv
venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt

copy .env.example .env
:: Edit .env with your GROQ_API_KEY

cd app
python app.py
```

---

## DEMO CHECKLIST (for hackathon judges)

Before presenting, verify:
- [ ] Webcam is plugged in and recognized
- [ ] Browser open at http://localhost:5000
- [ ] Goal set in the text box
- [ ] START button pressed
- [ ] Audio is unmuted on laptop
- [ ] Venue WiFi connected (Groq API needs internet)

Offline fallback: if venue internet drops, the system still runs
detection + depth + quant. Only the LLM agent switches to rule-based
fallback mode (defined in agent.py → _fallback_response).

---

*Hikari Shinro AI — 光の進路 — Path of Light*
*CodeRonin · Ahouba 3.0 · IIIT Manipur · 21–22 March 2026*
