# Hikari Shinro AI 光の進路
**Voice-First Agentic Navigation for the Visually Impaired**
CodeRonin · Ahouba 3.0 · IIIT Manipur · PS-01

## Quick Start (Windows)
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env
# Add your GROQ_API_KEY from console.groq.com (free)
cd app
python app.py
# Open http://localhost:5000
```

## How It Works
1. Speak your goal → Whisper transcribes locally
2. LLM Agent (Groq) decomposes goal into subtasks
3. YOLO-World detects objects (no retraining)
4. MiDaS estimates real-world depth
5. Quant Engine: Kalman Filter + A* + Euclidean Distance
6. pyttsx3 speaks navigation guidance aloud
7. Flask HUD shows live annotated feed
