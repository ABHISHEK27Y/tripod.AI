# Hikari Shinro AI 光の進路

> **Agentic AI Navigation for the Visually Impaired**
> CodeRonin Hackathon · Ahouba 3.0 · IIIT Manipur · 21–22 March 2026
> Problem Statement: PS-01

---

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_TEAM/hikari-shinro-ai
cd hikari-shinro-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
cp .env.example .env
# Edit .env — add your GROQ_API_KEY from console.groq.com (free)

# 5. Run
cd app
python app.py

# 6. Open browser
# http://localhost:5000
```

---

## Architecture

```
Voice Input (Whisper STT)
    ↓
LLM Agent Brain (Groq · LLaMA 3)
    ↓ [detect_next labels]
Camera Frame (OpenCV) ──────────────────────────────────┐
    ↓                                                   ↓
YOLO-World Detection              MiDaS Depth Estimation
(open-vocab, zero retraining)     (monocular depth map)
    ↓                                                   ↓
    └───────────────── Quant Engine ────────────────────┘
                    ① Kalman Filter
                    ② A* Pathfinding
                    ③ Euclidean Distance
                    ④ Confidence Threshold
                           ↓
                    LLM Agent (replans)
                      ↓           ↓
                 TTS Output    Flask HUD
               (pyttsx3 audio) (Socket.IO)
```

---

## File Structure

```
hikari-shinro-ai/
├── app/
│   ├── app.py          # Flask server + Socket.IO + main loop
│   ├── agent.py        # LLM agent brain (Groq · LLaMA 3)
│   ├── speech.py       # Whisper STT + pyttsx3 TTS
│   ├── vision.py       # Camera capture + frame pipeline
│   ├── detection.py    # YOLO-World open-vocab detection
│   ├── depth.py        # MiDaS depth estimation
│   └── quant.py        # Kalman · A* · Euclidean · Confidence
├── templates/
│   └── index.html      # Anime-styled HUD web interface
├── requirements.txt
├── .env.example
└── README.md           # ← This file (Technical Disclosure Form)
```

---

# Hackathon Technical Disclosure & Compliance Document

## Team Information

- **Team Name**: _(fill your team name)_
- **Project Title**: Hikari Shinro AI — 光の進路
- **Problem Statement / Track**: PS-01 — Agentic AI for Visually Impaired Users
- **Team Members**: _(fill member names)_
- **Repository Link**: _(fill GitHub URL)_

---

## 1. APIs & External Services Used

### API / Service Entry — 1: Whisper (OpenAI)
- **Provider**: OpenAI (open-source, run locally)
- **Purpose**: Speech → Text transcription
- **API Type**: [x] SDK
- **License**: [x] Open Source (MIT)
- **Rate Limits**: None — runs offline
- **Commercial Use**: [x] Yes

### API / Service Entry — 2: YOLO-World (Ultralytics)
- **Provider**: Tencent AI Lab / Ultralytics
- **Purpose**: Open-vocabulary object detection — no retraining
- **API Type**: [x] SDK
- **License**: [x] Open Source (AGPL-3.0)
- **Rate Limits**: None — runs locally
- **Commercial Use**: [x] Yes

### API / Service Entry — 3: MiDaS (Intel ISL)
- **Provider**: Intel ISL via PyTorch Hub
- **Purpose**: Monocular depth estimation
- **API Type**: [x] SDK
- **License**: [x] Open Source (MIT)
- **Rate Limits**: None — runs locally
- **Commercial Use**: [x] Yes

### API / Service Entry — 4: Groq API (LLaMA 3)
- **Provider**: Groq Inc.
- **Purpose**: LLM agent brain — goal decomposition and navigation reasoning
- **API Type**: [x] REST
- **License**: [x] Free Tier
- **Rate Limits**: 30 req/min, 6000 tokens/min
- **Commercial Use**: [ ] Unclear (free tier)

### API / Service Entry — 5: pyttsx3
- **Provider**: Open Source (PyPI)
- **Purpose**: Offline text-to-speech output
- **API Type**: [x] SDK
- **License**: [x] Open Source (MIT)
- **Rate Limits**: None — fully offline
- **Commercial Use**: [x] Yes

---

## 2. API Keys & Credentials Declaration

- **API Key Source**: [x] Self-generated from official provider
- **Key Storage Method**: [x] Environment Variables (.env file)
- **Hardcoded in Repository**: [x] No

> GROQ_API_KEY stored in .env which is listed in .gitignore.

---

## 3. Open Source Libraries & Frameworks

| Name | Version | Purpose | License |
|------|---------|---------|---------|
| Python | 3.10+ | Core backend | PSF |
| Flask | 3.0.3 | Web server | MIT |
| Flask-SocketIO | 5.3.6 | Real-time HUD | MIT |
| OpenCV | 4.9.0 | Camera + annotation | Apache 2.0 |
| PyTorch | 2.3.0 | ML inference engine | BSD-3 |
| Ultralytics | 8.2.0 | YOLO-World detection | AGPL-3.0 |
| openai-whisper | 20231117 | Speech-to-text | MIT |
| pyttsx3 | 2.90 | Text-to-speech | MIT |
| NumPy | 1.26.4 | Numerical computing | BSD |
| SciPy | 1.13.0 | Kalman filter math | BSD |
| groq | 0.8.0 | Groq Python SDK | Apache 2.0 |
| python-dotenv | 1.0.1 | Environment variables | BSD |

---

## 4. AI Models Used

| Model | Provider | Used For | Access |
|-------|----------|---------|--------|
| YOLO-World-S | Tencent AI Lab | Open-vocab object detection | [x] Local |
| MiDaS Small | Intel ISL | Monocular depth estimation | [x] Local |
| Whisper Base | OpenAI (OSS) | Speech-to-text | [x] Local |
| LLaMA 3 8B | Meta AI via Groq | Agent reasoning + planning | [x] API |

---

## 5. AI Agent Usage Declaration

- **AI Agents Used**: [x] Yes

**Agent: HikariAgent (LLaMA 3 via Groq)**
- **Capabilities Used**: [x] Decision making, [x] Autonomous workflows
- **Human Intervention Level**: [x] High (manual design & logic)

> **Justification**: The LLM agent receives only scene context (detected objects + distances
> from the Quant Engine) and the user's goal. ALL architecture decisions — detection pipeline,
> Kalman filter, A* pathfinding, depth math, and HUD — were entirely human-designed.
> The LLM acts as a language interface for navigation reasoning only.

---

## 6. Restricted / Discouraged Services

- [x] No emergent-style autonomous app builders used
- [x] No full-stack auto-generation agents used
- [x] No prompt-to-product systems used

All code was written manually. AI tools used only for code completion assistance.
Every architectural decision can be explained by any team member.

---

## 7. Originality & Human Contribution

**Human-Designed & Implemented:**
- Full 8-layer system architecture
- Quantitative Engine: Kalman Filter, A* Pathfinding, Euclidean Distance, Confidence Scoring
- LLM agent prompt engineering and reasoning loop
- Anime-styled HUD web interface
- YOLO-World open-vocabulary integration (zero-shot detection)
- Flask + Socket.IO real-time streaming pipeline

**AI-Assisted (Code Completion Only):**
- Docstring suggestions (reviewed and edited)
- Minor syntax completions (all verified)

**What Makes This Unique:**
1. Zero-shot YOLO-World satisfies PS1 "no retraining" requirement directly
2. 4 visible quantitative algorithms make the Quant×AI theme explicit
3. Genuine agentic loop — LLM replans every N frames, no human re-prompting
4. Fully offline capable (only Groq requires internet)
5. Dual output: spoken audio for user + visual HUD for caregivers

---

## 8. Ethical & Compliance Checklist

- [x] No copyrighted data used
- [x] No private datasets
- [x] API usage complies with provider TOS
- [x] No malicious automation
- [x] No AI-generated plagiarism
- [x] No PII collected or stored

---

## 9. Final Declaration

> We confirm all information above is accurate and complete.
> We understand that misrepresentation may lead to disqualification.

**Team Representative Name**: _(fill name)_
**Date**: 21–22 March 2026
