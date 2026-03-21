# Hackathon Technical Disclosure & Compliance Document

## Team Information

- **Team Name**:Tripod.AI
- **Project Title**: Hikari-Shinro-AI
- **Problem Statement / Track**: Problem Statement 1 — Agentic AI System for Visually Impaired Users (Quant × AI)
- **Team Members**:Abhishek Yadav,Apoorva Yadav,Atul Yadav
- **Repository Link (if public)**:https://github.com/ABHISHEK27Y/tripod.AI.git
- **Deployment Link (if applicable)**:.Will be Deployed later

---

## 1. APIs & External Services Used

---

### API / Service Entry — 1

- **API / Service Name**: Whisper (OpenAI)
- **Provider / Organization**: OpenAI (open-source model, run locally)
- **Purpose in Project**: Converts spoken user voice commands into text (Speech-to-Text)
- **API Type**:
  - [ ] REST
  - [ ] GraphQL
  - [x] SDK
  - [ ] Other (specify)
- **License Type**:
  - [x] Open Source
  - [ ] Free Tier
  - [ ] Academic
  - [ ] Commercial
- **License Link / Documentation URL**: https://github.com/openai/whisper
- **Rate Limits (if any)**: None — runs fully offline/locally
- **Commercial Use Allowed**:
  - [x] Yes
  - [ ] No
  - [ ] Unclear

---

### API / Service Entry — 2

- **API / Service Name**: YOLO-World
- **Provider / Organization**: Tencent AI Lab / Ultralytics
- **Purpose in Project**: Open-vocabulary real-time object detection — detects any object by text label without retraining
- **API Type**:
  - [ ] REST
  - [ ] GraphQL
  - [x] SDK
  - [ ] Other (specify)
- **License Type**:
  - [x] Open Source
  - [ ] Free Tier
  - [ ] Academic
  - [ ] Commercial
- **License Link / Documentation URL**: https://github.com/AILab-CVC/YOLO-World
- **Rate Limits (if any)**: None — runs locally
- **Commercial Use Allowed**:
  - [x] Yes
  - [ ] No
  - [ ] Unclear

---

### API / Service Entry — 3

- **API / Service Name**: MiDaS (Monocular Depth Estimation)
- **Provider / Organization**: Intel ISL / PyTorch Hub
- **Purpose in Project**: Estimates depth from a single RGB camera frame — produces a depth map for distance calculation
- **API Type**:
  - [ ] REST
  - [ ] GraphQL
  - [x] SDK
  - [ ] Other (specify)
- **License Type**:
  - [x] Open Source
  - [ ] Free Tier
  - [ ] Academic
  - [ ] Commercial
- **License Link / Documentation URL**: https://github.com/isl-org/MiDaS
- **Rate Limits (if any)**: None — runs locally
- **Commercial Use Allowed**:
  - [x] Yes
  - [ ] No
  - [ ] Unclear

---

### API / Service Entry — 4

- **API / Service Name**: Groq API (LLaMA 3)
- **Provider / Organization**: Groq Inc.
- **Purpose in Project**: LLM Agent brain — decomposes high-level user goals into subtasks, performs autonomous reasoning and planning without re-prompting
- **API Type**:
  - [x] REST
  - [ ] GraphQL
  - [ ] SDK
  - [ ] Other (specify)
- **License Type**:
  - [ ] Open Source
  - [x] Free Tier
  - [ ] Academic
  - [ ] Commercial
- **License Link / Documentation URL**: https://console.groq.com/docs
- **Rate Limits (if any)**: Free tier — 30 requests/minute, 6000 tokens/minute
- **Commercial Use Allowed**:
  - [ ] Yes
  - [ ] No
  - [x] Unclear (free tier — non-commercial use)

---

### API / Service Entry — 5

- **API / Service Name**: pyttsx3
- **Provider / Organization**: Open Source (PyPI)
- **Purpose in Project**: Text-to-Speech — converts agent guidance text into spoken audio for the visually impaired user
- **API Type**:
  - [ ] REST
  - [ ] GraphQL
  - [x] SDK
  - [ ] Other (specify)
- **License Type**:
  - [x] Open Source
  - [ ] Free Tier
  - [ ] Academic
  - [ ] Commercial
- **License Link / Documentation URL**: https://pypi.org/project/pyttsx3/
- **Rate Limits (if any)**: None — fully offline
- **Commercial Use Allowed**:
  - [x] Yes
  - [ ] No
  - [ ] Unclear

---

## 2. API Keys & Credentials Declaration

- **API Key Source**:
  - [x] Self-generated from official provider
  - [ ] Hackathon-provided key
  - [ ] Open / Keyless API
- **Key Storage Method**:
  - [x] Environment Variables
  - [ ] Secure Vault
  - [ ] Backend-only (not exposed)
- **Hardcoded in Repository**:
  - [ ] Yes
  - [x] No

> **Note**: Groq API key is stored in a `.env` file which is listed in `.gitignore`. No keys are hardcoded or committed to the repository.

---

## 3. Open Source Libraries & Frameworks

| Name | Version | Purpose | License |
|------|---------|---------|---------|
| Python | 3.10+ | Core backend language | PSF |
| Flask | 3.x | Web server and REST API | MIT |
| OpenCV | 4.x | Camera frame capture and image processing | Apache 2.0 |
| PyTorch | 2.x | Deep learning inference engine | BSD-3 |
| Ultralytics (YOLO-World) | 8.x | Open-vocabulary object detection | AGPL-3.0 |
| MiDaS | 3.x | Monocular depth estimation | MIT |
| Whisper | Latest | Speech-to-Text transcription | MIT |
| pyttsx3 | 2.x | Offline Text-to-Speech | MIT |
| NumPy | 1.x | Numerical computation and matrix operations | BSD |
| SciPy | 1.x | Kalman filter and scientific computing | BSD |
| groq | Latest | Groq API Python SDK (LLM agent) | Apache 2.0 |
| python-dotenv | Latest | Secure environment variable management | BSD |
| HTML / CSS / JS | — | Anime-styled frontend HUD UI | — |

---

## 4. AI Models, Tools & Agents Used

### AI Models

**Model 1 — Object Detection**
- **Model Name**: YOLO-World (Open-Vocabulary)
- **Provider**: Tencent AI Lab
- **Used For**: Real-time detection of any object specified by the user — no retraining required
- **Access Method**:
  - [ ] API
  - [x] Local Model
  - [ ] Hosted Platform

**Model 2 — Depth Estimation**
- **Model Name**: MiDaS (DPT-Large)
- **Provider**: Intel ISL
- **Used For**: Estimating metric depth from single RGB frames to calculate real-world distances
- **Access Method**:
  - [ ] API
  - [x] Local Model
  - [ ] Hosted Platform

**Model 3 — Speech-to-Text**
- **Model Name**: Whisper (base / small)
- **Provider**: OpenAI (open source)
- **Used For**: Transcribing user's spoken natural language goals into text for the LLM agent
- **Access Method**:
  - [ ] API
  - [x] Local Model
  - [ ] Hosted Platform

**Model 4 — LLM Agent Brain**
- **Model Name**: LLaMA 3 (via Groq)
- **Provider**: Meta AI / Groq
- **Used For**: Agentic goal decomposition, subtask planning, and spoken guidance generation
- **Access Method**:
  - [x] API
  - [ ] Local Model
  - [ ] Hosted Platform

---

### AI Tools / Platforms

- **Tool Name**: Groq API
- **Role in Project**: Powers the LLM agent reasoning loop — receives scene description and user goal, returns structured navigation guidance
- **Level of Dependency**:
  - [ ] Assistive
  - [x] Core Logic
  - [ ] Entire Solution

---

## 5. AI Agent Usage Declaration

- **AI Agents Used** (if any):
  - [ ] None
  - [x] Yes (listed below)

### Agent Details

- **Agent Name / Platform**: Custom LLM Agent (LLaMA 3 via Groq API)
- **Capabilities Used**:
  - [ ] Code generation
  - [ ] Full app scaffolding
  - [x] Decision making
  - [x] Autonomous workflows
- **Human Intervention Level**:
  - [x] High (manual design & logic)
  - [ ] Medium
  - [ ] Low (mostly autonomous)

> **Justification**: The LLM agent only receives scene context (detected objects + distances) and a user goal. All architecture decisions — detection pipeline, Kalman filter tracking, A\* path planning, depth math, and the web UI — were entirely human-designed and implemented by the team. The LLM acts purely as a language interface, not an architect.

---

## 6. Restricted / Discouraged AI Services

We confirm that **none of the following restricted services were used**:

- [x] No emergent-style autonomous app builders used
- [x] No full-stack auto-generation agents used
- [x] No prompt-to-product systems used

All code was written manually by team members. AI tools (GitHub Copilot / Claude) were used only for **code completion assistance** — all logic, architecture, and design decisions were made by humans and can be fully explained by any team member.

---

## 7. Originality & Human Contribution Statement

### Human-Designed & Implemented
- Full system architecture (pipeline from voice → detection → quant → speech output)
- Quantitative Engine: Kalman Filter for object tracking, A\* algorithm for path planning, Euclidean distance calculations on MiDaS depth maps, confidence score thresholding
- LLM agent prompt design and reasoning loop
- Anime-themed HUD web interface (Flask + HTML/CSS/JS)
- Integration of all models into a unified real-time pipeline
- GitHub repository structure and documentation

### AI-Assisted (Assistive Only)
- Code completion suggestions (accepted/modified by team members)
- Documentation drafting (reviewed and edited by team)

### What Makes VisionGuide Unique
1. **Zero-shot object detection** — YOLO-World detects any object the user asks about without any fine-tuning or retraining, directly satisfying the PS1 requirement
2. **Quantitative Navigation Engine** — Explicit Kalman filter tracking + A\* path planning makes this genuinely quant-heavy, not just an AI wrapper
3. **End-to-end agentic loop** — The system continuously replans without user re-prompting, fulfilling the "agentic" requirement of PS1
4. **Fully offline-capable** — Whisper, YOLO-World, MiDaS, and pyttsx3 all run locally; only the LLM call requires internet, making it robust in low-connectivity environments

---

## 8. Ethical, Legal & Compliance Checklist

- [x] No copyrighted data used without permission
- [x] No leaked or private datasets
- [x] API usage complies with provider Terms of Service
- [x] No malicious automation or scraping
- [x] No AI-generated plagiarism
- [x] No personally identifiable information (PII) collected or stored
- [x] Camera access used only during active session with user consent

---

## 9. Final Declaration

> We confirm that all information provided above is accurate and complete.
> We understand that misrepresentation, plagiarism, or use of restricted tools
> without disclosure may lead to immediate disqualification.
> All team members have reviewed and agree to this declaration.

**Team Representative Name**: Abhishek Yadav

**Date**: 21–22 March 2026

**Hackathon**: CodeRonin — Ahouba 3.0 | IIIT Manipur

---

*This document is submitted as part of the mandatory Technical Disclosure & Compliance requirement for CodeRonin Hackathon, Ahouba 3.0, IIIT Manipur.*
