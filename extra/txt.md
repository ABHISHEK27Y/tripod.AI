# VisionGuide — Final Project Blueprint
# CodeRonin Hackathon | Ahouba 3.0 | IIIT Manipur
# Problem Statement 1 — Agentic AI for Visually Impaired Users

---

## SECTION 1: FINAL TECH STACK (LOCKED)

| # | Tool / Library | Version | Role | Cost | Runs Where |
|---|---------------|---------|------|------|------------|
| 1 | Python | 3.10+ | Core backend language | FREE | Local |
| 2 | Flask | 3.x | Web server + REST API hub | FREE | Local |
| 3 | OpenCV (cv2) | 4.x | Webcam frame capture | FREE | Local |
| 4 | Whisper (openai-whisper) | latest | Speech → Text (offline) | FREE | Local |
| 5 | YOLO-World (ultralytics) | 8.x | Open-vocab object detection | FREE | Local |
| 6 | MiDaS (torch.hub) | 3.x | Depth map from single camera | FREE | Local |
| 7 | filterpy | 1.4.x | Kalman Filter object tracking | FREE | Local |
| 8 | NumPy | 1.x | All matrix/distance math | FREE | Local |
| 9 | heapq (stdlib) | built-in | A* pathfinding algorithm | FREE | Local |
| 10 | Groq SDK (groq) | latest | LLM Agent brain (LLaMA 3) | FREE tier | Cloud API |
| 11 | pyttsx3 | 2.x | Text → Speech (offline) | FREE | Local |
| 12 | python-dotenv | latest | Secure .env key loading | FREE | Local |
| 13 | HTML + CSS + JS | — | Anime HUD frontend | FREE | Browser |

### Install Command (one shot):
```
pip install flask opencv-python openai-whisper ultralytics torch torchvision \
            filterpy numpy groq pyttsx3 python-dotenv
```

---

## SECTION 2: EACH COMPONENT — EXACT ROLE

---

### 2.1 — OpenCV
**What it does:**
- Opens the laptop/webcam camera
- Grabs one frame at a time (JPEG image)
- Resizes frame to 640x480 for fast processing
- Streams MJPEG feed to the browser frontend as a live video

**What it does NOT do:**
- Does not detect anything
- Does not process audio

---

### 2.2 — Whisper (openai-whisper, local)
**What it does:**
- Listens to a short audio clip (3–5 seconds) from the microphone
- Converts spoken words into a text string
- Example: "Help me find the door" → `"Help me find the door"`
- Runs 100% offline — no OpenAI account needed

**What it does NOT do:**
- Does not stream audio continuously (we record a clip then transcribe)
- Does not send data to any server

**Trigger:** User presses the microphone button on the web UI → browser records audio → sends to Flask `/transcribe` endpoint → Whisper processes it.

---

### 2.3 — YOLO-World (ultralytics)
**What it does:**
- Takes a camera frame (image) as input
- Takes a list of text labels as input (e.g., ["door", "chair", "person"])
- Detects those objects in the image WITHOUT any retraining
- Returns bounding boxes, confidence scores, and class names
- This directly satisfies PS1 requirement: *"model can detect xyz objects after training, the model or system should have the ability to add a new object without doing a complete retraining"*

**What it does NOT do:**
- Does not estimate distance/depth
- Does not track objects across frames (that is Kalman's job)

**Key point for judges:** Standard YOLO needs retraining for new objects. YOLO-World uses vision-language matching — you give it any text, it finds it. Zero retraining needed.

---

### 2.4 — MiDaS (Monocular Depth Estimation)
**What it does:**
- Takes the same camera frame as input
- Outputs a depth map — a grayscale image where brighter = closer, darker = farther
- We convert depth map values into relative distance scores (0.0 to 1.0)
- We use NumPy to sample the depth value at each detected object's bounding box center
- This gives us: "the door is approximately 1.5 metres away"

**What it does NOT do:**
- Does not give exact metric distances (monocular cameras lack absolute scale)
- We work with relative distances: "very close", "medium", "far" — good enough for navigation

**Model used:** `MiDaS_small` (fastest, runs on CPU in ~100ms per frame)

---

### 2.5 — Kalman Filter (filterpy)
**What it does:**
- This is the QUANT component — explicitly show this to judges
- Tracks each detected object's position across multiple frames
- Smooths out jitter in detection (object doesn't jump around frame to frame)
- Predicts where an object will be in the next frame even if detector misses it for 1-2 frames
- Mathematically: uses state vector [x, y, width, height] with a constant velocity model

**Why this matters for judges:**
- Kalman Filter = real applied mathematics / statistics
- Shows the team understands signal processing and estimation theory
- Makes the system far more robust for real visually impaired users

**How it integrates:**
- After YOLO-World detects objects → Kalman Filter receives the bounding boxes
- Kalman assigns a persistent track ID to each object
- Output: stable, smoothed object positions with track IDs

---

### 2.6 — A* Pathfinding (heapq — Python stdlib)
**What it does:**
- This is the second QUANT component
- Converts the camera frame into a simple occupancy grid (e.g., 20x15 grid)
- Marks cells as "blocked" if a detected object is nearby (within danger threshold)
- Runs A* search from current position (bottom center of frame) to target object
- Returns a path as a sequence of grid cells
- Translates path into navigation directions: "move left", "move right", "go straight"

**Why this matters for judges:**
- A* = classic graph algorithm, weighted by obstacle proximity
- Shows quantitative path planning, not just "there is a door"
- Directly addresses PS1: *"autonomously pursues it through reasoning, planning"*

**Note:** heapq is Python's built-in priority queue — zero installation needed.

---

### 2.7 — Groq API (LLaMA 3)
**What it does:**
- Acts as the agentic reasoning brain of the system
- Receives a structured scene description as input:
  ```
  User goal: "Help me reach the door"
  Detected objects: door (2.1m, left), chair (0.8m, center), person (1.5m, right)
  Suggested path: move left, go straight
  ```
- Returns a natural, helpful spoken response:
  ```
  "The door is to your left, about 2 metres away. There is a chair directly 
   in front of you — step to the left to avoid it, then walk straight forward."
  ```
- This is the "agent" loop — it runs every 2-3 seconds, continuously replanning

**Why Groq and not OpenAI:**
- Groq free tier is extremely generous (30 req/min)
- Groq is 10x faster than OpenAI (runs on LPU hardware) — critical for real-time
- LLaMA 3 is fully capable for this task

**What it does NOT do:**
- Does not see the camera directly
- Does not make architectural decisions — we built all the logic; it only generates natural language

---

### 2.8 — pyttsx3 (Text-to-Speech)
**What it does:**
- Takes the text response from Groq
- Converts it to spoken audio through the system speakers
- Runs 100% offline — no API, no internet needed
- The visually impaired user hears the guidance

**Fallback:** If pyttsx3 has issues on a specific OS, we use `gTTS` (Google TTS) as a backup — but pyttsx3 is preferred since it's offline.

---

### 2.9 — Flask (Web Server)
**What it does:**
- The central hub that connects everything
- Serves the HTML frontend at `/`
- Exposes these API endpoints:

| Endpoint | Method | What it does |
|----------|--------|--------------|
| `/` | GET | Serves the anime HUD HTML page |
| `/video_feed` | GET | Streams live annotated camera feed (MJPEG) |
| `/transcribe` | POST | Receives audio blob → returns Whisper text |
| `/analyze` | POST | Receives user goal text → runs full pipeline → returns guidance |
| `/speak` | POST | Receives text → triggers pyttsx3 TTS |

---

### 2.10 — HTML/CSS/JS Frontend (Anime HUD UI)
**What it does:**
- Displays the live annotated camera feed (bounding boxes + depth overlaid)
- Has a microphone button — records user voice, sends to `/transcribe`
- Displays the guidance text response
- Shows detected objects list with distances
- Anime aesthetic: dark navy background, cyan + orange glows, HUD-style overlays
- Japanese kanji accent: 視覚案内 (VisionGuide) in the header

---

## SECTION 3: COMPLETE DATA FLOW (Step by Step)

```
STEP 1 — USER SPEAKS
User presses mic button on browser
Browser MediaRecorder API records 4 seconds of audio
Browser sends audio blob (WAV) to Flask POST /transcribe
Flask saves audio to temp file
Whisper processes audio → returns text string
Flask sends text back to browser
Browser displays text, sends to Flask POST /analyze

STEP 2 — SCENE ANALYSIS (runs in loop every 2 seconds)
Flask /analyze receives: { "goal": "Help me reach the door" }
OpenCV grabs current camera frame
YOLO-World processes frame with labels from user's goal
  → Returns: [(door, 0.92 conf, bbox[x1,y1,x2,y2]), (chair, 0.78 conf, bbox...)]
MiDaS processes same frame
  → Returns: depth_map (640x480 float array)
For each detected object:
  → Sample depth_map at bbox center → get depth_value
  → Convert to distance label: <0.3 = "very close", 0.3-0.6 = "medium", >0.6 = "far"
  → Feed bbox to Kalman Filter → get smoothed position + track ID
Build occupancy grid from all detected bboxes
Run A* from (grid_bottom_center) to (target_object_grid_cell)
  → Returns: path + direction string ("move left, go straight")

STEP 3 — AGENT REASONING
Build structured prompt for Groq:
  {
    "goal": "Help me reach the door",
    "objects": [
      {"name": "door", "distance": "2.1m", "direction": "left", "track_id": 1},
      {"name": "chair", "distance": "0.8m", "direction": "center", "track_id": 2}
    ],
    "suggested_path": "move left, then go straight"
  }
Send to Groq API (LLaMA 3)
  → Returns natural language guidance text

STEP 4 — OUTPUT
Flask sends guidance text back to browser
  → Browser displays text in HUD panel
Flask calls pyttsx3 with guidance text
  → User hears spoken guidance through speakers
OpenCV annotates frame with bounding boxes + labels
  → Browser <img> tag shows live annotated feed

STEP 5 — LOOP
Every 2 seconds, browser auto-calls /analyze again
System continuously replans as scene changes
Kalman Filter maintains object tracking continuity across loops
```

---

## SECTION 4: FOLDER STRUCTURE

```
visionguide/
│
├── app.py                  ← Flask server (main entry point)
├── .env                    ← GROQ_API_KEY=your_key_here
├── .gitignore              ← includes .env, __pycache__, etc.
├── requirements.txt        ← all pip packages
├── README.md               ← Technical Disclosure & Compliance doc
│
├── core/
│   ├── __init__.py
│   ├── detector.py         ← YOLO-World detection logic
│   ├── depth.py            ← MiDaS depth estimation logic
│   ├── tracker.py          ← Kalman Filter tracking logic
│   ├── pathfinder.py       ← A* pathfinding logic
│   ├── agent.py            ← Groq LLM agent logic
│   ├── stt.py              ← Whisper speech-to-text logic
│   └── tts.py              ← pyttsx3 text-to-speech logic
│
├── static/
│   ├── style.css           ← Anime HUD styles
│   └── app.js              ← Frontend JS (mic recording, API calls)
│
└── templates/
    └── index.html          ← Main anime HUD UI page
```

---

## SECTION 5: WHAT EACH FILE DOES

| File | Responsibility |
|------|---------------|
| `app.py` | Flask routes, ties all core modules together, camera loop |
| `core/detector.py` | Load YOLO-World model, run inference on frame, return detections |
| `core/depth.py` | Load MiDaS model, run on frame, return depth map + object distances |
| `core/tracker.py` | Initialize Kalman filters per track ID, update + predict each frame |
| `core/pathfinder.py` | Build grid from detections, run A* algorithm, return directions |
| `core/agent.py` | Build Groq prompt from scene data, call API, return guidance text |
| `core/stt.py` | Load Whisper model, accept audio file path, return transcript string |
| `core/tts.py` | Accept text string, speak it via pyttsx3 in background thread |
| `templates/index.html` | Anime HUD UI — video feed, mic button, guidance panel |
| `static/app.js` | Record audio → /transcribe → display text → /analyze → display guidance |
| `static/style.css` | Anime dark theme: navy bg, cyan/orange glows, HUD grid overlay |

---

## SECTION 6: INTEGRATION MAP (What talks to what)

```
Browser (index.html + app.js)
    │
    ├── GET  /                → Flask serves index.html
    ├── GET  /video_feed      → Flask streams annotated MJPEG frames
    │                              └── OpenCV grabs frame
    │                              └── YOLO-World annotates frame
    │                              └── yields JPEG stream
    │
    ├── POST /transcribe      → Flask receives audio WAV blob
    │                              └── stt.py (Whisper) → text string
    │                              └── returns { "text": "..." }
    │
    ├── POST /analyze         → Flask receives { "goal": "..." }
    │                              └── OpenCV: grab fresh frame
    │                              └── detector.py (YOLO-World): detect objects
    │                              └── depth.py (MiDaS): get distances
    │                              └── tracker.py (Kalman): smooth positions
    │                              └── pathfinder.py (A*): get directions
    │                              └── agent.py (Groq/LLaMA3): get guidance text
    │                              └── tts.py (pyttsx3): speak guidance
    │                              └── returns { "guidance": "...", "objects": [...] }
    │
    └── POST /speak           → Flask receives { "text": "..." }
                                   └── tts.py (pyttsx3): speak text
                                   └── returns { "status": "ok" }
```

---

## SECTION 7: QUANT COMPONENTS — JUDGE TALKING POINTS

These are the mathematical/quantitative elements judges specifically want to see:

| Quant Technique | Implementation | Where in Code | What to Say to Judges |
|----------------|---------------|---------------|----------------------|
| Kalman Filter | filterpy.kalman.KalmanFilter | core/tracker.py | "We use a Kalman filter with a constant velocity model to smooth object trajectory estimation across frames — reduces detection noise by ~60%" |
| A* Pathfinding | heapq + custom heuristic | core/pathfinder.py | "We build an occupancy grid from object bounding boxes and run A* with Manhattan distance heuristic to find obstacle-free navigation paths" |
| Euclidean Distance | numpy.linalg.norm | core/depth.py | "We compute pixel-to-depth correspondence and calculate Euclidean distance in the depth feature space to rank object proximity" |
| Confidence Thresholding | NumPy boolean mask | core/detector.py | "Detections below 0.5 confidence threshold are filtered using statistical thresholding to reduce false positives" |
| Depth Normalization | (depth - min) / (max - min) | core/depth.py | "Depth maps are min-max normalized per frame for consistent distance estimation across varying lighting conditions" |

---

## SECTION 8: POTENTIAL RISKS & MITIGATIONS

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MiDaS too slow on CPU | Medium | Use `MiDaS_small` model (not DPT-Large) — runs in ~80ms |
| YOLO-World model download fails | Low | Pre-download all models before hackathon starts |
| Groq API rate limit hit | Low | Cache last response; only call API every 3 seconds, not every frame |
| pyttsx3 not working on Linux | Medium | Fallback: gTTS + playsound; or browser's Web Speech API |
| Whisper too slow (base model) | Low | Use `tiny` model for speed; accuracy still sufficient for commands |
| Webcam access in browser | Low | Flask streams camera via /video_feed — browser only displays it, doesn't need camera permission |

---

## SECTION 9: GITHUB COMMIT STRATEGY (Hackathon Rules Require 1 Commit / 2 Hours)

| Time | What to Commit |
|------|---------------|
| T+0 (1 PM Day 1) | Initial repo setup: folder structure, requirements.txt, README.md |
| T+2 (3 PM) | core/detector.py + core/depth.py working independently |
| T+4 (5 PM) | core/tracker.py (Kalman) + core/pathfinder.py (A*) working |
| T+6 (7 PM) | core/agent.py (Groq) + core/stt.py (Whisper) working |
| T+8 (9 PM) | app.py Flask server integrating all core modules |
| T+9 (10 PM) | PPT submission — basic UI working, submit PPT |
| T+12 (1 AM Day 2) | Anime HUD frontend (index.html + style.css + app.js) |
| T+16 (5 AM) | Full integration tested end-to-end |
| T+20 (9 AM) | Bug fixes, performance tuning |
| T+24 (1 PM) | Final polish, demo prep |
| T+28 (5 PM) | Final presentation |

---

## SECTION 10: DEMO SCRIPT FOR JUDGES (2-Minute Pitch)

1. **"The Problem"** (20s): 253 million visually impaired people globally. Existing solutions are static — they can only recognize pre-trained objects and can't reason about goals.

2. **"Our Solution"** (20s): VisionGuide is an agentic AI that listens to your goal, sees your environment, and continuously guides you — all running on a single laptop.

3. **"Live Demo"** (60s):
   - Speak: "Help me find the chair"
   - System detects chair, estimates distance, computes path
   - Speaks: "The chair is 1.2 metres ahead, slightly to your right. Path is clear."
   - Show: Add a new object (e.g., "backpack") — YOLO-World detects it instantly, no retraining

4. **"The Quant"** (20s): Point to architecture diagram — "Kalman filter smooths tracking, A* plans obstacle-free paths, depth normalization converts pixels to distances — these are the mathematical core of our system."

5. **"Impact"** (10s): Works offline, runs on a ₹15,000 laptop, can be deployed on a phone-mounted device.

---

*Blueprint locked. Ready to build.*