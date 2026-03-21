## Local Setup

```bash
python3 -m venv .venv
```

Activate the env:
```bash
source .venv/bin/activate # linux
```

```cmd
.\\venv\\Scripts\\activate.bat  # Windows
```

Install packages:
```bash
pip install -r requirements.txt
```

Environment Setup:
```bash
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

Run Backend:
```bash
python main.py
```
*The Flask HUD stream will be available at `http://localhost:5000`*

---

## Docker

### Build
```bash
docker build -t tripod-ai .
```

### Run
```bash
docker run --rm \
  --device /dev/video0:/dev/video0 \
  --device /dev/snd \
  -p 5000:5000 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --env-file .env \
  tripod-ai
```

**Note:** For GUI display (OpenCV window), X11 forwarding is required. Allow local connections:
```bash
xhost +local:docker
```

### Headless Mode
For servers without display, modify `Dockerfile` CMD or disable visualization in code.

