FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# - libgl1, libglib2.0-0: for OpenCV
# - portaudio19-dev: for PyAudio (Microphone STT)
# - ffmpeg: for OpenAI Whisper audio processing
# - espeak, alsa-utils: for pyttsx3 (Offline Voice TTS)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    espeak \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Flask HUD Socket.IO Port
EXPOSE 5000

# Set Python path
ENV PYTHONPATH=/app

# Run the VisionGuide Backend
CMD ["python", "main.py"]
