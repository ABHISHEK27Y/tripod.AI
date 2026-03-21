"""
speech.py — Voice I/O Layer
============================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur

Whisper STT  — offline speech-to-text (microphone → text)
pyttsx3 TTS  — offline text-to-speech (text → audio output)

Both run fully locally — no API calls, no internet required.
"""

import threading
import queue
import logging
import time
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────
# TEXT-TO-SPEECH (pyttsx3) — always loaded, very fast
# ─────────────────────────────────────────────────────────
class TTSEngine:
    """
    Offline text-to-speech via pyttsx3.
    Queues speech requests so the pipeline never blocks.
    """

    def __init__(self, rate: int = 155, volume: float = 1.0):
        self._rate    = rate
        self._volume  = volume
        self._queue: queue.Queue = queue.Queue()
        self._thread  = None
        self._engine  = None
        self._running = False
        self._last_text = ""

    def start(self):
        """Start TTS worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("TTS engine started.")

    def stop(self):
        self._running = False
        self._queue.put(None)

    def speak(self, text: str, priority: bool = False):
        """
        Queue text for speaking.
        If priority=True, clears queue first (for urgent alerts).
        Skips if same as last spoken text (avoids repetition).
        """
        if not text or text == self._last_text:
            return
        if priority:
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        self._queue.put(text)
        self._last_text = text

    def _worker(self):
        """Worker thread: initialises pyttsx3 and processes queue."""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)

            # Choose a clear voice if available
            voices = engine.getProperty("voices")
            if voices:
                # Prefer English voice
                eng_voices = [v for v in voices if "english" in v.name.lower() or "en" in v.id.lower()]
                if eng_voices:
                    engine.setProperty("voice", eng_voices[0].id)

            self._engine = engine
            logger.info("pyttsx3 engine initialised.")

            while self._running:
                try:
                    text = self._queue.get(timeout=0.5)
                    if text is None:
                        break
                    engine.say(text)
                    engine.runAndWait()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.warning(f"TTS speak error: {e}")

        except ImportError:
            logger.error("pyttsx3 not installed. Run: pip install pyttsx3")
        except Exception as e:
            logger.error(f"TTS engine error: {e}")


# ─────────────────────────────────────────────────────────
# SPEECH-TO-TEXT (Whisper) — loaded on demand
# ─────────────────────────────────────────────────────────
class WhisperSTT:
    """
    Offline speech-to-text using OpenAI Whisper (local model).
    Records from microphone, transcribes on demand.
    """

    SAMPLE_RATE   = 16000   # Whisper expects 16kHz
    RECORD_SECS   = 4       # record window in seconds
    SILENCE_THRESH = 0.008  # amplitude below this = silence

    def __init__(self, model_size: str = "base"):
        """
        model_size options:
          "tiny"   — fastest,  ~75MB,  lower accuracy
          "base"   — balanced, ~145MB  ← recommended for hackathon
          "small"  — better,   ~465MB
        """
        self.model_size = model_size
        self._model     = None
        self._loaded    = False

    def load(self):
        """Load Whisper model (downloads on first call, ~145MB for base)."""
        if self._loaded:
            return
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_size} ...")
            self._model  = whisper.load_model(self.model_size)
            self._loaded = True
            logger.info("Whisper loaded.")
        except ImportError:
            logger.error("openai-whisper not installed. Run: pip install openai-whisper")
        except Exception as e:
            logger.error(f"Whisper load error: {e}")

    def record_and_transcribe(self) -> Optional[str]:
        """
        Record RECORD_SECS of audio from default microphone,
        run Whisper transcription, return text string.
        Returns None if recording or transcription fails.
        """
        if not self._loaded or self._model is None:
            return None

        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.SAMPLE_RATE,
                input=True,
                frames_per_buffer=1024,
            )

            logger.info("Listening...")
            frames = []
            total_frames = int(self.SAMPLE_RATE / 1024 * self.RECORD_SECS)
            for _ in range(total_frames):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))

            stream.stop_stream()
            stream.close()
            pa.terminate()

            audio = np.concatenate(frames)

            # Skip transcription if mostly silence
            if float(np.abs(audio).mean()) < self.SILENCE_THRESH:
                logger.debug("Silence detected, skipping transcription.")
                return None

            result = self._model.transcribe(audio, fp16=False, language="en")
            text = result.get("text", "").strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text if text else None

        except ImportError:
            logger.error("pyaudio not installed. Run: pip install pyaudio")
            return None
        except Exception as e:
            logger.warning(f"STT error: {e}")
            return None

    def transcribe_file(self, path: str) -> Optional[str]:
        """Transcribe an audio file (for testing without microphone)."""
        if not self._loaded:
            return None
        try:
            result = self._model.transcribe(path, fp16=False, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logger.warning(f"File transcription error: {e}")
            return None


# ─────────────────────────────────────────────────────────
# VOICE CONTROLLER — combines STT + TTS with listen loop
# ─────────────────────────────────────────────────────────
class VoiceController:
    """
    High-level voice I/O controller.
    Runs a background thread that listens for voice commands
    and calls a callback with the transcribed text.
    """

    def __init__(
        self,
        on_command: Callable[[str], None],
        tts_rate: int = 155,
        whisper_size: str = "base",
    ):
        self.tts = TTSEngine(rate=tts_rate)
        self.stt = WhisperSTT(model_size=whisper_size)
        self._on_command   = on_command
        self._listen_thread = None
        self._listening     = False

    def start(self):
        """Load models and start TTS. STT loads separately."""
        self.tts.start()
        self.tts.speak("Hikari Shinro AI is ready. Please speak your navigation goal.")
        threading.Thread(target=self._load_stt, daemon=True).start()

    def _load_stt(self):
        self.stt.load()
        self.tts.speak("Voice recognition ready.")
        self.start_listening()

    def start_listening(self):
        """Start continuous listen loop in background thread."""
        if self._listening:
            return
        self._listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()

    def stop_listening(self):
        self._listening = False

    def _listen_loop(self):
        """Continuously listens and calls callback when speech detected."""
        logger.info("Voice listen loop started.")
        while self._listening:
            text = self.stt.record_and_transcribe()
            if text:
                self._on_command(text)
            time.sleep(0.1)

    def speak(self, text: str, priority: bool = False):
        self.tts.speak(text, priority=priority)

    def stop(self):
        self.stop_listening()
        self.tts.stop()
