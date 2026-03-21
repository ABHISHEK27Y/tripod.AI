"""
speech.py — Voice I/O Layer
============================
Hikari Shinro AI · CodeRonin · Ahouba 3.0 · IIIT Manipur
Uses sounddevice for mic capture (works on Python 3.13).
"""

import threading
import queue
import logging
import time
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class TTSEngine:

    def __init__(self, rate: int = 155, volume: float = 1.0):
        self._rate      = rate
        self._volume    = volume
        self._queue     = queue.Queue()
        self._thread    = None
        self._engine    = None
        self._running   = False
        self._last_text = ""

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("TTS engine started.")

    def stop(self):
        self._running = False
        self._queue.put(None)

    def speak(self, text: str, priority: bool = False):
        if not text:
            return
        if priority:
            self._last_text = ""        # reset so priority always speaks
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
        if text == self._last_text:
            return
        self._queue.put(text)
        self._last_text = text

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate",   self._rate)
            engine.setProperty("volume", self._volume)
            voices = engine.getProperty("voices")
            if voices:
                eng = [v for v in voices if "english" in v.name.lower() or "en" in v.id.lower()]
                if eng:
                    engine.setProperty("voice", eng[0].id)
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
            logger.error("pyttsx3 not installed.")
        except Exception as e:
            logger.error(f"TTS engine error: {e}")


class WhisperSTT:

    SAMPLE_RATE    = 16000
    RECORD_SECS    = 5
    SILENCE_THRESH = 0.003

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model     = None
        self._loaded    = False

    def load(self):
        if self._loaded:
            return
        try:
            import whisper
            logger.info(f"Loading Whisper model: {self.model_size} ...")
            self._model  = whisper.load_model(self.model_size)
            self._loaded = True
            logger.info("Whisper loaded successfully.")
        except ImportError:
            logger.error("openai-whisper not installed.")
        except Exception as e:
            logger.error(f"Whisper load error: {e}")

    def record_and_transcribe(self) -> Optional[str]:
        if not self._loaded or self._model is None:
            return None
        try:
            import sounddevice as sd
            logger.info(f"Listening for {self.RECORD_SECS}s ...")
            audio_2d = sd.rec(
                int(self.SAMPLE_RATE * self.RECORD_SECS),
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio_2d.flatten()

            rms = float(np.sqrt(np.mean(audio ** 2)))
            logger.debug(f"Audio RMS: {rms:.5f} (threshold: {self.SILENCE_THRESH})")

            if rms < self.SILENCE_THRESH:
                logger.debug("Silence — skipping.")
                return None

            result = self._model.transcribe(audio, fp16=False, language="en")
            text   = result.get("text", "").strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text if text else None

        except ImportError:
            logger.error("sounddevice not installed.")
            return None
        except Exception as e:
            logger.warning(f"STT error: {e}")
            return None

    def transcribe_file(self, path: str) -> Optional[str]:
        if not self._loaded:
            return None
        try:
            result = self._model.transcribe(path, fp16=False, language="en")
            return result.get("text", "").strip()
        except Exception as e:
            logger.warning(f"File transcription error: {e}")
            return None


class VoiceController:

    def __init__(
        self,
        on_command:   Callable[[str], None],
        tts_rate:     int = 155,
        whisper_size: str = "base",
        tts_engine=None,
    ):
        self.tts            = tts_engine if tts_engine is not None else TTSEngine(rate=tts_rate)
        self.stt            = WhisperSTT(model_size=whisper_size)
        self._on_command    = on_command
        self._listen_thread = None
        self._listening     = False

    def start(self):
        if not self.tts._running:
            self.tts.start()
        self.tts.speak("Hikari Shinro AI is ready. Please speak your navigation goal.")
        threading.Thread(target=self._load_stt, daemon=True).start()

    def _load_stt(self):
        self.stt.load()
        self.tts.speak("Voice recognition ready. Speak your goal now.")
        self.start_listening()

    def start_listening(self):
        if self._listening:
            return
        self._listening     = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        logger.info("Voice listen loop started.")

    def stop_listening(self):
        self._listening = False

    def _listen_loop(self):
        while self._listening:
            try:
                text = self.stt.record_and_transcribe()
                if text:
                    logger.info(f"Voice command: '{text}'")
                    self._on_command(text)
            except Exception as e:
                logger.warning(f"Listen loop error: {e}")
            time.sleep(0.05)

    def speak(self, text: str, priority: bool = False):
        self.tts.speak(text, priority=priority)

    def stop(self):
        self.stop_listening()
        self.tts.stop()