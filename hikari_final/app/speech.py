"""speech.py — Voice I/O · Hikari Shinro AI"""
import threading
import queue
import logging
import time
import numpy as np
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class TTSEngine:

    def __init__(self, rate=160, volume=1.0):
        self._rate      = rate
        self._volume    = volume
        self._queue     = queue.Queue()
        self._thread    = None
        self._running   = False
        self._last_text = ""
        self._speaking  = False      # track when TTS is actively playing

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

    def is_speaking(self) -> bool:
        return self._speaking or not self._queue.empty()

    def speak(self, text: str, priority: bool = False):
        if not text:
            return
        if priority:
            self._last_text = ""
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
            import sys
            import os
            is_win = sys.platform == "win32"
            is_linux = sys.platform.startswith("linux")
            
            if is_win:
                import pythoncom
                import win32com.client
                try:
                    pythoncom.CoInitialize()
                except Exception:
                    pass
                speaker = win32com.client.Dispatch("SAPI.SpVoice")
                # SAPI rate is between -10 and 10
                speaker.Rate = 1 
                speaker.Volume = int(self._volume * 100)
                logger.info("SAPI5 engine initialised via win32com.")
            elif is_linux:
                speaker = None
                logger.info("Linux espeak enabled.")
            else:
                import pyttsx3
                speaker = pyttsx3.init()
                speaker.setProperty("rate", self._rate)
                speaker.setProperty("volume", self._volume)
                voices = speaker.getProperty("voices")
                if voices:
                    eng = [v for v in voices if "english" in v.name.lower() or "en" in v.id.lower()]
                    if eng:
                        speaker.setProperty("voice", eng[0].id)
                logger.info("pyttsx3 engine initialised.")

            while self._running:
                try:
                    text = self._queue.get(timeout=0.5)
                    if text is None:
                        break
                    self._speaking = True
                    
                    if is_win:
                        speaker.Speak(text)
                    elif is_linux:
                        safe = text.replace("'", "")
                        os.system(f"espeak -s {self._rate} '{safe}' >/dev/null 2>&1")
                    else:
                        speaker.say(text)
                        speaker.runAndWait()
                        
                    self._speaking = False
                    time.sleep(1.5)   # pause after speaking so mic doesn't catch output
                except queue.Empty:
                    self._speaking = False
                    continue
                except Exception as e:
                    self._speaking = False
                    logger.warning(f"TTS error: {e}")
        except Exception as e:
            logger.error(f"TTS engine failed: {e}")


class WhisperSTT:

    SAMPLE_RATE    = 16000
    RECORD_SECS    = 5
    SILENCE_THRESH = 0.003

    def __init__(self, model_size="base"):
        self.model_size = model_size
        self._model     = None
        self._loaded    = False

    def load(self):
        if self._loaded:
            return
        try:
            import whisper
            logger.info(f"Loading Whisper {self.model_size}...")
            self._model  = whisper.load_model(self.model_size)
            self._loaded = True
            logger.info("Whisper loaded.")
        except Exception as e:
            logger.error(f"Whisper load error: {e}")

    def record_and_transcribe(self) -> Optional[str]:
        if not self._loaded or self._model is None:
            return None
        try:
            import sounddevice as sd
            logger.info(f"Listening for {self.RECORD_SECS}s...")
            audio = sd.rec(
                int(self.SAMPLE_RATE * self.RECORD_SECS),
                samplerate=self.SAMPLE_RATE,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            rms = float(np.sqrt(np.mean(audio ** 2)))
            logger.debug(f"RMS={rms:.4f} thresh={self.SILENCE_THRESH}")
            if rms < self.SILENCE_THRESH:
                return None

            result = self._model.transcribe(audio, fp16=False, language="en")
            text   = result.get("text", "").strip()
            if text:
                logger.info(f"Transcribed: '{text}'")
            return text or None
        except Exception as e:
            logger.warning(f"STT error: {e}")
            return None


class VoiceController:

    def __init__(
        self,
        on_command:   Callable[[str], None],
        tts_rate:     int = 155,
        whisper_size: str = "base",
        tts_engine    = None,
    ):
        self.tts            = tts_engine if tts_engine is not None else TTSEngine(rate=tts_rate)
        self.stt            = WhisperSTT(model_size=whisper_size)
        self._on_command    = on_command
        self._listen_thread = None
        self._listening     = False

    def start(self):
        if not self.tts._running:
            self.tts.start()
        self.tts.speak("Hikari Shinro AI is ready. Please speak your navigation goal.", priority=True)
        threading.Thread(target=self._load_stt, daemon=True).start()

    def _load_stt(self):
        self.stt.load()
        self.tts.speak("Voice recognition ready. Speak your goal now.", priority=True)
        self._start_listen_loop()

    def _start_listen_loop(self):
        if self._listening:
            return
        self._listening     = True
        self._listen_thread = threading.Thread(target=self._loop, daemon=True)
        self._listen_thread.start()
        logger.info("Voice listen loop started.")

    def _loop(self):
        while self._listening:
            try:
                # Wait if TTS is currently speaking — prevents mic picking up speaker output
                if self.tts.is_speaking():
                    time.sleep(0.3)
                    continue

                text = self.stt.record_and_transcribe()

                if text:
                    # Filter out very short results and obvious noise
                    words = text.strip().split()
                    if len(words) < 2:
                        logger.debug(f"Too short, ignoring: '{text}'")
                        continue
                    # Filter repetitive TTS echo patterns
                    lower = text.lower()
                    if any(phrase in lower for phrase in [
                        "i am not going to",
                        "i'm not going to",
                        "same thing",
                        "hikari shinro",
                        "voice recognition ready",
                        "navigation goal",
                    ]):
                        logger.debug(f"Echo detected, ignoring: '{text}'")
                        continue

                    logger.info(f"Voice command: '{text}'")
                    self._on_command(text)

            except Exception as e:
                logger.warning(f"Listen error: {e}")
            time.sleep(0.05)

    def stop(self):
        self._listening = False