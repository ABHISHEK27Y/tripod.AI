import speech_recognition as sr
import threading
import queue
import whisper
import os

class SpeechToText:
    def __init__(self, energy_threshold=4000, model_type="small"):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.mic = sr.Microphone()
        
        print(f"[STT] Loading local Whisper {model_type} model...")
        self.model = whisper.load_model(model_type)
        print("[STT] Local Whisper loaded.")
        
        self.text_queue = queue.Queue()
        self._is_listening = False
        self._thread = None

    def start_listening(self):
        """Starts a background thread to continuously listen to the microphone."""
        if self._is_listening:
            return
            
        self._is_listening = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        print("[STT] Started listening (Whisper)...")

    def stop_listening(self):
        self._is_listening = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def _listen_loop(self):
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
        temp_wav = "/tmp/whisper_buffer.wav"
        
        while self._is_listening:
            try:
                with self.mic as source:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                    
                # Save chunk to disk for Whisper inference
                with open(temp_wav, "wb") as f:
                    f.write(audio.get_wav_data())
                    
                result = self.model.transcribe(temp_wav, fp16=False)
                text = result["text"].strip()
                
                if text:
                    print(f"[STT Local Whisper] Heard: '{text}'")
                    self.text_queue.put(text)
                    
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                pass

    def get_latest_goal(self):
        """Returns the most recent spoken goal, or None if empty."""
        goals = []
        while not self.text_queue.empty():
            goals.append(self.text_queue.get())
            
        if goals:
            return " ".join(goals)
        return None

