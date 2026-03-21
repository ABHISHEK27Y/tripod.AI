import pyttsx3
import queue
import threading

speech_queue = queue.Queue()

engine = pyttsx3.init()
engine.setProperty("rate", 170)

def speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text: str):
    speech_queue.put(text)