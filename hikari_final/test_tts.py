import threading
import pyttsx3
import time

def worker():
    try:
        from pyttsx3 import init
        engine = init()
        engine.say("Hello")
        engine.runAndWait()
        print("Done speaking")
    except Exception as e:
        print(f"Error: {e}")

t = threading.Thread(target=worker)
t.start()
t.join()
