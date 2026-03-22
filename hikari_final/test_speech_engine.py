import sys
import time

try:
    from app.speech import TTSEngine
    print("Testing TTSEngine...")
    tts = TTSEngine(rate=150)
    tts.start()
    tts.speak("This is a test. Did you hear this? One, two, three.", priority=True)
    time.sleep(5)
    print("Test finished.")
except Exception as e:
    print("Error:", e)
