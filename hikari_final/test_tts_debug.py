import sys
import os

print("--- Testing PowerShell TTS ---")
os.system(r'powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'Testing PowerShell Speech\')"')
print("PowerShell TTS command finished.")

try:
    if sys.platform == "win32":
        import pythoncom
        pythoncom.CoInitialize()

    import pyttsx3
    print("\n--- Testing pyttsx3 default voice ---")
    engine = pyttsx3.init()
    engine.say("Testing default py text to speech voice")
    engine.runAndWait()
    print("Default pyttsx3 test done.")
    
    print("\n--- Listing Voices ---")
    voices = engine.getProperty('voices')
    for idx, v in enumerate(voices):
        print(f"Voice {idx}: {v.name} (ID: {v.id})")
        if "zira" in str(v.name).lower() or "david" in str(v.name).lower():
            print(f"  -> Found common voice at idx {idx}, trying it out:")
            engine.setProperty('voice', v.id)
            engine.say("Testing voice number " + str(idx))
            engine.runAndWait()
            print(f"  -> Voice idx {idx} test done.")

except Exception as e:
    print("Error:", e)
