import sys

def test_sapi():
    try:
        if sys.platform == "win32":
            import pythoncom
            pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak("Testing SAPI directly. This bypasses pyttsx3.")
            print("SAPI Test completed successfully.")
    except Exception as e:
        print("SAPI test error:", e)

test_sapi()
