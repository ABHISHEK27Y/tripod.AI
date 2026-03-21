@echo off
:: ============================================================
::  Hikari Shinro AI — Windows Auto-Setup Script
::  Double-click this file to set up the project
::  Requires: Python 3.10/3.11, C++ Build Tools, FFmpeg
:: ============================================================

title Hikari Shinro AI — Setup

echo.
echo  ================================================
echo   Hikari Shinro AI  -  Windows Setup
echo   CodeRonin  Ahouba 3.0  IIIT Manipur
echo  ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install from https://python.org
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)
echo [OK] Python found

:: Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found. Reinstall Python with pip enabled.
    pause
    exit /b 1
)
echo [OK] pip found

:: Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARN] FFmpeg not found in PATH.
    echo        Whisper STT may not work without FFmpeg.
    echo        Install from: https://www.gyan.dev/ffmpeg/builds/
    echo        Then add C:\ffmpeg\bin to your PATH.
    echo.
)

:: Create virtual environment
if not exist "venv" (
    echo [STEP 1/6] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv. Check Python installation.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

:: Activate venv
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated

:: Install PyTorch (CPU)
echo.
echo [STEP 2/6] Installing PyTorch (CPU version, ~200MB)...
echo            This may take several minutes...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo [ERROR] PyTorch install failed. Check internet connection.
    pause
    exit /b 1
)
echo [OK] PyTorch installed

:: Install PyAudio
echo.
echo [STEP 3/6] Installing PyAudio...
pip install pipwin --quiet
pipwin install pyaudio
if errorlevel 1 (
    echo [WARN] pipwin failed. Trying alternative...
    pip install pyaudio --quiet
    if errorlevel 1 (
        echo [WARN] PyAudio install failed.
        echo        Voice input (microphone) will not work.
        echo        Everything else (detection, depth, HUD) will still work.
        echo        See SETUP_WINDOWS.md for manual PyAudio install.
    )
) else (
    echo [OK] PyAudio installed
)

:: Install all other requirements
echo.
echo [STEP 4/6] Installing dependencies (~500MB, please wait)...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    echo        Check internet connection and try running this script again.
    pause
    exit /b 1
)
echo [OK] All dependencies installed

:: Copy .env if not exists
echo.
echo [STEP 5/6] Setting up environment file...
if not exist ".env" (
    copy .env.example .env >nul
    echo [OK] .env created from template
    echo.
    echo  !! IMPORTANT !!
    echo  You must add your GROQ_API_KEY to .env before running.
    echo  Get a FREE key at: https://console.groq.com
    echo  Then open .env in Notepad and replace the placeholder.
    echo.
    notepad .env
) else (
    echo [OK] .env already exists
)

:: Done
echo.
echo [STEP 6/6] Setup complete!
echo.
echo  ================================================
echo   To run Hikari Shinro AI:
echo.
echo   1. Open Command Prompt in this folder
echo   2. Run: venv\Scripts\activate
echo   3. Run: cd app
echo   4. Run: python app.py
echo   5. Open browser: http://localhost:5000
echo  ================================================
echo.
echo  If models are not downloaded yet, first run will
echo  download ~500MB (Whisper + YOLO-World + MiDaS).
echo.
pause
