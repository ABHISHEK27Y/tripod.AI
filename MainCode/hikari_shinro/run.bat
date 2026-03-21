@echo off
title Hikari Shinro AI
echo.
echo  Starting Hikari Shinro AI...
echo  Open browser at: http://localhost:5000
echo.

:: Activate venv and run
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [WARN] venv not found. Run setup_windows.bat first.
)

cd app
python app.py
pause
