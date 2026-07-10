@echo off
cd /d "%~dp0"
echo Starting Khmer Video Translator...
echo Open http://127.0.0.1:5050 after the server starts.
python subtitle_pipeline.py
if errorlevel 1 (
  echo.
  echo The server could not start. Review the error above.
  pause
)
