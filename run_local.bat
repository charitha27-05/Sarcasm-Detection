@echo off
echo ========================================================
echo   Starting Sarcasm Detection AI Server locally
echo ========================================================
echo.
echo Installing dependencies (if needed)...
python -m pip install -r requirements.txt
echo.
echo Starting Flask Application on http://localhost:5050 ...
python app.py
pause
