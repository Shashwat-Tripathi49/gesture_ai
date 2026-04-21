@echo off
echo Setting up Gesture AI...
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Application...
python main.py

pause
