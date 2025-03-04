@echo off
echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies with additional options...
pip install --no-cache-dir -r requirements.txt

if %errorlevel% neq 0 (
    echo Failed to install dependencies. Trying alternative installation method...
    pip install --upgrade --force-reinstall -r requirements.txt
)

if %errorlevel% neq 0 (
    echo Critical error installing dependencies
    pause
    exit /b %errorlevel%
)

echo Checking installed dependencies...
pip list

echo Starting FastAPI Server...
python app.py
pause