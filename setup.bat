@echo off
echo ============================================
echo  Face Recognition - Environment Setup
echo ============================================

REM Create venv if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo  To activate the venv manually:
echo    call project\venv\Scripts\activate.bat
echo ============================================
