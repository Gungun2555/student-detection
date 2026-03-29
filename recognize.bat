@echo off
call venv\Scripts\activate.bat
python recognize_photos.py %*
