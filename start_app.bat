@echo off
cd /d "%~dp0"
echo Starting Auto Captions Studio at http://127.0.0.1:5050 ...
start "" http://127.0.0.1:5050
python app.py
pause
