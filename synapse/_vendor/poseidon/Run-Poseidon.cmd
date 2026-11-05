@echo off
cd /d "%~dp0"
start "Poseidon browser" http://127.0.0.1:8787
python -m poseidon serve --port 8787
pause
