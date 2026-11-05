@echo off
cd /d "%~dp0"
python -m synapse serve --port 8765
pause
