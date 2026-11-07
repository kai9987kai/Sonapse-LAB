@echo off
cd /d "%~dp0"
python -m synapse --total-neurons 4000 serve --port 8765
pause
