@echo off
cd /d "%~dp0"
python sweep_comparison_gui.py %*
if errorlevel 1 pause
