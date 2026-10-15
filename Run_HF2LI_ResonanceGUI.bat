@echo off
cd /d "%~dp0"
python resonance_gui.py
if errorlevel 1 pause
