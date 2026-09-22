@echo off
cd /d "%~dp0"
python baseline_offline_gui.py %*
if errorlevel 1 pause
