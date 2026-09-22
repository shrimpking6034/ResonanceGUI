@echo off
cd /d "%~dp0"
python resonance_fpga_gui.py
if errorlevel 1 pause
