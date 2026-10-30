@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "DICE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DICE_PYTHON%" set "DICE_PYTHON=python"
echo 本地标注台：http://127.0.0.1:8767
echo 原网页不受影响。再次双击会打开已有标注台。
"%DICE_PYTHON%" -m training.launch
pause
