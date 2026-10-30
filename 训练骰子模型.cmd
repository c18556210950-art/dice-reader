@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "DICE_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%DICE_PYTHON%" set "DICE_PYTHON=python"
echo 下载公开数据，按拍摄批次导出，再用本机显卡训练两个模型。
echo 图片和模型保存在 .runtime\dice-training，原网页服务不会改变。
"%DICE_PYTHON%" -m training.workflow
pause
