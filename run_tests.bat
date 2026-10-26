@echo off
rem 双击运行用：跑全部测试（21 项）
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
".venv\Scripts\python.exe" -m pytest -q tests %*
echo.
pause
