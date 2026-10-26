@echo off
rem 启动平台：浏览器打开 http://127.0.0.1:8000
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo 平台地址: http://127.0.0.1:8000
echo 关闭此窗口即停止服务。
echo.
start "" http://127.0.0.1:8000
".venv\Scripts\python.exe" -m webapp.app
pause
