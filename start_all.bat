@echo off
title Air Link Imaging - Viewer + API
echo ============================================================
echo  Air Link Imaging - Starting API server and React viewer
echo ============================================================
echo.
echo  API server will run at http://localhost:8000
echo  React viewer will run at http://localhost:3000
echo.
echo  Open http://localhost:3000 in your browser.
echo ============================================================

cd /d "%~dp0"

:: Start API server in a new window
start "API Server" cmd /k "cd /d "%~dp0" && pip install fastapi uvicorn sse-starlette -q 2>nul & uvicorn api_server:app --host 0.0.0.0 --port 8000"

:: Wait a moment for API to bind
timeout /t 3 /nobreak >nul

:: Start React dev server in a new window
start "React Viewer" cmd /k "cd /d "%~dp0drone-viewer" && npm start"

echo.
echo Both servers are starting in separate windows.
echo Close this window when done (the other two will keep running).
pause
