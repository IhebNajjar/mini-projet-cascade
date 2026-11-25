@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo  Mini-projet cascade - START
echo ===================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Install Python 3.10 or newer, then run this file again.
    pause
    exit /b 1
)

if not exist "models\model.pkl" (
    echo Models not found. Training them now, please wait...
    python train_model.py
    if errorlevel 1 (
        echo ERROR: training failed.
        pause
        exit /b 1
    )
)

rem ---- API on port 8000 ----
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo API already running on port 8000.
    goto check_web
)

echo Starting API on port 8000...
start "CASCADE API" /D "%~dp0" cmd /k python -m uvicorn main:app --host 127.0.0.1 --port 8000

set TRIES=0
:wait_api
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 goto api_ready
set /a TRIES+=1
if %TRIES% GEQ 25 (
    echo ERROR: API did not start. Look at the black "CASCADE API" window.
    pause
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait_api
:api_ready
echo API is ready.

:check_web
rem ---- Web page on port 8001 ----
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo Web page already running on port 8001.
    goto open_browsers
)

echo Starting web page on port 8001...
start "CASCADE WEB" /D "%~dp0" cmd /k python -m http.server 8001 --bind 127.0.0.1

set TRIES=0
:wait_web
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if not errorlevel 1 goto web_ready
set /a TRIES+=1
if %TRIES% GEQ 25 (
    echo ERROR: web page did not start. Look at the "CASCADE WEB" window.
    pause
    exit /b 1
)
ping -n 2 127.0.0.1 >nul
goto wait_web
:web_ready
echo Web page is ready.

:open_browsers
echo.
echo Opening:
echo   API documentation : http://127.0.0.1:8000/docs
echo   Test web page     : http://127.0.0.1:8001/web_app.html
start "" http://127.0.0.1:8000/docs
start "" http://127.0.0.1:8001/web_app.html
echo.
echo Keep the two black windows open while you use the app.
echo Close those windows to stop the app.
pause
