@echo off
setlocal
cd /d "%~dp0"

echo ===================================================
echo  Mini-projet cascade - TESTS
echo ===================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    pause
    exit /b 1
)

python -m pytest -q
if errorlevel 1 (
    echo.
    echo RESULT: TESTS FAILED. Read the red messages above.
) else (
    echo.
    echo RESULT: ALL TESTS PASSED.
)
pause
