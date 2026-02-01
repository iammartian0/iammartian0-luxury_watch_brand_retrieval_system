@echo off

REM ========================================
REM   Simple Streamlit Launcher
REM ========================================

echo ========================================
echo   Watch Retrieval System - Streamlit Demo
echo ========================================
echo.

REM Check if streamlit exists in virtual environment
if not exist "venv_gpu\Scripts\streamlit.exe" (
    echo ERROR: Streamlit not found in virtual environment
    echo.
    echo Please install streamlit first:
    echo   pip install streamlit
    echo.
    pause
    exit /b 1
)

echo Streamlit found. Starting server...
echo.
echo Demo will open at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Change to project root directory (important for correct path resolution)
cd /d "%~dp0"

echo Working directory: %CD%
echo.

venv_gpu\Scripts\streamlit.exe run streamlit/app.py

echo.
echo Streamlit server stopped.
echo.
pause