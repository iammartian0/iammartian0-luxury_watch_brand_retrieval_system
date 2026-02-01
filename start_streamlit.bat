@echo off

REM ========================================
REM   Watch Retrieval System - Streamlit Demo Launcher
REM ========================================

echo ========================================
echo   Watch Retrieval System - Streamlit Demo
echo ========================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv_gpu\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Virtual environment activated successfully
echo.

REM Check if streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Installing Streamlit...
    pip install streamlit
    if errorlevel 1 (
        echo ERROR: Failed to install Streamlit
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo   Starting Streamlit Server
echo ========================================
echo.
echo Demo will open in your browser at:
echo   http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
echo ========================================
echo.

REM Run Streamlit from the virtual environment
venv_gpu\Scripts\streamlit.exe run streamlit/app.py

REM If user closes with Ctrl+C
echo.
echo Streamlit server stopped.
echo.
pause