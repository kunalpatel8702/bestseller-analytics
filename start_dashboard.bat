@echo off
REM Quick Start Script for Streamlit Dashboard
REM This script launches the interactive web application

echo ========================================
echo  Amazon Bestselling Books Dashboard
echo ========================================
echo.
echo Starting Streamlit application...
echo.
echo The dashboard will open in your browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo ========================================
echo.

python -m streamlit run app.py
