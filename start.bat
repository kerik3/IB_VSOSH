@echo off
echo ========================================
echo VVM Online School Platform
echo Starting Backend and Frontend...
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "backend\venv\" (
    echo Creating virtual environment...
    cd backend
    python -m venv venv
    cd ..
)

REM Start backend in new window
echo Starting Backend Server...
start "VVM Backend" cmd /k "cd backend && venv\Scripts\activate && python app.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak >nul

REM Start frontend in new window
echo Starting Frontend Server...
start "VVM Frontend" cmd /k "cd frontend && npm start"

echo.
echo ========================================
echo Both servers are starting!
echo Backend: http://localhost:5000
echo Frontend: http://localhost:3000
echo ========================================
echo.
echo Press any key to exit this window...
pause >nul
