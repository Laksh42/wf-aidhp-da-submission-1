@echo off
setlocal EnableDelayedExpansion

REM Parse command line arguments
set USE_LOCAL_DB=false
:parse_args
if "%~1"=="" goto end_parse
if "%~1"=="--help" (
    echo Usage: start.bat [options]
    echo Options:
    echo   --help     Show this help message
    echo   --local    Use local MongoDB (default: false^)
    echo   --remote   Use remote MongoDB (default: true^)
    exit /b 0
)
if "%~1"=="--local" set USE_LOCAL_DB=true
if "%~1"=="--remote" set USE_LOCAL_DB=false
shift
goto parse_args
:end_parse

REM Update .env file with database toggle setting
powershell -Command "(Get-Content .env) -replace '^USE_LOCAL_DB=.*', 'USE_LOCAL_DB=%USE_LOCAL_DB%' | Set-Content .env"

REM Kill any existing processes on ports 8000 and 3000
echo Stopping any existing processes...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| find ":3000" ^| find "LISTENING"') do taskkill /f /pid %%a >nul 2>&1

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Check if MongoDB is running locally when using local connection
if "%USE_LOCAL_DB%"=="true" (
    echo Using local MongoDB connection
    REM Check if MongoDB service is running
    sc query MongoDB >nul 2>&1
    if errorlevel 1 (
        echo Warning: MongoDB service is not running. Please start MongoDB service if needed.
    )
) else (
    echo Using remote MongoDB connection to MongoDB Atlas
)

REM Start the backend server in a new window
echo Starting the backend server on port 8000...
start "Backend Server" cmd /c "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 0.0.0.0 --port 8000"

REM Wait for backend to start
echo Waiting for backend to start...
timeout /t 3 /nobreak >nul

REM Start the frontend
echo Starting the frontend on port 3000...
cd frontend
start "Frontend Server" cmd /c "npm start"

echo.
echo Application started!
echo Backend is running on http://localhost:8000
echo Frontend is running on http://localhost:3000
echo.
echo Press Ctrl+C in the respective windows to stop the servers
echo. 