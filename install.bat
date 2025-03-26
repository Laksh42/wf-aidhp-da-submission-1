@echo off
echo Starting installation process...

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.9 or higher.
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if errorlevel 1 (
    echo Node.js is not installed! Please install Node.js 14 or higher.
    pause
    exit /b 1
)

REM Create and activate virtual environment
echo Creating Python virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

REM Install Python dependencies
echo Installing Python requirements...
python -m pip install --upgrade pip
pip install -r requirements.txt

REM Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
call npm install
cd ..

REM Create .env file if it doesn't exist
echo Checking .env file...
if not exist .env (
    echo Creating .env file...
    (
        echo # MongoDB Configuration
        echo MONGODB_URL=mongodb+srv://wf-hack:eAzy%%40123@dataset-db.ky0bo.mongodb.net/dataset?retryWrites=true^&w=majority
        echo MONGODB_DB=dataset
        echo MONGODB_USER=wf-hack
        echo MONGODB_PASSWORD=eAzy@123
        echo USE_LOCAL_DB=false
        echo.
        echo # API Keys
        echo OPENAI_API_KEY=
        echo HUGGINGFACE_TOKEN=
        echo MISTRAL_API_KEY=Uf6oAM3GC8D18fi3Zn6lVNUNxp92Z592
    ) > .env
)

echo.
echo Installation completed successfully!
echo You can now run start.bat to start the application.
echo.
pause 