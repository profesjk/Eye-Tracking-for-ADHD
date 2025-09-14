@echo off
REM Eye Tracking App Deployment Script for Windows
REM Supports local and Docker deployment

echo 🚀 Eye Tracking App Deployment Script
echo ======================================

if "%1"=="local" goto deploy_local
if "%1"=="docker" goto deploy_docker
if "%1"=="cloud" goto deploy_cloud
if "%1"=="help" goto show_help
if "%1"=="-h" goto show_help
if "%1"=="--help" goto show_help
if "%1"=="" goto deploy_local

echo ❌ Unknown option: %1
goto show_help

:deploy_local
echo 🏠 Starting local deployment...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
pip install --upgrade pip

REM Install requirements
echo 📦 Installing requirements...
pip install -r requirements_streamlit.txt

REM Start the application
echo 🎬 Starting Streamlit application...
echo 💡 The app will be available at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the application

streamlit run streamlit_app.py
goto end

:deploy_docker
echo 🐳 Starting Docker deployment...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is required but not installed.
    echo Please install Docker from: https://docs.docker.com/get-docker/
    exit /b 1
)

REM Build Docker image
echo 🔨 Building Docker image...
docker build -t eye-tracking-app .

REM Run Docker container
echo 🚀 Starting Docker container...
echo 💡 The app will be available at: http://localhost:8501
echo 🛑 Press Ctrl+C to stop the container

docker run -p 8501:8501 --name eye-tracking-container --rm eye-tracking-app
goto end

:deploy_cloud
echo ☁️ Preparing for cloud deployment...
echo.
echo 📝 Cloud deployment checklist:
echo 1. Push your code to GitHub
echo 2. Go to https://share.streamlit.io
echo 3. Connect your GitHub repository
echo 4. Set the main file to: streamlit_app.py
echo 5. Set requirements file to: requirements_streamlit.txt
echo.
echo 📋 Repository structure should be:
echo   - streamlit_app.py (main application)
echo   - requirements_streamlit.txt (dependencies)
echo   - .streamlit/config.toml (configuration)
echo.
echo 🔗 Alternative cloud platforms:
echo   - Heroku: Use the provided Dockerfile
echo   - Railway: Deploy directly from GitHub
echo   - Google Cloud Run: Use Docker deployment
echo   - AWS App Runner: Use Docker deployment
goto end

:show_help
echo Usage: deploy.bat [option]
echo.
echo Options:
echo   local    - Deploy locally with virtual environment
echo   docker   - Deploy using Docker container
echo   cloud    - Show cloud deployment instructions
echo   help     - Show this help message
echo.
echo Examples:
echo   deploy.bat local    # Start local development server
echo   deploy.bat docker   # Build and run Docker container
echo   deploy.bat cloud    # Show cloud deployment guide

:end
pause
