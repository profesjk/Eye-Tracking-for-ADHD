@echo off
echo 🚀 Setting up Eye Tracking App for Streamlit Cloud Deployment
echo =============================================================

REM Check if git is initialized
if not exist ".git" (
    echo ❌ Git repository not found. Please run this from your repository root.
    pause
    exit /b 1
)

echo 📝 Checking deployment files...

REM Check if required files exist
set "missing_files="
if not exist "streamlit_app.py" set "missing_files=%missing_files% streamlit_app.py"
if not exist "requirements_streamlit.txt" set "missing_files=%missing_files% requirements_streamlit.txt"
if not exist "packages.txt" set "missing_files=%missing_files% packages.txt"
if not exist ".streamlit\config.toml" set "missing_files=%missing_files% .streamlit\config.toml"

if not "%missing_files%"=="" (
    echo ❌ Missing required files:%missing_files%
    echo.
    echo Please run the deployment optimization first.
    pause
    exit /b 1
)

echo ✅ All deployment files present

REM Check git status and commit
echo 📝 Staging changes for commit...
git add .
git commit -m "Optimize for Streamlit Cloud deployment - fix dependency issues"
echo ✅ Changes committed

REM Push to GitHub
echo 🚀 Pushing to GitHub...
git push origin main

echo.
echo 🎉 Repository is ready for Streamlit Cloud deployment!
echo.
echo Next steps:
echo 1. Go to https://share.streamlit.io
echo 2. Create NEW app (don't reuse existing)
echo 3. Repository: [Your GitHub repo URL]
echo 4. Branch: main
echo 5. Main file: streamlit_app.py
echo.
echo 📋 Deployment Configuration:
echo    - Python version: 3.10
echo    - Requirements file: requirements_streamlit.txt
echo    - System packages: packages.txt
echo.
echo 🐞 If deployment fails, see DEPLOYMENT_FIX.md for troubleshooting
pause
