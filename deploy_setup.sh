#!/bin/bash
# Quick Deployment Setup Script

echo "🚀 Setting up Eye Tracking App for Streamlit Cloud Deployment"
echo "============================================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "❌ Git repository not found. Please run this from your repository root."
    exit 1
fi

# Create optimized files for deployment
echo "📝 Creating deployment files..."

# Verify required files exist
required_files=("streamlit_app.py" "requirements_streamlit.txt" "packages.txt" ".streamlit/config.toml")
missing_files=()

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ Missing required files:"
    printf ' - %s\n' "${missing_files[@]}"
    echo ""
    echo "Please run the deployment optimization first."
    exit 1
fi

echo "✅ All deployment files present"

# Check git status
if [[ `git status --porcelain` ]]; then
    echo "📝 Staging changes for commit..."
    git add .
    git commit -m "Optimize for Streamlit Cloud deployment - fix dependency issues"
    echo "✅ Changes committed"
else
    echo "✅ No changes to commit"
fi

# Push to GitHub
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "🎉 Repository is ready for Streamlit Cloud deployment!"
echo ""
echo "Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Create NEW app (don't reuse existing)"
echo "3. Repository: $(git config --get remote.origin.url | sed 's/.*github.com[:/]\([^.]*\).*/\1/')"
echo "4. Branch: main"
echo "5. Main file: streamlit_app.py"
echo ""
echo "📋 Deployment Configuration:"
echo "   - Python version: 3.10"
echo "   - Requirements file: requirements_streamlit.txt"
echo "   - System packages: packages.txt"
echo ""
echo "🐞 If deployment fails, see DEPLOYMENT_FIX.md for troubleshooting"
