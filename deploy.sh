#!/bin/bash

# Eye Tracking App Deployment Script
# Supports local, Docker, and cloud deployment

set -e

echo "🚀 Eye Tracking App Deployment Script"
echo "======================================"

# Function to check dependencies
check_dependencies() {
    echo "📋 Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python 3 is required but not installed."
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        echo "❌ pip is required but not installed."
        exit 1
    fi
    
    echo "✅ Dependencies check passed"
}

# Function for local deployment
deploy_local() {
    echo "🏠 Starting local deployment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate || source venv/Scripts/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    echo "📦 Installing requirements..."
    pip install -r requirements_streamlit.txt
    
    # Start the application
    echo "🎬 Starting Streamlit application..."
    echo "💡 The app will be available at: http://localhost:8501"
    echo "🛑 Press Ctrl+C to stop the application"
    
    streamlit run streamlit_app.py
}

# Function for Docker deployment
deploy_docker() {
    echo "🐳 Starting Docker deployment..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is required but not installed."
        echo "Please install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Build Docker image
    echo "🔨 Building Docker image..."
    docker build -t eye-tracking-app .
    
    # Run Docker container
    echo "🚀 Starting Docker container..."
    echo "💡 The app will be available at: http://localhost:8501"
    echo "🛑 Press Ctrl+C to stop the container"
    
    docker run -p 8501:8501 \
        --name eye-tracking-container \
        --rm \
        eye-tracking-app
}

# Function for cloud deployment (Streamlit Cloud)
deploy_cloud() {
    echo "☁️ Preparing for cloud deployment..."
    
    echo "📝 Cloud deployment checklist:"
    echo "1. Push your code to GitHub"
    echo "2. Go to https://share.streamlit.io"
    echo "3. Connect your GitHub repository"
    echo "4. Set the main file to: streamlit_app.py"
    echo "5. Set requirements file to: requirements_streamlit.txt"
    echo ""
    echo "📋 Repository structure should be:"
    echo "  - streamlit_app.py (main application)"
    echo "  - requirements_streamlit.txt (dependencies)"
    echo "  - .streamlit/config.toml (configuration)"
    echo ""
    echo "🔗 Alternative cloud platforms:"
    echo "  - Heroku: Use the provided Dockerfile"
    echo "  - Railway: Deploy directly from GitHub"
    echo "  - Google Cloud Run: Use Docker deployment"
    echo "  - AWS App Runner: Use Docker deployment"
}

# Function to show help
show_help() {
    echo "Usage: ./deploy.sh [option]"
    echo ""
    echo "Options:"
    echo "  local    - Deploy locally with virtual environment"
    echo "  docker   - Deploy using Docker container"
    echo "  cloud    - Show cloud deployment instructions"
    echo "  help     - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./deploy.sh local    # Start local development server"
    echo "  ./deploy.sh docker   # Build and run Docker container"
    echo "  ./deploy.sh cloud    # Show cloud deployment guide"
}

# Main script logic
case "${1:-local}" in
    "local")
        check_dependencies
        deploy_local
        ;;
    "docker")
        deploy_docker
        ;;
    "cloud")
        deploy_cloud
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    *)
        echo "❌ Unknown option: $1"
        show_help
        exit 1
        ;;
esac
