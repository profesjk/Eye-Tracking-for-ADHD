# 👁️ Eye Tracking Application - Deployment Guide

A real-time eye tracking application with ADHD attention assistant, optimized for Streamlit deployment.

## 🚀 Quick Start

### Option 1: Local Development (Recommended for testing)

```bash
# Windows
deploy.bat local

# Linux/Mac
chmod +x deploy.sh
./deploy.sh local
```

### Option 2: Docker Deployment

```bash
# Windows
deploy.bat docker

# Linux/Mac
./deploy.sh docker
```

### Option 3: Cloud Deployment

See the [Cloud Deployment](#cloud-deployment) section below.

## 📋 Features

- ✅ **Real-time face and eye detection**
- ✅ **Pupil tracking with optimized algorithms**
- ✅ **ADHD attention assistant with focus monitoring**
- ✅ **Interactive Streamlit interface**
- ✅ **Session statistics and analytics**
- ✅ **Attention alerts and focus rewards**
- ✅ **Optimized for deployment on multiple platforms**

## 🛠️ Requirements

### System Requirements
- Python 3.8 or higher
- Webcam access
- Good lighting conditions
- Modern web browser

### Dependencies
All dependencies are listed in `requirements_streamlit.txt`:
- Streamlit (web interface)
- OpenCV (computer vision)
- NumPy (numerical computations)
- Pillow (image processing)

## 📦 Installation Methods

### 1. Manual Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Eye-Tracking

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements_streamlit.txt

# Run the application
streamlit run streamlit_app.py
```

### 2. Docker Installation

```bash
# Build the image
docker build -t eye-tracking-app .

# Run the container
docker run -p 8501:8501 eye-tracking-app
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Recommended)

1. **Push to GitHub**: Upload your code to a GitHub repository
2. **Visit Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io)
3. **Connect Repository**: Link your GitHub account and select the repository
4. **Configure Deployment**:
   - Main file: `streamlit_app.py`
   - Requirements file: `requirements_streamlit.txt`
5. **Deploy**: Click deploy and wait for the build to complete

### Other Cloud Platforms

#### Heroku
```bash
# Install Heroku CLI
heroku create your-app-name
heroku container:push web
heroku container:release web
```

#### Railway
1. Connect your GitHub repository to Railway
2. Select the repository
3. Railway will auto-detect the Dockerfile and deploy

#### Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/eye-tracking
gcloud run deploy --image gcr.io/PROJECT-ID/eye-tracking --platform managed
```

#### AWS App Runner
1. Create an App Runner service
2. Connect to your GitHub repository
3. Use the provided Dockerfile for deployment

## 🎮 Usage

### Starting the Application

1. **Select Mode**: Choose between Eye Detection, ADHD Assistant, or Calibration
2. **Start Camera**: Click "Start Camera" to begin video processing
3. **Monitor**: Watch the live feed and statistics in real-time
4. **Stop**: Click "Stop Camera" when finished

### Modes Explained

#### Eye Detection Mode
- Basic face and eye detection
- Pupil tracking visualization
- Real-time video processing stats

#### ADHD Assistant Mode
- Monitors attention and focus
- Provides alerts when attention wanes
- Tracks focus time and session statistics
- Helps maintain concentration during work/study

#### Calibration Mode
- Calibrates eye tracking for cursor control
- Maps eye movements to screen coordinates
- Improves tracking accuracy

### Controls

- **Start Camera**: Begin video processing
- **Stop Camera**: End video processing
- **Mode Selection**: Switch between different tracking modes
- **Statistics Panel**: View real-time metrics

## 🔧 Configuration

### Camera Settings
The application automatically configures optimal camera settings:
- Resolution: 640x480 (for stability)
- FPS: 15 (optimized for deployment)
- Auto-focus and exposure adjustment

### Performance Optimization
- Reduced frame processing for cloud deployment
- Optimized OpenCV operations
- Efficient memory management
- Streamlined detection algorithms

## 🐛 Troubleshooting

### Common Issues

#### Camera Access Denied
- **Solution**: Grant camera permissions in your browser
- **Chrome**: Click the camera icon in the address bar
- **Firefox**: Click the shield icon and allow camera access

#### Poor Detection Performance
- **Lighting**: Ensure good, even lighting on your face
- **Position**: Keep your face centered and at appropriate distance
- **Background**: Use a plain background for better detection

#### High CPU Usage
- **Solution**: The application is optimized for cloud deployment
- Close other applications to free up resources
- Use Docker deployment for better resource isolation

#### Application Won't Start
1. Check Python version: `python --version` (should be 3.8+)
2. Verify dependencies: `pip list`
3. Try reinstalling: `pip install -r requirements_streamlit.txt --force-reinstall`

### Performance Tips

1. **Good Lighting**: Ensure your face is well-lit
2. **Stable Position**: Maintain consistent distance from camera
3. **Plain Background**: Use a simple background for better detection
4. **Browser Performance**: Use Chrome or Firefox for best results
5. **Network**: Ensure stable internet for cloud deployments

## 📊 Technical Details

### Architecture
```
User Interface (Streamlit)
    ↓
Camera Input (OpenCV)
    ↓
Face Detection (Haar Cascades)
    ↓
Eye Detection & Pupil Tracking
    ↓
ADHD Assistant Logic
    ↓
Real-time Display & Statistics
```

### File Structure
```
Eye-Tracking/
├── streamlit_app.py           # Main Streamlit application
├── requirements_streamlit.txt # Optimized dependencies
├── Dockerfile                 # Container configuration
├── deploy.sh                 # Linux/Mac deployment script
├── deploy.bat                # Windows deployment script
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README_DEPLOYMENT.md      # This file
```

### Security Considerations
- Camera access is only used locally in the browser
- No video data is stored or transmitted
- All processing happens in real-time
- Privacy-focused design

## 🤝 Support

### Getting Help
1. Check the troubleshooting section above
2. Review the application logs in Streamlit
3. Ensure your system meets the requirements
4. Test camera access in other applications

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Browser type and version
- Error messages or logs
- Steps to reproduce the issue

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎯 Roadmap

### Upcoming Features
- [ ] Multi-face tracking
- [ ] Advanced calibration options
- [ ] Export functionality for session data
- [ ] Mobile device support
- [ ] Additional attention metrics

### Performance Improvements
- [ ] WebGL acceleration
- [ ] Optimized models for better detection
- [ ] Real-time performance monitoring
- [ ] Adaptive quality settings

---

**🎉 Ready to deploy? Choose your preferred method above and get started!**
