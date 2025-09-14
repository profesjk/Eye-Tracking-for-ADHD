# 🚀 Streamlit Cloud Deployment Guide

## ❌ Error Fix: "installer returned a non-zero exit code"

### The Problem
The error you encountered is caused by dependency conflicts during Streamlit Cloud deployment. Here's how to fix it:

### ✅ **Solution 1: Use Simplified Requirements (Recommended)**

Your `requirements_streamlit.txt` has been optimized to:
```
streamlit==1.35.0
opencv-python-headless==4.8.1.78
numpy==1.24.3
Pillow==10.0.0
```

### ✅ **Solution 2: Force Clean Deployment**

1. **Delete your current app** from Streamlit Cloud dashboard
2. **Clear browser cache** completely
3. **Push the optimized files** to GitHub:
   ```bash
   git add .
   git commit -m "Fix deployment dependencies"
   git push origin main
   ```
4. **Redeploy fresh** - don't reuse existing deployment

### ✅ **Solution 3: System Dependencies**

The following files ensure system packages are installed:

**packages.txt** (for Streamlit Cloud):
```
libgl1-mesa-glx
libglib2.0-0
```

**apt-packages.txt** (alternative naming):
```
libgl1-mesa-glx
libglib2.0-0
```

## 🔧 Step-by-Step Deployment

### 1. Prepare Repository
```bash
# Ensure all files are committed
git add requirements_streamlit.txt packages.txt apt-packages.txt
git commit -m "Optimize for Streamlit Cloud deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. **Create NEW app** (don't reuse existing)
3. Repository: `your-username/Eye-Tracking`
4. Branch: `main`
5. Main file: `streamlit_app.py`
6. **Advanced settings:**
   - Python version: `3.10`
   - Requirements file: `requirements_streamlit.txt`

### 3. Monitor Deployment
- Watch logs for any errors
- Deployment typically takes 3-5 minutes
- First run may take longer for package installation

## 🐞 Troubleshooting

### If Deployment Still Fails:

1. **Try Python 3.9 instead of 3.10**
2. **Use even simpler requirements:**
   ```
   streamlit
   opencv-python-headless
   numpy
   pillow
   ```

3. **Remove all version constraints:**
   ```
   streamlit
   opencv-python-headless
   numpy
   Pillow
   ```

### Common Issues:

❌ **"Rich dependency conflict"**
- Solution: Pin `rich<14` in requirements

❌ **"OpenCV import error"**
- Solution: Use `opencv-python-headless` not `opencv-python`

❌ **"System packages missing"**
- Solution: Ensure `packages.txt` is in repository root

## 🚀 Alternative Deployment Options

### Option 1: Railway
```bash
# Use our provided deploy script
chmod +x deploy.sh
./deploy.sh railway
```

### Option 2: Heroku
```bash
# Use our provided deploy script  
chmod +x deploy.sh
./deploy.sh heroku
```

### Option 3: Docker Local
```bash
docker build -t eye-tracking-app .
docker run -p 8501:8501 eye-tracking-app
```

## 📝 Files Created for Deployment

✅ `streamlit_app.py` - Optimized web application
✅ `requirements_streamlit.txt` - Minimal dependencies
✅ `packages.txt` - System packages
✅ `apt-packages.txt` - Alternative system packages
✅ `Dockerfile` - Container configuration
✅ `.streamlit/config.toml` - Streamlit settings
✅ `runtime.txt` - Python version specification

## 🎯 Success Indicators

When deployment works correctly, you'll see:
- ✅ "Installing requirements..."
- ✅ "Building application..."
- ✅ "Your app is live!"
- ✅ Camera permission request in browser

---

**Need Help?** The optimized files are now ready for deployment. Try the simplified requirements first!
