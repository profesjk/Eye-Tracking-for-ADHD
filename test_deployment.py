#!/usr/bin/env python3
"""
Test script for Eye Tracking Application deployment
Verifies all dependencies and basic functionality
"""

import sys
import importlib
import subprocess
import platform

def test_python_version():
    """Test if Python version is compatible."""
    print("🐍 Testing Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def test_dependencies():
    """Test if all required dependencies are installed."""
    print("\n📦 Testing dependencies...")
    
    required_packages = [
        'streamlit',
        'cv2',
        'numpy',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                importlib.import_module('cv2')
                print(f"✅ OpenCV - Available")
            elif package == 'PIL':
                importlib.import_module('PIL')
                print(f"✅ Pillow - Available")
            else:
                importlib.import_module(package)
                print(f"✅ {package} - Available")
        except ImportError:
            print(f"❌ {package} - Not found")
            failed_imports.append(package)
    
    return len(failed_imports) == 0

def test_opencv_functionality():
    """Test basic OpenCV functionality."""
    print("\n📹 Testing OpenCV functionality...")
    
    try:
        import cv2
        
        # Test Haar cascades
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        
        if face_cascade.empty():
            print("❌ Face cascade failed to load")
            return False
        else:
            print("✅ Face cascade loaded successfully")
        
        if eye_cascade.empty():
            print("❌ Eye cascade failed to load")
            return False
        else:
            print("✅ Eye cascade loaded successfully")
        
        # Test camera access (without actually opening)
        print("✅ OpenCV functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ OpenCV test failed: {e}")
        return False

def test_streamlit():
    """Test Streamlit availability and basic functionality."""
    print("\n🌊 Testing Streamlit...")
    
    try:
        import streamlit as st
        version = st.__version__
        print(f"✅ Streamlit {version} - Available")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def test_camera_access():
    """Test basic camera access (optional)."""
    print("\n📷 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera accessible")
            cap.release()
            return True
        else:
            print("⚠️ Camera not accessible (this is optional for deployment)")
            return True  # Not critical for deployment
            
    except Exception as e:
        print(f"⚠️ Camera test failed: {e} (this is optional for deployment)")
        return True  # Not critical for deployment

def test_file_structure():
    """Test if required files are present."""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'streamlit_app.py',
        'requirements_streamlit.txt',
        '.streamlit/config.toml'
    ]
    
    missing_files = []
    
    import os
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - Found")
        else:
            print(f"❌ {file} - Missing")
            missing_files.append(file)
    
    return len(missing_files) == 0

def run_deployment_test():
    """Run a quick Streamlit test."""
    print("\n🚀 Testing Streamlit deployment...")
    
    try:
        # Test if streamlit command is available
        result = subprocess.run(['streamlit', '--version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print("✅ Streamlit command available")
            print("💡 You can now run: streamlit run streamlit_app.py")
            return True
        else:
            print("❌ Streamlit command failed")
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️ Streamlit command timeout")
        return False
    except FileNotFoundError:
        print("❌ Streamlit command not found")
        return False
    except Exception as e:
        print(f"❌ Streamlit test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Eye Tracking Application - Deployment Test")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Dependencies", test_dependencies),
        ("OpenCV Functionality", test_opencv_functionality),
        ("Streamlit", test_streamlit),
        ("File Structure", test_file_structure),
        ("Camera Access", test_camera_access),
        ("Deployment Readiness", run_deployment_test)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your application is ready for deployment.")
        print("\n🚀 Next steps:")
        print("1. Run locally: streamlit run streamlit_app.py")
        print("2. Deploy to cloud: Follow the README_DEPLOYMENT.md guide")
        print("3. Use Docker: docker build -t eye-tracking-app .")
    elif passed_tests >= total_tests - 1:
        print("⚠️ Most tests passed. Check the failed test above.")
        print("The application should still work with minor issues.")
    else:
        print("❌ Several tests failed. Please fix the issues before deploying.")
        print("Check the error messages above and install missing dependencies.")
    
    print(f"\n💻 System Info:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Python: {sys.version}")
    print(f"   Architecture: {platform.machine()}")

if __name__ == "__main__":
    main()
