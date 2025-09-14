import streamlit as st
import sys
import os

try:
    import cv2
    import numpy as np
    import time
    from collections import deque
    import threading
    import queue
    import tempfile
    from pathlib import Path
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

# Check if OpenCV is properly configured
try:
    cv2.CascadeClassifier()
except Exception as e:
    st.error(f"OpenCV configuration error: {e}")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="Eye Tracking Application",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
MODE_DETECTING = 0
MODE_CALIBRATING = 1
MODE_CALIBRATING_RECORDING = 2
MODE_CONTROLLING = 3
MODE_ADHD_ASSISTANT = 4

CALIBRATION_POINTS = [
    (0.1, 0.1), (0.9, 0.1),  # Top-left, Top-right
    (0.1, 0.9), (0.9, 0.9),  # Bottom-left, Bottom-right
    (0.5, 0.5)              # Center
]

CALIBRATION_SAMPLES = 10
CALIBRATION_RECORD_TIME = 3
GAZE_SMOOTHING_BUFFER_SIZE = 10
CURSOR_SMOOTHING_FACTOR = 0.3
GAZE_DEADZONE = 1.0
CURSOR_RADIUS = 8
CURSOR_COLOR = (255, 0, 255)

# ADHD Assistant Constants
ATTENTION_COOLDOWN_SEC = 3.0
BORDER_THRESHOLD_PERCENT = 0.15
ATTENTION_LOST_FRAMES = 15
FOCUS_REWARD_TIME_SEC = 8.0
BRIEF_DISTRACTION_THRESHOLD_SEC = 1.5

class EyeTracker:
    def __init__(self):
        """Initialize the eye tracker with optimized settings for deployment."""
        self.face_cascade = None
        self.eye_cascade = None
        self.current_mode = MODE_DETECTING
        self.calibration_step = 0
        self.calibration_data = {}
        self.gaze_map_params = {}
        self.gaze_offset_buffer = deque(maxlen=GAZE_SMOOTHING_BUFFER_SIZE)
        
        # ADHD Assistant State
        self.attention_lost_counter = 0
        self.last_attention_alert_time = 0
        self.continuous_focus_time = 0.0
        self.session_start_time = time.time()
        self.total_focus_time = 0.0
        
        # Initialize cascades
        self._initialize_cascades()
    
    def _initialize_cascades(self):
        """Initialize Haar cascades with error handling."""
        try:
            # Try to load cascades from OpenCV
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            if self.face_cascade.empty() or self.eye_cascade.empty():
                st.error("Failed to load Haar cascades. Please check OpenCV installation.")
                return False
            return True
        except Exception as e:
            st.error(f"Error initializing cascades: {e}")
            return False
    
    def find_pupil(self, eye_roi_gray):
        """Find pupil center in eye ROI with optimized parameters."""
        if eye_roi_gray is None or eye_roi_gray.size == 0:
            return None, None
        
        try:
            # Enhanced preprocessing for better pupil detection
            eye_roi_gray = cv2.equalizeHist(eye_roi_gray)
            eye_roi_blurred = cv2.GaussianBlur(eye_roi_gray, (5, 5), 0)
            
            # Adaptive thresholding
            _, threshold = cv2.threshold(eye_roi_blurred, 30, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                h, w = eye_roi_gray.shape
                min_area = (w * h) * 0.02
                max_area = (w * h) * 0.4
                
                valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
                
                if valid_contours:
                    pupil_contour = max(valid_contours, key=cv2.contourArea)
                    M = cv2.moments(pupil_contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        return (cX, cY), pupil_contour
        except Exception:
            pass
        
        return None, None
    
    def process_frame(self, frame):
        """Process a single frame for eye tracking."""
        if frame is None:
            return None
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        detected_pupils = []
        left_eye_detected = False
        right_eye_detected = False
        face_center = None
        
        if len(faces) > 0:
            # Use the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face_center = (x + w // 2, y + h // 2)
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face ROI
            roi_gray = gray_frame[y:y + h, x:x + w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.05, minNeighbors=6,
                minSize=(w // 12, h // 12), maxSize=(w // 3, h // 3)
            )
            
            eyes_sorted = sorted(eyes, key=lambda e: e[0])
            processed_eyes = 0
            
            for i, (ex, ey, ew, eh) in enumerate(eyes_sorted):
                if ey + eh // 2 > h * 0.6 or processed_eyes >= 2:
                    continue
                
                is_left_eye = (ex + ew // 2) < w // 2
                
                # Calculate absolute coordinates
                eye_x_abs, eye_y_abs = x + ex, y + ey
                
                # Draw eye rectangle
                cv2.rectangle(frame, (eye_x_abs, eye_y_abs), 
                             (eye_x_abs + ew, eye_y_abs + eh), (0, 255, 0), 1)
                
                # Label eye
                label = "LEFT" if is_left_eye else "RIGHT"
                cv2.putText(frame, label, (eye_x_abs, eye_y_abs - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # Extract eye ROI
                eye_roi_gray = roi_gray[ey:ey + eh, ex:ex + ew]
                
                # Track detection
                if is_left_eye:
                    left_eye_detected = True
                else:
                    right_eye_detected = True
                
                # Detect pupil
                pupil_center_rel, _ = self.find_pupil(eye_roi_gray)
                
                if pupil_center_rel:
                    pupil_x_abs = eye_x_abs + pupil_center_rel[0]
                    pupil_y_abs = eye_y_abs + pupil_center_rel[1]
                    
                    # Draw pupil
                    cv2.circle(frame, (pupil_x_abs, pupil_y_abs), 3, (0, 0, 255), -1)
                    
                    # Calculate eye center
                    eye_center_x = eye_x_abs + ew // 2
                    eye_center_y = eye_y_abs + eh // 2
                    
                    # Store pupil data
                    detected_pupils.append({
                        'offset': (pupil_x_abs - eye_center_x, pupil_y_abs - eye_center_y),
                        'is_left': is_left_eye
                    })
                
                processed_eyes += 1
        
        return {
            'frame': frame,
            'detected_pupils': detected_pupils,
            'left_eye_detected': left_eye_detected,
            'right_eye_detected': right_eye_detected,
            'face_center': face_center,
            'faces_count': len(faces)
        }
    
    def update_adhd_assistant(self, detection_result):
        """Update ADHD assistant logic."""
        now = time.time()
        frame_height, frame_width = 480, 640  # Default frame size
        
        # Check attention conditions
        attention_issue = False
        attention_message = ""
        
        if detection_result['faces_count'] == 0:
            attention_issue = True
            attention_message = "No face detected"
        elif not (detection_result['left_eye_detected'] or detection_result['right_eye_detected']):
            attention_issue = True
            attention_message = "Eyes not detected"
        elif detection_result['face_center']:
            face_x, face_y = detection_result['face_center']
            border_x = int(frame_width * BORDER_THRESHOLD_PERCENT)
            border_y = int(frame_height * BORDER_THRESHOLD_PERCENT)
            
            if (face_x < border_x or face_x > frame_width - border_x or 
                face_y < border_y or face_y > frame_height - border_y):
                attention_issue = True
                attention_message = "Face near edge"
        
        if attention_issue:
            self.attention_lost_counter += 1
            if self.attention_lost_counter >= ATTENTION_LOST_FRAMES:
                if now - self.last_attention_alert_time > ATTENTION_COOLDOWN_SEC:
                    self.last_attention_alert_time = now
                    return True, attention_message  # Trigger alert
        else:
            self.attention_lost_counter = 0
            self.continuous_focus_time += 0.1  # Approximate frame time
            self.total_focus_time += 0.1
        
        return False, attention_message

def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("👁️ Eye Tracking Application")
    st.markdown("### Real-time eye tracking with ADHD attention assistant")
    
    # Sidebar controls
    st.sidebar.title("🎮 Controls")
    
    # Initialize session state
    if 'tracker' not in st.session_state:
        st.session_state.tracker = EyeTracker()
        st.session_state.camera_started = False
        st.session_state.current_mode = MODE_DETECTING
    
    # Mode selection
    mode_options = {
        "Eye Detection": MODE_DETECTING,
        "ADHD Assistant": MODE_ADHD_ASSISTANT,
        "Calibration": MODE_CALIBRATING
    }
    
    selected_mode = st.sidebar.selectbox(
        "Select Mode",
        list(mode_options.keys()),
        index=0
    )
    
    st.session_state.current_mode = mode_options[selected_mode]
    st.session_state.tracker.current_mode = st.session_state.current_mode
    
    # Camera controls
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        start_camera = st.button("📹 Start Camera", use_container_width=True)
    
    with col2:
        stop_camera = st.button("⏹️ Stop Camera", use_container_width=True)
    
    if start_camera:
        st.session_state.camera_started = True
    
    if stop_camera:
        st.session_state.camera_started = False
    
    # Information panel
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Information")
    
    if st.session_state.current_mode == MODE_ADHD_ASSISTANT:
        st.sidebar.markdown("""
        **ADHD Assistant Mode:**
        - Monitors your attention in real-time
        - Alerts when you look away
        - Tracks focus time and statistics
        - Helps maintain concentration
        """)
    elif st.session_state.current_mode == MODE_DETECTING:
        st.sidebar.markdown("""
        **Detection Mode:**
        - Basic eye and face detection
        - Shows pupil tracking
        - Real-time video processing
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📹 Live Video Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("#### 📈 Statistics")
        stats_placeholder = st.empty()
        
        st.markdown("#### 🚨 Alerts")
        alert_placeholder = st.empty()
    
    # Camera processing
    if st.session_state.camera_started:
        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Could not access camera. Please check permissions.")
                st.session_state.camera_started = False
            else:
                st.success("✅ Camera connected successfully!")
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for stability
                
                # Main processing loop
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.camera_started:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("❌ Failed to read from camera")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    result = st.session_state.tracker.process_frame(frame)
                    
                    if result:
                        processed_frame = result['frame']
                        
                        # ADHD Assistant logic
                        if st.session_state.current_mode == MODE_ADHD_ASSISTANT:
                            alert_triggered, alert_msg = st.session_state.tracker.update_adhd_assistant(result)
                            
                            # Add mode indicator
                            cv2.putText(processed_frame, "ADHD Assistant Mode", 
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Add focus timer
                            focus_time = st.session_state.tracker.continuous_focus_time
                            cv2.putText(processed_frame, f"Focus: {focus_time:.1f}s", 
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            
                            # Show alert
                            if alert_triggered:
                                with alert_placeholder.container():
                                    st.error(f"🚨 **Attention Alert:** {alert_msg}")
                                    st.markdown("Please refocus on the screen!")
                            else:
                                with alert_placeholder.container():
                                    st.success("✅ **Good Focus**")
                                    st.markdown("Keep it up!")
                        
                        # Add general info
                        cv2.putText(processed_frame, f"Mode: {selected_mode}", 
                                   (10, processed_frame.shape[0] - 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        cv2.putText(processed_frame, f"Eyes: {len(result['detected_pupils'])}", 
                                   (10, processed_frame.shape[0] - 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Convert BGR to RGB for Streamlit
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Display frame
                        video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
                        
                        # Update statistics
                        if st.session_state.current_mode == MODE_ADHD_ASSISTANT:
                            session_duration = time.time() - st.session_state.tracker.session_start_time
                            focus_percentage = (st.session_state.tracker.total_focus_time / max(session_duration, 1)) * 100
                            
                            with stats_placeholder.container():
                                st.metric("Session Duration", f"{int(session_duration)}s")
                                st.metric("Total Focus Time", f"{int(st.session_state.tracker.total_focus_time)}s")
                                st.metric("Focus Percentage", f"{focus_percentage:.1f}%")
                                st.metric("Continuous Focus", f"{st.session_state.tracker.continuous_focus_time:.1f}s")
                        else:
                            with stats_placeholder.container():
                                st.metric("Faces Detected", result['faces_count'])
                                st.metric("Eyes Detected", len(result['detected_pupils']))
                                st.metric("Left Eye", "✅" if result['left_eye_detected'] else "❌")
                                st.metric("Right Eye", "✅" if result['right_eye_detected'] else "❌")
                    
                    frame_count += 1
                    
                    # Add small delay to prevent overwhelming
                    time.sleep(0.05)
                    
                    # Break if camera stopped
                    if not st.session_state.camera_started:
                        break
                
                # Clean up
                cap.release()
                
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.camera_started = False
    else:
        with video_placeholder.container():
            st.info("📹 Click 'Start Camera' to begin eye tracking")
            st.markdown("""
            **Features:**
            - Real-time face and eye detection
            - Pupil tracking
            - ADHD attention assistant
            - Focus time monitoring
            - Attention alerts
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>Eye Tracking Application | Built with Streamlit & OpenCV</p>
        <p>💡 <strong>Tip:</strong> Ensure good lighting and position your face clearly in the camera view</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
