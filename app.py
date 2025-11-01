from flask import Flask, render_template, Response, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from gesture_detector import GestureDetector
import datetime
import os
import time
from threading import Lock
import logging

# Reduce Flask logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photobooth_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize gesture detector with optimized settings
gesture_detector = GestureDetector()

# Optimize MediaPipe settings if possible
try:
    gesture_detector.hands.min_detection_confidence = 0.6  # Lower for speed
    gesture_detector.hands.min_tracking_confidence = 0.5   # Lower for speed
except:
    pass  # If gesture_detector doesn't expose these settings

# Session management
if not os.path.exists("sessions"):
    os.mkdir("sessions")

SESSION_DIR = None
current_state = {
    'state': 'PROMPT_TIMER',
    'timer_value': None,
    'countdown_end': None,
    'detected_gesture': None,
    'last_count': None,
    'count_streak': 0,
    'thumb_up_streak': 0,
    'fist_streak': 0,
    'capture_triggered': False  # Track if capture was triggered
}
state_lock = Lock()

CONSECUTIVE_REQUIRED = 5

# ==============================
# Routes
# ==============================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sessions/<path:filename>')
def serve_photo(filename):
    return send_from_directory('sessions', filename)

# ==============================
# WebSocket Events
# ==============================

@socketio.on('connect')
def handle_connect():
    global SESSION_DIR
    SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(SESSION_DIR, exist_ok=True)
    print(f"✅ Client connected. Session: {SESSION_DIR}")
    emit('connected', {'session': SESSION_DIR})

@socketio.on('disconnect')
def handle_disconnect():
    print("❌ Client disconnected")

@socketio.on('video_frame')
def handle_video_frame(data):
    global current_state
    
    try:
        # Decode base64 image - OPTIMIZED
        img_str = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            emit('state_update', {
                'frame': data['image'],  # Send back original
                'state': current_state['state'],
                'timer_value': current_state['timer_value'],
                'gesture': None,
                'countdown': get_countdown(),
                'streak_progress': get_streak_progress()
            })
            return
        
        # Detect gesture
        frame, gesture_name = gesture_detector.detect_gesture(frame)
        
        with state_lock:
            current_state['detected_gesture'] = gesture_name
            process_state_machine(gesture_name)
            
            # Encode frame back to base64 with aggressive JPEG compression
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            
            if not success:
                return
                
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Send state and frame back to client
            emit('state_update', {
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'state': current_state['state'],
                'timer_value': current_state['timer_value'],
                'gesture': gesture_name,
                'countdown': get_countdown(),
                'streak_progress': get_streak_progress(),
                'trigger_capture': (current_state['state'] == 'CAPTURE_DONE')  # Signal to capture
            })
    
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        # Send minimal response to keep client running
        emit('state_update', {
            'frame': data.get('image', ''),
            'state': current_state['state'],
            'timer_value': current_state['timer_value'],
            'gesture': None,
            'countdown': get_countdown(),
            'streak_progress': get_streak_progress(),
            'trigger_capture': False
        })

def process_state_machine(gesture_name):
    """Process the photobooth state machine"""
    global current_state
    
    state = current_state['state']
    current_time = time.time()
    
    # Map gestures to finger counts
    finger_count_map = {
        "One Finger": 1,
        "Peace Sign": 2,
        "Three Fingers": 3,
        "Four Fingers": 4,
        "Open Palm": 5
    }
    
    detected_count = finger_count_map.get(gesture_name)
    thumb_up = (gesture_name == "Thumbs Up")
    fist_detected = (gesture_name == "Fist")
    
    # STATE: PROMPT_TIMER
    if state == 'PROMPT_TIMER':
        if detected_count and 1 <= detected_count <= 5:
            current_state['state'] = 'DETECTING_FINGERS'
            current_state['last_count'] = detected_count
            current_state['count_streak'] = 1
    
    # STATE: DETECTING_FINGERS
    elif state == 'DETECTING_FINGERS':
        if detected_count:
            if detected_count == current_state['last_count'] and 1 <= detected_count <= 5:
                current_state['count_streak'] += 1
                if current_state['count_streak'] >= CONSECUTIVE_REQUIRED:
                    current_state['timer_value'] = detected_count
                    current_state['state'] = 'TIMER_SET'
                    current_state['count_streak'] = 0
                    print(f"⏱ Timer set to: {detected_count}s")
            else:
                current_state['count_streak'] = 1
                current_state['last_count'] = detected_count
        else:
            current_state['state'] = 'PROMPT_TIMER'
            current_state['last_count'] = None
            current_state['count_streak'] = 0
    
    # STATE: TIMER_SET
    elif state == 'TIMER_SET':
        current_state['state'] = 'AWAIT_THUMBS_UP'
        current_state['thumb_up_streak'] = 0
        current_state['fist_streak'] = 0
    
    # STATE: AWAIT_THUMBS_UP
    elif state == 'AWAIT_THUMBS_UP':
        if thumb_up:
            current_state['thumb_up_streak'] += 1
            current_state['fist_streak'] = 0
            if current_state['thumb_up_streak'] >= CONSECUTIVE_REQUIRED:
                current_state['countdown_end'] = current_time + current_state['timer_value']
                current_state['state'] = 'COUNTDOWN'
                print(f"▶ Starting countdown: {current_state['timer_value']}s")
        elif fist_detected:
            current_state['fist_streak'] += 1
            current_state['thumb_up_streak'] = 0
            if current_state['fist_streak'] >= CONSECUTIVE_REQUIRED:
                print("🔄 Resetting timer")
                reset_to_prompt()
        else:
            current_state['thumb_up_streak'] = 0
            current_state['fist_streak'] = 0
    
    # STATE: COUNTDOWN
    elif state == 'COUNTDOWN':
        remaining = get_countdown()
        if remaining is not None and remaining <= 0:
            current_state['state'] = 'CAPTURE_DONE'
            print("📸 Triggering photo capture...")
    
    # STATE: CAPTURE_DONE
    elif state == 'CAPTURE_DONE':
        # Wait for photo to be saved, then reset
        pass

def reset_to_prompt():
    """Reset state machine to initial state"""
    global current_state
    current_state = {
        'state': 'PROMPT_TIMER',
        'timer_value': None,
        'countdown_end': None,
        'detected_gesture': None,
        'last_count': None,
        'count_streak': 0,
        'thumb_up_streak': 0,
        'fist_streak': 0
    }

def get_countdown():
    """Get remaining countdown time"""
    if current_state['state'] == 'COUNTDOWN' and current_state['countdown_end']:
        remaining = current_state['countdown_end'] - time.time()
        return max(0, int(round(remaining)))
    return None

def get_streak_progress():
    """Get progress for gesture streaks"""
    if current_state['state'] == 'DETECTING_FINGERS':
        return {'current': current_state['count_streak'], 'required': CONSECUTIVE_REQUIRED}
    elif current_state['state'] == 'AWAIT_THUMBS_UP':
        if current_state['thumb_up_streak'] > 0:
            return {'current': current_state['thumb_up_streak'], 'required': CONSECUTIVE_REQUIRED}
        elif current_state['fist_streak'] > 0:
            return {'current': current_state['fist_streak'], 'required': CONSECUTIVE_REQUIRED}
    return None

@socketio.on('save_photo')
def handle_save_photo(data):
    """Save captured photo"""
    global SESSION_DIR
    try:
        # Ensure session directory exists
        if SESSION_DIR is None or not os.path.exists(SESSION_DIR):
            SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(SESSION_DIR, exist_ok=True)
            print(f"📁 Created session directory: {SESSION_DIR}")
        
        # Decode image
        img_str = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        
        # Generate filename with full path
        timestamp = datetime.datetime.now().strftime('%H%M%S')
        filename = f"photo_{timestamp}.png"
        full_path = os.path.join(SESSION_DIR, filename)
        
        # Save file
        with open(full_path, 'wb') as f:
            f.write(img_data)
        
        # Verify file was saved
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print(f"✅ [Saved] {full_path} ({file_size} bytes)")
            
            # Send relative path for web access
            relative_path = f"{SESSION_DIR}/{filename}"
            emit('photo_saved', {'filename': relative_path})
        else:
            print(f"❌ Failed to save: {full_path}")
        
        # Reset state after photo is saved
        time.sleep(0.5)  # Brief pause
        reset_to_prompt()
        
    except Exception as e:
        print(f"❌ Error saving photo: {e}")
        import traceback
        traceback.print_exc()
        # Still reset state even if save failed
        reset_to_prompt()

# ==============================
# Run App
# ==============================

if __name__ == '__main__':
    print("=" * 50)
    print("🎉 PhotoBooth Web App Starting...")
    print("📸 Open your browser to: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)