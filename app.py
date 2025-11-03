import os
os.environ['GLOG_minloglevel'] = '2'  # Suppress MediaPipe warnings

from flask import Flask, render_template, Response, jsonify, send_from_directory, url_for
from flask_socketio import SocketIO, emit
import cv2
import base64
import numpy as np
from gesture_detector import GestureDetector
import datetime
import time
from threading import Lock
import logging
from PIL import Image, ImageDraw, ImageFont
import io

# ======================================
# Flask Setup
# ======================================
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'photobooth_secret'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    max_http_buffer_size=50 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25
)

# ======================================
# Initialize Gesture Detector
# ======================================
gesture_detector = GestureDetector()
try:
    gesture_detector.hands.min_detection_confidence = 0.6
    gesture_detector.hands.min_tracking_confidence = 0.5
except:
    pass

# ======================================
# Session Management
# ======================================
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
    'capture_count': 0,
    'captured_images': [],
    'strip_filename': None
}
state_lock = Lock()

CONSECUTIVE_REQUIRED = 5
PHOTOS_PER_STRIP = 4

# ======================================
# ROUTES
# ======================================

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/guide')
def guide():
    return render_template('guide.html')

@app.route('/index')
def photobooth():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/sessions/<path:filename>')
def serve_photo(filename):
    return send_from_directory('sessions', filename)

# ======================================
# SOCKET EVENTS
# ======================================

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
        # Decode base64 image
        img_str = data['image'].split(',')[1]
        img_data = base64.b64decode(img_str)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None or frame.size == 0:
            emit('state_update', get_default_state(data['image']))
            return

        # Gesture detection
        try:
            frame, gesture_name = gesture_detector.detect_gesture(frame)
        except Exception as gesture_error:
            print(f"⚠️ Gesture detection error: {gesture_error}")
            gesture_name = None

        with state_lock:
            current_state['detected_gesture'] = gesture_name
            process_state_machine(gesture_name)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            success, buffer = cv2.imencode('.jpg', frame, encode_param)
            if not success:
                return

            frame_base64 = base64.b64encode(buffer).decode('utf-8')

            emit('state_update', {
                'frame': f'data:image/jpeg;base64,{frame_base64}',
                'state': current_state['state'],
                'timer_value': current_state['timer_value'],
                'gesture': gesture_name,
                'countdown': get_countdown(),
                'streak_progress': get_streak_progress(),
                'trigger_capture': (current_state['state'] == 'CAPTURE_DONE'),
                'capture_count': current_state['capture_count'],
                'total_captures': PHOTOS_PER_STRIP,
                'strip_ready': current_state['capture_count'] >= PHOTOS_PER_STRIP,
                'strip_filename': current_state['strip_filename']
            })

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        emit('state_update', get_default_state(data.get('image', '')))

def get_default_state(image):
    return {
        'frame': image,
        'state': current_state['state'],
        'timer_value': current_state['timer_value'],
        'gesture': None,
        'countdown': get_countdown(),
        'streak_progress': get_streak_progress(),
        'trigger_capture': False,
        'capture_count': current_state['capture_count'],
        'total_captures': PHOTOS_PER_STRIP
    }

# ======================================
# STATE MACHINE
# ======================================

def process_state_machine(gesture_name):
    global current_state

    state = current_state['state']
    current_time = time.time()

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

    if state == 'PROMPT_TIMER':
        if detected_count and 1 <= detected_count <= 5:
            current_state.update({'state': 'DETECTING_FINGERS', 'last_count': detected_count, 'count_streak': 1})

    elif state == 'DETECTING_FINGERS':
        if detected_count:
            if detected_count == current_state['last_count']:
                current_state['count_streak'] += 1
                if current_state['count_streak'] >= CONSECUTIVE_REQUIRED:
                    current_state.update({'timer_value': detected_count, 'state': 'TIMER_SET', 'count_streak': 0})
                    print(f"⏱ Timer set to: {detected_count}s")
            else:
                current_state.update({'count_streak': 1, 'last_count': detected_count})
        else:
            reset_to_prompt()

    elif state == 'TIMER_SET':
        current_state.update({'state': 'AWAIT_THUMBS_UP', 'thumb_up_streak': 0, 'fist_streak': 0})

    elif state == 'AWAIT_THUMBS_UP':
        if thumb_up:
            current_state['thumb_up_streak'] += 1
            if current_state['thumb_up_streak'] >= CONSECUTIVE_REQUIRED:
                current_state.update({'countdown_end': current_time + current_state['timer_value'], 'state': 'COUNTDOWN'})
                print(f"▶ Starting countdown: {current_state['timer_value']}s")
        elif fist_detected:
            current_state['fist_streak'] += 1
            if current_state['fist_streak'] >= CONSECUTIVE_REQUIRED:
                print("🔄 Resetting timer")
                reset_to_prompt()
        else:
            current_state['thumb_up_streak'] = current_state['fist_streak'] = 0

    elif state == 'COUNTDOWN':
        if get_countdown() is not None and get_countdown() <= 0:
            current_state.update({'state': 'CAPTURE_DONE', 'countdown_end': None})
            print(f"📸 Capture {current_state['capture_count'] + 1}/{PHOTOS_PER_STRIP}")

def reset_to_prompt():
    global current_state
    current_state.update({
        'state': 'PROMPT_TIMER',
        'timer_value': None,
        'countdown_end': None,
        'detected_gesture': None,
        'last_count': None,
        'count_streak': 0,
        'thumb_up_streak': 0,
        'fist_streak': 0,
        'capture_count': 0,
        'captured_images': [],
        'strip_filename': None
    })

def get_countdown():
    if current_state['state'] == 'COUNTDOWN' and current_state['countdown_end']:
        remaining = current_state['countdown_end'] - time.time()
        return max(0, int(round(remaining)))
    return None

def get_streak_progress():
    if current_state['state'] == 'DETECTING_FINGERS':
        return {'current': current_state['count_streak'], 'required': CONSECUTIVE_REQUIRED}
    elif current_state['state'] == 'AWAIT_THUMBS_UP':
        if current_state['thumb_up_streak'] > 0:
            return {'current': current_state['thumb_up_streak'], 'required': CONSECUTIVE_REQUIRED}
        elif current_state['fist_streak'] > 0:
            return {'current': current_state['fist_streak'], 'required': CONSECUTIVE_REQUIRED}
    return None

# ======================================
# SAVE PHOTOS
# ======================================

@socketio.on('save_photo')
def handle_save_photo(data):
    global SESSION_DIR

    try:
        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            return

        if SESSION_DIR is None or not os.path.exists(SESSION_DIR):
            SESSION_DIR = f"sessions/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(SESSION_DIR, exist_ok=True)

        img_data = data.get('image')
        if not img_data:
            emit('photo_error', {'error': 'No image data'})
            return

        current_state['captured_images'].append(img_data)
        current_state['capture_count'] += 1

        emit('photo_received', {'count': current_state['capture_count'], 'total': PHOTOS_PER_STRIP})

        if current_state['capture_count'] >= PHOTOS_PER_STRIP:
            current_state['state'] = 'STRIP_GENERATING'
            strip_filename = create_photo_strip(current_state['captured_images'], SESSION_DIR)
            if strip_filename:
                current_state['strip_filename'] = strip_filename
                emit('strip_ready', {'filename': strip_filename, 'message': 'Photo strip ready!'})
            reset_to_prompt()
        else:
            time.sleep(1)
            current_state.update({
                'countdown_end': time.time() + current_state['timer_value'],
                'state': 'COUNTDOWN'
            })

    except Exception as e:
        print(f"❌ Error saving photo: {e}")
        reset_to_prompt()
        emit('photo_error', {'error': str(e)})

# ======================================
# PHOTO STRIP CREATION
# ======================================

def create_photo_strip(images, session_dir):
    try:
        PHOTO_WIDTH, PHOTO_HEIGHT, BORDER = 800, 600, 20
        FRAME_COLOR = (173, 216, 230)

        strip_width = PHOTO_WIDTH + BORDER * 2
        strip_height = PHOTO_HEIGHT * PHOTOS_PER_STRIP + BORDER * (PHOTOS_PER_STRIP + 1)
        strip = Image.new('RGB', (strip_width, strip_height), FRAME_COLOR)

        for i, img_data in enumerate(images):
            img_bytes = base64.b64decode(img_data.split(',')[1])
            photo = Image.open(io.BytesIO(img_bytes)).resize((PHOTO_WIDTH, PHOTO_HEIGHT))
            y_pos = BORDER + i * (PHOTO_HEIGHT + BORDER)
            strip.paste(photo, (BORDER, y_pos))

        draw = ImageDraw.Draw(strip)
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
        tw, th = draw.textbbox((0, 0), timestamp, font=font)[2:]
        tx, ty = (strip_width - tw) // 2, strip_height - BORDER + 5
        draw.text((tx + 2, ty + 2), timestamp, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), timestamp, fill=(255, 255, 255), font=font)

        filename = f"strip_{datetime.datetime.now().strftime('%H%M%S')}.png"
        path = os.path.join(session_dir, filename)
        strip.save(path)
        return f"{session_dir}/{filename}" if os.path.exists(path) else None
    except Exception as e:
        print(f"❌ Error creating strip: {e}")
        return None

# ======================================
# RUN APP
# ======================================
if __name__ == '__main__':
    print("=" * 50)
    print("🎉 VisionBooth Starting...")
    print("📸 Open browser at: http://localhost:5000")
    print("=" * 50)
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
