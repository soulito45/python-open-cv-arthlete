import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
from queue import Queue

# Set up environment for GPU usage if available
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize MediaPipe with body-only tracking
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False
)

# Custom drawing styles
correct_joint_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=12, circle_radius=8)
incorrect_joint_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=12, circle_radius=8)
connection_style = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=9, circle_radius=3)


# Global variables
counter = 0
stage = None
prev_stage = None
speech_queue = Queue()
last_rep_time = 0
last_feedback_time = 0
MIN_REP_GAP = 0.5
MIN_FEEDBACK_GAP = 2.0
start_time = None
timer_active = False
timer_duration = 60

class SpeechThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self, daemon=True)
        self.engine = None
        self.running = True
        self.last_spoken = ""

    def run(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        while self.running:
            try:
                message = speech_queue.get(timeout=0.1)
                if message is not None and message != self.last_spoken:
                    self.engine.say(str(message))
                    self.engine.runAndWait()
                    self.last_spoken = message
            except:
                continue

    def stop(self):
        self.running = False
        if self.engine:
            self.engine.stop()

def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def check_form(landmarks):
    """Check push-up form with comprehensive metrics"""
    issues = []
    
    # Get relevant landmarks
    shoulders = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]]
    elbows = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW],
              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]]
    wrists = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST],
              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]]
    hips = [landmarks[mp_pose.PoseLandmark.LEFT_HIP],
            landmarks[mp_pose.PoseLandmark.RIGHT_HIP]]
    
    # Calculate angles
    left_elbow_angle = calculate_angle(shoulders[0], elbows[0], wrists[0])
    right_elbow_angle = calculate_angle(shoulders[1], elbows[1], wrists[1])
    avg_elbow_angle = (left_elbow_angle + right_elbow_angle) / 2
    
    # Back alignment
    back_angle = calculate_angle(
        shoulders[0],
        hips[0],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    )
    
    # Form checks with lenient thresholds
    if avg_elbow_angle < 60:
        issues.append("Going too low")
    elif avg_elbow_angle > 160:
        issues.append("Extend arms more")
        
    if abs(left_elbow_angle - right_elbow_angle) > 20:
        issues.append("Keep arms even")
        
    if not (150 <= back_angle <= 190):
        issues.append("Straighten your back")

    return len(issues) == 0, issues

def detect_pose(image):
    """Detect and visualize pose without any face tracking"""
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_rgb.flags.writeable = False
    results = pose.process(image_rgb)
    image_rgb.flags.writeable = True
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        good_form, _ = check_form(landmarks)
        
        # Define all landmarks from neck down (exclude face/head)
        BODY_LANDMARKS = {
            mp_pose.PoseLandmark.LEFT_SHOULDER,
            mp_pose.PoseLandmark.RIGHT_SHOULDER,
            mp_pose.PoseLandmark.LEFT_ELBOW,
            mp_pose.PoseLandmark.RIGHT_ELBOW,
            mp_pose.PoseLandmark.LEFT_WRIST,
            mp_pose.PoseLandmark.RIGHT_WRIST,
            mp_pose.PoseLandmark.LEFT_HIP,
            mp_pose.PoseLandmark.RIGHT_HIP,
            mp_pose.PoseLandmark.LEFT_KNEE,
            mp_pose.PoseLandmark.RIGHT_KNEE,
            mp_pose.PoseLandmark.LEFT_ANKLE,
            mp_pose.PoseLandmark.RIGHT_ANKLE
        }
        
        # Define connections for body only
        body_connections = [
            # Arms
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
            (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
            (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
            # Shoulders
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
            # Torso
            (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
            (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP),
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
            # Legs
            (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
            (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
            (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
            (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]
        
        # Create connection style
        connection_style = mp_drawing.DrawingSpec(
            color=(255, 255, 255),  # White lines
            thickness=9,            # Thicker lines for smoothness
            circle_radius=5
        )
        
        # Draw pose connections
        for connection in body_connections:
            start = landmarks[connection[0].value]
            end = landmarks[connection[1].value]
            
            # Convert normalized coordinates to pixel coordinates
            height, width, _ = image.shape
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            
            # Draw line
            cv2.line(image, start_point, end_point, connection_style.color, connection_style.thickness)
        
        # Draw joint points
        for landmark in BODY_LANDMARKS:
            idx = landmark.value
            point = landmarks[idx]
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(image, (x, y), 4, (255, 255, 255), -1)
    
    return image, results

def analyze_pushup_position(results):
    """Analyze push-up position and count reps with improved logic."""
    global counter, stage, prev_stage, last_rep_time, last_feedback_time, timer_active

    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    good_form, issues = check_form(landmarks)

    # Extract relevant joints
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Calculate elbow angles
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    elbow_angle = (left_angle + right_angle) / 2

    # **Check if hands are on the ground**
    hands_on_ground = (left_wrist.y > left_elbow.y) and (right_wrist.y > right_elbow.y)

    # **Push-up stage detection (only when hands are on the ground)**
    if hands_on_ground:
        if elbow_angle > 150:
            current_stage = "up"
        elif elbow_angle < 100:
            current_stage = "down"
        else:
            current_stage = stage
    else:
        current_stage = None  # Ignore arm movement if hands are not placed on the ground

    # Start timer on first valid movement
    if not timer_active and current_stage == "down":
        timer_active = True
        global start_time
        start_time = time.time()

    current_time = time.time()

    # Count rep only if hands remain on the ground throughout
    if hands_on_ground and current_stage == "up" and stage == "down":
        if (current_time - last_rep_time) > MIN_REP_GAP:
            counter += 1
            speech_queue.put(str(counter))
            last_rep_time = current_time

    stage = current_stage

    # Provide feedback for incorrect form
    if not good_form and (current_time - last_feedback_time) > MIN_FEEDBACK_GAP:
        if issues:
            speech_queue.put(issues[0])
            last_feedback_time = current_time

    return {
        'count': counter,
        'stage': stage,
        'form': good_form,
        'issues': issues,
        'elbow_angle': elbow_angle,
        'hands_on_ground': hands_on_ground
    }

def draw_countdown(image, countdown):
    """Draw initial 5 second countdown"""
    frame_height, frame_width, _ = image.shape
    center_x = frame_width // 2
    center_y = frame_height // 2
    
    # Draw large countdown number
    cv2.putText(image, str(countdown),
                (center_x - 50, center_y + 50),
                cv2.FONT_HERSHEY_DUPLEX,
                4.0,
                (255, 255, 255),
                4,
                cv2.LINE_AA)
# Remove the original draw_timer function and replace it with this:
def draw_timer(image, elapsed_time, timer_duration=60):
    """Draw timer display showing countdown from 60 seconds"""
    frame_height, frame_width, _ = image.shape
    
    # Calculate remaining time
    remaining_time = max(0, timer_duration - elapsed_time)
    minutes = int(remaining_time) // 60
    seconds = int(remaining_time) % 60
    
    # Format timer text (counting down)
    time_text = f"{minutes:02d}:{seconds:02d}"
    
    # Digital timer at bottom
    bottom_font = cv2.FONT_HERSHEY_SIMPLEX
    bottom_scale = 1.5
    bottom_thickness = 2
    
    text_size = cv2.getTextSize(time_text, bottom_font, bottom_scale, bottom_thickness)[0]
    text_x = frame_width // 2 - (text_size[0] // 2)
    text_y = frame_height - 40  # 40 pixels from bottom
    
    cv2.putText(image, time_text,
                (text_x, text_y),
                bottom_font,
                bottom_scale,
                (0, 0, 0),
                bottom_thickness,
                cv2.LINE_AA)
    
    return image

def draw_rep_counter(image):
    """Draw rep counter in top center with circular background"""
    frame_height, frame_width, _ = image.shape
    
    # Position in top center
    center_x = frame_width // 2
    center_y = 60  # Distance from top
    
    # Draw black circle background
    radius = 35
    cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), -1)
    cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)  # White border
    
    # Draw counter text
    text = f"{counter}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    
    # Center text in circle
    text_x = center_x - text_size[0] // 2
    text_y = center_y + text_size[1] // 2
    
    cv2.putText(image, text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA)
def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    speech_thread = SpeechThread()
    speech_thread.start()
    
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (1280, 720))
            frame, results = detect_pose(frame)
            analysis = analyze_pushup_position(results)
            
            if analysis:
                draw_rep_counter(frame)
                if timer_active and start_time is not None:
                    elapsed_time = time.time() - start_time
                    if elapsed_time <= timer_duration:
                        draw_timer(frame, elapsed_time)
            
            cv2.imshow('Push-up Form Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
    finally:
        cap.release()
        cv2.destroyAllWindows()
        speech_thread.stop()

if __name__ == "__main__":
    main()
