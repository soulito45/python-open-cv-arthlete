import os
import cv2
import mediapipe as mp
import numpy as np
import time
import pyttsx3
import threading
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional

# GPU Configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

@dataclass
class ExerciseState:
    counter: int = 0
    stage: str = "down"
    prev_stage: str = "down"
    last_rep_time: float = 0
    last_feedback_time: float = 0
    start_time: float = 0
    exercise_duration: int = 60  # 60 seconds countdown
    confidence_threshold: float = 0.7
    min_rep_gap: float = 0.3
    min_feedback_gap: float = 2.0
    arm_up_angle_threshold: float = 70
    arm_down_angle_threshold: float = 110
    leg_spread_threshold: float = 0.35
    leg_closed_threshold: float = 0.15

class PoseDetector:
    def __init__(self):
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=2,
            smooth_landmarks=True,
            enable_segmentation=False
        )
        
        # Drawing styles
        self.correct_joint_style = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=12, circle_radius=8)
        self.incorrect_joint_style = self.mp_drawing.DrawingSpec(
            color=(0, 0, 255), thickness=12, circle_radius=8)
        self.connection_style = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=9, circle_radius=3)

        # Define body landmarks to track
        self.BODY_LANDMARKS = {
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            self.mp_pose.PoseLandmark.LEFT_WRIST,
            self.mp_pose.PoseLandmark.RIGHT_WRIST,
            self.mp_pose.PoseLandmark.LEFT_HIP,
            self.mp_pose.PoseLandmark.RIGHT_HIP,
            self.mp_pose.PoseLandmark.LEFT_KNEE,
            self.mp_pose.PoseLandmark.RIGHT_KNEE,
            self.mp_pose.PoseLandmark.LEFT_ANKLE,
            self.mp_pose.PoseLandmark.RIGHT_ANKLE
        }

        # Define body connections
        self.BODY_CONNECTIONS = [
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_ELBOW),
            (self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.LEFT_WRIST),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_ELBOW),
            (self.mp_pose.PoseLandmark.RIGHT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_WRIST),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER),
            (self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.LEFT_HIP),
            (self.mp_pose.PoseLandmark.RIGHT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP),
            (self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.LEFT_KNEE),
            (self.mp_pose.PoseLandmark.RIGHT_HIP, self.mp_pose.PoseLandmark.RIGHT_KNEE),
            (self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.LEFT_ANKLE),
            (self.mp_pose.PoseLandmark.RIGHT_KNEE, self.mp_pose.PoseLandmark.RIGHT_ANKLE)
        ]

    def calculate_angle(self, a, b, c) -> float:
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle = np.degrees(angle)
        
        return angle

    def check_form(self, landmarks) -> Tuple[bool, List[str]]:
        """Check if jumping jack form is correct"""
        issues = []
        
        # Get landmarks
        shoulders = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        ]
        hips = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        ]
        knees = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ]
        ankles = [
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        ]
        
        # Check shoulder level
        shoulder_slope = abs(shoulders[0].y - shoulders[1].y)
        if shoulder_slope > 0.1:
            issues.append("Keep shoulders level")
        
        # Check hip level
        hip_slope = abs(hips[0].y - hips[1].y)
        if hip_slope > 0.1:
            issues.append("Keep hips level")
        
        # Check knee alignment
        for i in range(2):
            knee_angle = self.calculate_angle(hips[i], knees[i], ankles[i])
            if knee_angle < 160:
                issues.append("Keep legs straight")
                break
                
        return len(issues) == 0, issues

    # In PoseDetector class, modify the analyze_jumping_jack method:
    def analyze_jumping_jack(self, results, state: ExerciseState) -> Optional[dict]:
        """Analyze jumping jack movement and count reps"""
        if not results.pose_landmarks:
            return None
            
        landmarks = results.pose_landmarks.landmark
        
        # Calculate arm angles
        left_arm_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
        )
        
        right_arm_angle = self.calculate_angle(
            landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        )
        
        # Calculate leg spread
        leg_spread = abs(
            landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x -
            landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value].x
        )
        
        # Check form
        good_form, form_issues = self.check_form(landmarks)
        
        # Debug information
        debug_info = {
            "left_arm_angle": left_arm_angle,
            "right_arm_angle": right_arm_angle,
            "leg_spread": leg_spread,
            "good_form": good_form
        }
        
        # Determine exercise stage
        current_stage = state.stage  # Default to current stage
        
        # Check for "up" position
        if (left_arm_angle < state.arm_up_angle_threshold and 
            right_arm_angle < state.arm_up_angle_threshold and 
            leg_spread > state.leg_spread_threshold):
            current_stage = "up"
        # Check for "down" position
        elif (left_arm_angle > state.arm_down_angle_threshold and 
              right_arm_angle > state.arm_down_angle_threshold and 
              leg_spread < state.leg_closed_threshold):
            current_stage = "down"
        
        # Count rep only when completing a full cycle with good form
        current_time = time.time()
        rep_counted = False
        
        if (current_stage != state.stage):  # Stage has changed
            if (current_stage == "down" and state.stage == "up" and 
                (current_time - state.last_rep_time) > state.min_rep_gap):
                state.counter += 1
                state.last_rep_time = current_time
                rep_counted = True
            state.prev_stage = state.stage
            state.stage = current_stage
        
        return {
            "form": good_form,
            "issues": form_issues,
            "stage": current_stage,
            "counter": state.counter,
            "rep_counted": rep_counted,
            "debug_info": debug_info
        }
def draw_pose(self, image, landmarks, good_form: bool):
        """Draw pose landmarks and connections"""
        height, width, _ = image.shape
        
        # Draw connections
        for connection in self.BODY_CONNECTIONS:
            start = landmarks[connection[0].value]
            end = landmarks[connection[1].value]
            
            start_point = (int(start.x * width), int(start.y * height))
            end_point = (int(end.x * width), int(end.y * height))
            
            # Always use white color (255, 255, 255) for connections
            cv2.line(image, start_point, end_point, 
                    (255, 255, 255),  # White color
                    self.connection_style.thickness)
        
        # Draw joints
        for landmark in self.BODY_LANDMARKS:
            point = landmarks[landmark.value]
            x, y = int(point.x * width), int(point.y * height)
            cv2.circle(image, (x, y), 4, (255, 255, 255), -1)

def draw_timer(self, image, state: ExerciseState):
        """Draw countdown timer"""
        height, width, _ = image.shape
        
        if state.start_time == 0:
            state.start_time = time.time()
        
        elapsed_time = time.time() - state.start_time
        remaining_time = max(0, state.exercise_duration - int(elapsed_time))
        
        # Create background for timer
        timer_bg_radius = 45
        cv2.circle(image, (width - 60, 60), timer_bg_radius, (0, 0, 0), -1)
        cv2.circle(image, (width - 60, 60), timer_bg_radius, (255, 255, 255), 2)
        
        # Draw timer text
        timer_text = f"{remaining_time}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        text_size = cv2.getTextSize(timer_text, font, font_scale, 2)[0]
        
        text_x = width - 60 - text_size[0] // 2
        text_y = 60 + text_size[1] // 2
        
        cv2.putText(image, timer_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        
        return remaining_time <= 0

def draw_rep_counter(self, image, counter: int):
        """Draw rep counter with circular background"""
        # Create background circle
        radius = 35
        center_x = 60
        center_y = 60
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 0), -1)
        cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Draw counter text
        text = f"{counter}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        
        text_x = center_x - text_size[0] // 2
        text_y = center_y + text_size[1] // 2
        
        cv2.putText(image, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

def draw_debug_info(self, image, debug_info: dict):
        """Draw debug information on the frame"""
        y_position = 150
        for key, value in debug_info.items():
            text = f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}"
            cv2.putText(image, text, (10, y_position),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_position += 30

def process_frame(self, frame):
        """Process a single frame"""
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        
        return frame, results

class SpeechThread(threading.Thread):
    def __init__(self, speech_queue: Queue):
        super().__init__()
        self.daemon = True
        self.running = True
        self.speech_queue = speech_queue
        self.engine = None
    
    def initialize(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
    
    def run(self):
        self.initialize()
        while self.running:
            try:
                if not self.speech_queue.empty():
                    text = self.speech_queue.get()
                    self.engine.say(text)
                    self.engine.runAndWait()
                time.sleep(0.1)
            except Exception as e:
                print(f"Speech error: {e}")
                continue

    def stop(self):
        self.running = False

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize detector and state
    detector = PoseDetector()
    state = ExerciseState()
    
    # Initialize speech queue and thread
    speech_queue = Queue()
    speech_thread = SpeechThread(speech_queue)
    speech_thread.start()

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            frame, results = detector.process_frame(frame)

            if results.pose_landmarks:
                # Analyze jumping jack movement
                analysis = detector.analyze_jumping_jack(results, state)
                
                if analysis:
                    # Draw pose
                    detector.draw_pose(frame, results.pose_landmarks.landmark, analysis['form'])
                    
                    # Draw rep counter
                    detector.draw_rep_counter(frame, analysis['counter'])
                    
                    # Draw timer and check if exercise is complete
                    exercise_complete = detector.draw_timer(frame, state)
                    
                    # Provide audio feedback
                    current_time = time.time()
                    if analysis['rep_counted']:
                        speech_queue.put(f"Rep {analysis['counter']}")
                    
                    if not analysis['form'] and (current_time - state.last_feedback_time) > state.min_feedback_gap:
                        feedback = ". ".join(analysis['issues'])
                        speech_queue.put(feedback)
                        state.last_feedback_time = current_time
                    
                    if exercise_complete:
                        speech_queue.put(f"Exercise complete! You did {state.counter} repetitions.")
                        break

                    # Draw debug information if needed
                    # detector.draw_debug_info(frame, analysis['debug_info'])

            # Display frame
            cv2.imshow('Jumping Jack Detector', frame)

            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Cleanup
        speech_thread.stop()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
