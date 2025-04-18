import cv2
import numpy as np
import mediapipe as mp
import time
from enum import Enum
import json
from datetime import datetime

class ExerciseType(Enum):
    SQUAT = 1
    PUSHUP = 2
    SHOULDER_PRESS = 3
    BICEP_CURL = 4
    LUNGE = 5

class PoseTracker:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=2
        )
        
        # Exercise configuration
        self.exercise_config = {
            ExerciseType.SQUAT: {
                'name': 'Squat',
                'form_rules': self.check_squat_form,
                'rep_phase_detection': self.detect_squat_phase,
                'ideal_tempo': (3, 0, 1, 1)  # eccentric, bottom, concentric, top
            },
            ExerciseType.PUSHUP: {
                'name': 'Pushup',
                'form_rules': self.check_pushup_form,
                'rep_phase_detection': self.detect_pushup_phase,
                'ideal_tempo': (2, 0, 1, 1)
            }
        }
        
        # Workout tracking variables
        self.current_exercise = None
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.workout_history = []
        self.load_history()
        
        # Visual feedback
        self.feedback_messages = []
        self.performance_stats = {}
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def load_history(self):
        try:
            with open('workout_history.json', 'r') as f:
                self.workout_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.workout_history = []
    
    def save_history(self):
        with open('workout_history.json', 'w') as f:
            json.dump(self.workout_history, f, indent=2)
    
    def calculate_angle(self, a, b, c):
        """Calculate the angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def check_squat_form(self, landmarks):
        """Analyze squat form and provide feedback"""
        feedback = []
        
        # Get key landmarks
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        right_knee = landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Check knee alignment
        knee_ankle_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if knee_ankle_angle < 160:
            feedback.append("Knees too far forward")
        
        # Check depth
        hip_knee_diff = abs(left_hip.y - left_knee.y)
        if hip_knee_diff < 0.15:
            feedback.append("Good depth achieved")
        else:
            feedback.append("Aim for deeper squat")
        
        # Check torso position
        shoulder_hip_angle = self.calculate_angle(
            left_shoulder, left_hip, left_knee
        )
        if shoulder_hip_angle < 160:
            feedback.append("Keep chest more upright")
        
        return feedback
    
    def check_pushup_form(self, landmarks):
        """Analyze pushup form and provide feedback"""
        feedback = []
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Check body alignment
        shoulder_hip_angle = self.calculate_angle(
            left_shoulder, left_hip, left_wrist
        )
        if shoulder_hip_angle < 170:
            feedback.append("Keep body straight")
        
        # Check elbow position
        if left_elbow.x > left_shoulder.x + 0.1:
            feedback.append("Elbows flaring out")
        elif left_elbow.x < left_shoulder.x - 0.2:
            feedback.append("Elbows too tucked")
        
        # Check range of motion
        shoulder_wrist_diff = abs(left_shoulder.y - left_wrist.y)
        if shoulder_wrist_diff < 0.1:
            feedback.append("Go deeper in pushup")
        
        return feedback
    
    def detect_squat_phase(self, landmarks):
        """Determine current phase of squat movement"""
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        
        # Calculate vertical difference between hip and knee
        vertical_diff = left_hip.y - left_knee.y
        
        if vertical_diff < 0.1:
            return "bottom"
        elif left_hip.y < left_knee.y + 0.2:
            if left_hip.y < self.prev_hip_y:
                return "concentric"
            else:
                return "eccentric"
        else:
            return "top"
    
    def detect_pushup_phase(self, landmarks):
        """Determine current phase of pushup movement"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        vertical_diff = abs(left_shoulder.y - left_wrist.y)
        
        if vertical_diff < 0.1:
            return "top"
        elif left_wrist.y > left_shoulder.y + 0.2:
            return "bottom"
        elif left_wrist.y < self.prev_wrist_y:
            return "concentric"
        else:
            return "eccentric"
    
    def track_reps(self, new_phase):
        """Track repetitions based on phase changes"""
        if new_phase != self.current_phase:
            # Record phase duration
            if self.current_phase:
                phase_duration = time.time() - self.last_phase_change
                self.phase_times.append((self.current_phase, phase_duration))
            
            # Check for completed rep
            if (self.current_phase == "bottom" and new_phase == "concentric") or \
               (self.current_phase == "top" and new_phase == "eccentric"):
                self.rep_count += 1
                self.feedback_messages.append(f"Rep {self.rep_count} completed!")
                
                # Calculate tempo metrics
                if len(self.phase_times) >= 3:
                    eccentric_time = sum(t for p, t in self.phase_times if p == "eccentric")
                    concentric_time = sum(t for p, t in self.phase_times if p == "concentric")
                    self.performance_stats[f"rep_{self.rep_count}"] = {
                        "eccentric": eccentric_time,
                        "concentric": concentric_time
                    }
                
                self.phase_times = []
            
            self.current_phase = new_phase
            self.last_phase_change = time.time()
    
    def process_frame(self):
        """Process each video frame"""
        success, frame = self.cap.read()
        if not success:
            return None
        
        # Convert to RGB and process with MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        
        # Convert back to BGR for OpenCV
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and analyze
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Exercise-specific processing
            if self.current_exercise:
                config = self.exercise_config[self.current_exercise]
                
                # Form analysis
                form_feedback = config['form_rules'](landmarks)
                self.feedback_messages.extend(form_feedback)
                
                # Rep phase detection
                new_phase = config['rep_phase_detection'](landmarks)
                self.track_reps(new_phase)
                
                # Store previous positions for movement detection
                if self.current_exercise == ExerciseType.SQUAT:
                    self.prev_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
                elif self.current_exercise == ExerciseType.PUSHUP:
                    self.prev_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66)), 
                self.mp_drawing.DrawingSpec(color=(245, 66, 230))
            )
        
        return image
    
    def display_feedback(self, image):
        """Overlay feedback and stats on the image"""
        # Display exercise info
        if self.current_exercise:
            cv2.putText(image, f"Exercise: {self.exercise_config[self.current_exercise]['name']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Reps: {self.rep_count} | Set: {self.set_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self.current_phase:
                cv2.putText(image, f"Phase: {self.current_phase}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display form feedback
        for i, message in enumerate(self.feedback_messages[-3:], 1):
            cv2.putText(image, message, (10, 120 + 30 * i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Clear feedback messages for next frame
        self.feedback_messages = []
        
        return image
    
    def start_workout(self, exercise_type):
        """Initialize a new workout session"""
        self.current_exercise = exercise_type
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.performance_stats = {}
        self.feedback_messages = []
        
        print(f"Started {self.exercise_config[exercise_type]['name']} workout")
    
    def end_workout(self):
        """Finalize the current workout"""
        if self.current_exercise:
            workout_data = {
                "date": datetime.now().isoformat(),
                "exercise": self.exercise_config[self.current_exercise]['name'],
                "reps": self.rep_count,
                "sets": self.set_count,
                "performance": self.performance_stats
            }
            self.workout_history.append(workout_data)
            self.save_history()
            
            print(f"Workout saved: {workout_data}")
            self.current_exercise = None
    
    def run(self):
        """Main application loop"""
        print("Real-Time Exercise Tracker")
        print("Available exercises:")
        for ex in ExerciseType:
            print(f"{ex.value}. {ex.name}")
        
        exercise_choice = int(input("Select exercise (1-5): "))
        self.start_workout(ExerciseType(exercise_choice))
        
        try:
            while self.cap.isOpened():
                processed_frame = self.process_frame()
                if processed_frame is None:
                    break
                
                # Display feedback
                display_frame = self.display_feedback(processed_frame)
                cv2.imshow('Exercise Tracker', display_frame)
                
                # Check for quit command
                key = cv2.waitKey(10)
                if key == ord('q'):
                    self.end_workout()
                    break
                elif key == ord('n'):
                    self.set_count += 1
                    self.rep_count = 0
                    self.feedback_messages.append(f"Starting set {self.set_count}")
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = PoseTracker()
    tracker.run()