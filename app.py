import cv2
import numpy as np
import mediapipe as mp
import time
from enum import Enum
import json
from datetime import datetime
import pyttsx3  # Text-to-speech library
import threading
import queue

class ExerciseType(Enum):
    SQUAT = 1
    PUSHUP = 2
    SHOULDER_PRESS = 3
    BICEP_CURL = 4
    LUNGE = 5

class VoiceFeedback:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Slower speech rate
        self.engine.setProperty('volume', 1.0)
        self.message_queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.start()
    
    def _process_queue(self):
        while self.running or not self.message_queue.empty():
            try:
                msg = self.message_queue.get(timeout=1)
                self.engine.say(msg)
                self.engine.runAndWait()
                self.message_queue.task_done()
            except queue.Empty:
                continue
    
    def speak(self, message, priority=False):
        if priority:
            # Clear queue for important messages
            with self.message_queue.mutex:
                self.message_queue.queue.clear()
        self.message_queue.put(message)
    
    def stop(self):
        self.running = False
        self.thread.join()

class PoseTracker:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8,
            model_complexity=2
        )
        
        # Voice feedback system
        self.voice = VoiceFeedback()
        
        # Exercise configuration
        self.exercise_config = {
            ExerciseType.SQUAT: {
                'name': 'Squat',
                'form_rules': self.check_squat_form,
                'rep_phase_detection': self.detect_squat_phase,
                'ideal_tempo': (3, 0, 1, 1),  # eccentric, bottom, concentric, top
                'voice_cues': {
                    'depth_ok': "Good depth",
                    'depth_low': "Go deeper",
                    'knees_forward': "Keep knees behind toes",
                    'chest_up': "Chest up",
                    'rep_complete': "Good rep"
                }
            },
            ExerciseType.PUSHUP: {
                'name': 'Pushup',
                'form_rules': self.check_pushup_form,
                'rep_phase_detection': self.detect_pushup_phase,
                'ideal_tempo': (2, 0, 1, 1),
                'voice_cues': {
                    'body_straight': "Keep body straight",
                    'elbows_45': "Elbows at 45 degrees",
                    'full_range': "Touch chest to floor",
                    'rep_complete': "Nice pushup"
                }
            }
        }
        
        # Workout tracking
        self.current_exercise = None
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.workout_history = []
        self.load_history()
        
        # Feedback system
        self.feedback_messages = []
        self.performance_stats = {}
        self.last_feedback_time = 0
        self.feedback_cooldown = 2  # seconds
        
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
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def check_squat_form(self, landmarks):
        feedback = []
        voice_feedback = []
        
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Knee alignment check
        knee_ankle_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if knee_ankle_angle < 160:
            feedback.append("Knees too far forward")
            voice_feedback.append(('knees_forward', True))
        
        # Depth check
        hip_knee_diff = abs(left_hip.y - left_knee.y)
        if hip_knee_diff < 0.15:
            feedback.append("Good depth")
            voice_feedback.append(('depth_ok', False))
        else:
            feedback.append("Aim for deeper squat")
            voice_feedback.append(('depth_low', True))
        
        # Torso position
        shoulder_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        if shoulder_hip_angle < 160:
            feedback.append("Keep chest up")
            voice_feedback.append(('chest_up', True))
        
        # Process voice feedback
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            for cue, important in voice_feedback:
                if important or np.random.random() < 0.3:  # 30% chance for non-critical feedback
                    self.voice.speak(
                        self.exercise_config[self.current_exercise]['voice_cues'][cue],
                        priority=important
                    )
                    self.last_feedback_time = current_time
                    break  # Only say one thing at a time
        
        return feedback
    
    def check_pushup_form(self, landmarks):
        feedback = []
        voice_feedback = []
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Body alignment
        shoulder_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_wrist)
        if shoulder_hip_angle < 170:
            feedback.append("Keep body straight")
            voice_feedback.append(('body_straight', True))
        
        # Elbow position
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if not (45 <= elbow_angle <= 60):
            feedback.append("Elbows at 45 degrees")
            voice_feedback.append(('elbows_45', True))
        
        # Range of motion
        shoulder_wrist_diff = abs(left_shoulder.y - left_wrist.y)
        if shoulder_wrist_diff < 0.1:
            feedback.append("Go deeper")
            voice_feedback.append(('full_range', True))
        
        # Process voice feedback
        current_time = time.time()
        if current_time - self.last_feedback_time > self.feedback_cooldown:
            for cue, important in voice_feedback:
                if important or np.random.random() < 0.3:
                    self.voice.speak(
                        self.exercise_config[self.current_exercise]['voice_cues'][cue],
                        priority=important
                    )
                    self.last_feedback_time = current_time
                    break
        
        return feedback
    
    def detect_squat_phase(self, landmarks):
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        
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
        if new_phase != self.current_phase:
            if self.current_phase:
                phase_duration = time.time() - self.last_phase_change
                self.phase_times.append((self.current_phase, phase_duration))
            
            # Check for completed rep
            if (self.current_phase == "bottom" and new_phase == "concentric") or \
               (self.current_phase == "top" and new_phase == "eccentric"):
                self.rep_count += 1
                self.feedback_messages.append(f"Rep {self.rep_count} completed!")
                
                # Voice feedback for completed rep
                self.voice.speak(
                    self.exercise_config[self.current_exercise]['voice_cues']['rep_complete'],
                    priority=False
                )
                
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
        success, frame = self.cap.read()
        if not success:
            return None
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            if self.current_exercise:
                config = self.exercise_config[self.current_exercise]
                
                form_feedback = config['form_rules'](landmarks)
                self.feedback_messages.extend(form_feedback)
                
                new_phase = config['rep_phase_detection'](landmarks)
                self.track_reps(new_phase)
                
                if self.current_exercise == ExerciseType.SQUAT:
                    self.prev_hip_y = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP].y
                elif self.current_exercise == ExerciseType.PUSHUP:
                    self.prev_wrist_y = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST].y
            
            self.mp_drawing.draw_landmarks(
                image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66)), 
                self.mp_drawing.DrawingSpec(color=(245, 66, 230)))
        
        return image
    
    def display_feedback(self, image):
        if self.current_exercise:
            cv2.putText(image, f"Exercise: {self.exercise_config[self.current_exercise]['name']}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, f"Reps: {self.rep_count} | Set: {self.set_count}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if self.current_phase:
                cv2.putText(image, f"Phase: {self.current_phase}", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, message in enumerate(self.feedback_messages[-3:], 1):
            cv2.putText(image, message, (10, 120 + 30 * i), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        self.feedback_messages = []
        
        return image
    
    def start_workout(self, exercise_type):
        self.current_exercise = exercise_type
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.performance_stats = {}
        self.feedback_messages = []
        
        # Initial voice instructions
        exercise_name = self.exercise_config[exercise_type]['name']
        self.voice.speak(f"Starting {exercise_name} workout. Let's begin!", priority=True)
    
    def end_workout(self):
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
            
            # Final voice summary
            self.voice.speak(
                f"Workout complete. You did {self.rep_count} reps across {self.set_count} sets. Great job!",
                priority=True
            )
            
            self.current_exercise = None
    
    def run(self):
        print("Real-Time Exercise Tracker with Voice Feedback")
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
                
                display_frame = self.display_feedback(processed_frame)
                cv2.imshow('Exercise Tracker', display_frame)
                
                key = cv2.waitKey(10)
                if key == ord('q'):
                    self.end_workout()
                    break
                elif key == ord('n'):
                    self.set_count += 1
                    self.rep_count = 0
                    self.voice.speak(f"Starting set {self.set_count}", priority=True)
                elif key == ord(' '):
                    # Manual feedback trigger
                    self.last_feedback_time = 0  # Reset cooldown
        
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.voice.stop()

if __name__ == "__main__":
    tracker = PoseTracker()
    tracker.run()