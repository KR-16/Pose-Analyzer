import mediapipe as mp
import numpy as np
from ..config import MEDIAPIPE_CONFIG, EXERCISE_CONFIG, ExerciseType

class PoseTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
        self.connections = self.mp_pose.POSE_CONNECTIONS
        
    def process_frame(self, frame):
        """Process a frame and return landmarks"""
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = self.pose.process(image)
        return results.pose_landmarks
    
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(cosine_angle))
    
    def check_squat_form(self, landmarks):
        """Analyze squat form"""
        feedback = []
        
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        
        # Knee alignment
        knee_ankle_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
        if knee_ankle_angle < 160:
            feedback.append(("knees_forward", True))
        
        # Depth
        hip_knee_diff = abs(left_hip.y - left_knee.y)
        if hip_knee_diff < 0.15:
            feedback.append(("depth_ok", False))
        else:
            feedback.append(("depth_low", True))
        
        # Torso position
        shoulder_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_knee)
        if shoulder_hip_angle < 160:
            feedback.append(("chest_up", True))
        
        return feedback
    
    def check_pushup_form(self, landmarks):
        """Analyze pushup form"""
        feedback = []
        
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
        left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        
        # Body alignment
        shoulder_hip_angle = self.calculate_angle(left_shoulder, left_hip, left_wrist)
        if shoulder_hip_angle < 170:
            feedback.append(("body_straight", True))
        
        # Elbow position
        elbow_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
        if not (45 <= elbow_angle <= 60):
            feedback.append(("elbows_45", True))
        
        # Range of motion
        shoulder_wrist_diff = abs(left_shoulder.y - left_wrist.y)
        if shoulder_wrist_diff < 0.1:
            feedback.append(("full_range", True))
        
        return feedback
    
    def get_form_checker(self, exercise_type):
        """Get the appropriate form checker for the exercise"""
        if exercise_type == ExerciseType.SQUAT:
            return self.check_squat_form
        elif exercise_type == ExerciseType.PUSHUP:
            return self.check_pushup_form
        else:
            return lambda landmarks: []