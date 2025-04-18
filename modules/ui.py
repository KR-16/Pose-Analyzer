from PyQt5.QtWidgets import (QMainWindow, QVBoxLayout, QWidget, 
                             QLabel, QPushButton, QComboBox)
from PyQt5.QtCore import Qt, QTimer
import cv2
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self, tracker, visualizer, workout_manager, voice_feedback):
        super().__init__()
        self.tracker = tracker
        self.visualizer = visualizer
        self.workout_manager = workout_manager
        self.voice = voice_feedback
        
        self.setWindowTitle("AI Gym Trainer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.init_ui()
        self.init_camera()
    
    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # Exercise selection
        self.exercise_combo = QComboBox()
        for ex in ExerciseType:
            self.exercise_combo.addItem(ex.name, ex)
        
        # Start/stop buttons
        self.start_btn = QPushButton("Start Workout")
        self.start_btn.clicked.connect(self.start_workout)
        
        self.stop_btn = QPushButton("Stop Workout")
        self.stop_btn.clicked.connect(self.stop_workout)
        
        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        layout.addWidget(self.exercise_combo)
        layout.addWidget(self.start_btn)
        layout.addWidget(self.stop_btn)
        layout.addWidget(self.camera_label)
        
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
    
    def init_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS
    
    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            landmarks = self.tracker.process_frame(frame)
            
            # Update 3D visualizer
            if landmarks:
                self.visualizer.update(landmarks)
                
                # Form analysis if in workout
                if self.workout_manager.current_exercise:
                    self.analyze_form(landmarks)
            
            # Display camera feed
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_label.setPixmap(QPixmap.fromImage(qt_image))
    
    def analyze_form(self, landmarks):
        """Analyze form and provide feedback"""
        exercise_type = self.workout_manager.current_exercise
        form_checker = self.tracker.get_form_checker(exercise_type)
        feedback = form_checker(landmarks)
        
        for feedback_key, is_important in feedback:
            self.voice.give_exercise_feedback(exercise_type, feedback_key, is_important)
    
    def start_workout(self):
        exercise_type = self.exercise_combo.currentData()
        self.workout_manager.start_workout(exercise_type)
        self.voice.speak(f"Starting {exercise_type.name} workout", priority=True)
    
    def stop_workout(self):
        self.workout_manager.end_workout()
        self.voice.speak("Workout completed", priority=True)
    
    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        self.workout_manager.end_workout()
        self.voice.stop()
        event.accept()