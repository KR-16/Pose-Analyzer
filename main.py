import sys
from PyQt5.QtWidgets import QApplication
from modules.pose_tracker import PoseTracker
from modules.voice_feedback import VoiceFeedback
from modules.visualizer_3d import PoseVisualizer3D
from modules.workout_manager import WorkoutManager
from modules.ui import MainWindow

def main():
    app = QApplication(sys.argv)
    
    # Initialize components
    tracker = PoseTracker()
    visualizer = PoseVisualizer3D()
    workout_manager = WorkoutManager()
    voice_feedback = VoiceFeedback()
    
    # Create and show main window
    window = MainWindow(tracker, visualizer, workout_manager, voice_feedback)
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()