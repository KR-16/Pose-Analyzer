import time
import json
from datetime import datetime
from ..config import EXERCISE_CONFIG, ExerciseType

class WorkoutManager:
    def __init__(self):
        self.current_exercise = None
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.workout_history = []
        self.performance_stats = {}
        self.load_history()
    
    def load_history(self):
        try:
            with open('data/workout_history.json', 'r') as f:
                self.workout_history = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.workout_history = []
    
    def save_history(self):
        with open('data/workout_history.json', 'w') as f:
            json.dump(self.workout_history, f, indent=2)
    
    def start_workout(self, exercise_type):
        self.current_exercise = exercise_type
        self.rep_count = 0
        self.set_count = 1
        self.phase_times = []
        self.current_phase = None
        self.last_phase_change = time.time()
        self.performance_stats = {}
    
    def track_rep_phase(self, new_phase):
        """Track rep phases and count completed reps"""
        if new_phase != self.current_phase:
            # Record phase duration
            if self.current_phase:
                phase_duration = time.time() - self.last_phase_change
                self.phase_times.append((self.current_phase, phase_duration))
            
            # Check for completed rep
            if self._is_rep_completed(new_phase):
                self._handle_rep_completion()
            
            self.current_phase = new_phase
            self.last_phase_change = time.time()
    
    def _is_rep_completed(self, new_phase):
        """Determine if a rep was completed based on phase change"""
        if self.current_exercise == ExerciseType.SQUAT:
            return (self.current_phase == "bottom" and new_phase == "concentric")
        elif self.current_exercise == ExerciseType.PUSHUP:
            return (self.current_phase == "bottom" and new_phase == "concentric")
        return False
    
    def _handle_rep_completion(self):
        """Handle actions when a rep is completed"""
        self.rep_count += 1
        
        # Calculate tempo metrics
        if len(self.phase_times) >= 3:
            eccentric_time = sum(t for p, t in self.phase_times if p == "eccentric")
            concentric_time = sum(t for p, t in self.phase_times if p == "concentric")
            self.performance_stats[f"rep_{self.rep_count}"] = {
                "eccentric": eccentric_time,
                "concentric": concentric_time
            }
        
        self.phase_times = []
    
    def end_workout(self):
        if self.current_exercise:
            workout_data = {
                "date": datetime.now().isoformat(),
                "exercise": EXERCISE_CONFIG[self.current_exercise]['name'],
                "reps": self.rep_count,
                "sets": self.set_count,
                "performance": self.performance_stats
            }
            self.workout_history.append(workout_data)
            self.save_history()
            self.current_exercise = None
    
    def new_set(self):
        self.set_count += 1
        self.rep_count = 0
        self.phase_times = []