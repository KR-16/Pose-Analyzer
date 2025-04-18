from enum import Enum

class ExerciseType(Enum):
    SQUAT = 1
    PUSHUP = 2
    SHOULDER_PRESS = 3
    BICEP_CURL = 4
    LUNGE = 5

# Exercise configurations
EXERCISE_CONFIG = {
    ExerciseType.SQUAT: {
        'name': 'Squat',
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
        'ideal_tempo': (2, 0, 1, 1),
        'voice_cues': {
            'body_straight': "Keep body straight",
            'elbows_45': "Elbows at 45 degrees",
            'full_range': "Touch chest to floor",
            'rep_complete': "Nice pushup"
        }
    }
}

# MediaPipe settings
MEDIAPIPE_CONFIG = {
    'min_detection_confidence': 0.8,
    'min_tracking_confidence': 0.8,
    'model_complexity': 2
}

# Visualization settings
VISUALIZATION_CONFIG = {
    '3d_axes_limits': (-1, 1),
    'joint_color': 'red',
    'joint_size': 20,
    'bone_color': 'blue',
    'bone_width': 2
}