import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from ..config import VISUALIZATION_CONFIG

class PoseVisualizer3D:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Configure axes
        limits = VISUALIZATION_CONFIG['3d_axes_limits']
        self.ax.set_xlim3d(*limits)
        self.ax.set_ylim3d(*limits)
        self.ax.set_zlim3d(*limits)
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Motion Visualization')
        
        # Initialize skeleton elements
        self.lines = []
        self.points = None
        
        # Set initial view
        self.ax.view_init(elev=20, azim=45)
    
    def update(self, landmarks):
        """Update the 3D visualization with new landmarks"""
        if landmarks is None:
            return
        
        # Clear previous frame
        for line in self.lines:
            line.remove()
        if self.points:
            self.points.remove()
        
        self.lines = []
        
        # Extract coordinates
        x_coords = [landmark.x for landmark in landmarks.landmark]
        y_coords = [landmark.y for landmark in landmarks.landmark]
        z_coords = [-landmark.z for landmark in landmarks.landmark]  # Invert Z for better view
        
        # Plot joints
        self.points = self.ax.scatter(
            x_coords, y_coords, z_coords,
            c=VISUALIZATION_CONFIG['joint_color'],
            marker='o',
            s=VISUALIZATION_CONFIG['joint_size']
        )
        
        # Draw bones
        for connection in self.mp_pose.POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            
            line = self.ax.plot(
                [x_coords[start_idx], x_coords[end_idx]],
                [y_coords[start_idx], y_coords[end_idx]],
                [z_coords[start_idx], z_coords[end_idx]],
                VISUALIZATION_CONFIG['bone_color'],
                linewidth=VISUALIZATION_CONFIG['bone_width']
            )
            self.lines.extend(line)
        
        plt.draw()
        plt.pause(0.001)