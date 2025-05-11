import cv2
import numpy as np
from scipy.spatial import distance as dist

class DistanceCalculator:
    def __init__(self, min_distance_meters, reference_height):
        self.min_distance_meters = min_distance_meters
        self.reference_height = reference_height
        self.homography_matrix = None
    
    def set_homography(self, calib_points):
        """Set perspective transform using 4 calibration points"""
        if len(calib_points) != 4:
            raise ValueError(f"Expected 4 calibration points, got {len(calib_points)}")
        
        # Real-world coordinates (in meters)
        real_world_points = np.array([
            [0, 0],         # Origin (ground)
            [1, 0],         # 1m along ground (X-axis)
            [0, 1],         # 1m perpendicular (Y-axis)
            [0, self.reference_height]  # Head height (Z-axis)
        ], dtype=np.float32)
        
        image_points = np.array(calib_points, dtype=np.float32)
        self.homography_matrix = cv2.getPerspectiveTransform(
            image_points, real_world_points
        )
    
    def pixels_to_meters(self, points):
        """Convert image points to real-world coordinates"""
        if self.homography_matrix is None:
            raise RuntimeError("Homography matrix not set. Calibrate first.")
        
        pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(pts, self.homography_matrix)
        return transformed.reshape(-1, 2)
    
    def find_violations(self, centroids_pixels):
        if self.homography_matrix is None:
            raise RuntimeError("Distance calculator not calibrated")
        
        centroids_meters = self.pixels_to_meters(centroids_pixels)
        D = dist.cdist(centroids_meters, centroids_meters)
        
        violations = set()
        for i in range(len(D)):
            for j in range(i+1, len(D)):
                if D[i,j] < self.min_distance_meters:
                    violations.add(i)
                    violations.add(j)
        return violations