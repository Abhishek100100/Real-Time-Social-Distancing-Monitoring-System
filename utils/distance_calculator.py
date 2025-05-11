from scipy.spatial import distance as dist
import numpy as np

class DistanceCalculator:
    def __init__(self, min_distance_pixels):
        self.min_distance = min_distance_pixels

    def find_violations(self, centroids):
        violate = set()
        
        if len(centroids) >= 2:
            D = dist.cdist(centroids, centroids, metric="euclidean")
            
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    if D[i, j] < self.min_distance:
                        violate.add(i)
                        violate.add(j)
        
        return violate

    def calculate_violation_percentage(self, violate_count, total_people):
        if total_people == 0:
            return 0.0
        return (violate_count / total_people) * 100