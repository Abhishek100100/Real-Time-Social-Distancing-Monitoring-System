import cv2
import numpy as np
from datetime import datetime
from config.settings import (
    SAFE_COLOR, VIOLATION_COLOR, TEXT_COLOR
)

class Visualizer:
    @staticmethod
    def draw_detections(frame, results, violations):
        for i, (prob, bbox, centroid) in enumerate(results):
            color = VIOLATION_COLOR if i in violations else SAFE_COLOR
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid

            # Draw bounding box
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            
            # Draw confidence text
            text = f"{prob:.2f}"
            cv2.putText(frame, text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw centroid
            cv2.circle(frame, (cX, cY), 5, color, -1)

    @staticmethod
    def draw_violation_info(frame, violation_count, total_people):
        violation_percent = (violation_count / max(1, total_people)) * 100
        text = f"Violations: {violation_count}/{total_people} ({violation_percent:.1f}%)"
        cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, VIOLATION_COLOR, 2)

    @staticmethod
    def draw_timestamp(frame):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)