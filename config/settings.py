import os
import json

# Base configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/yolo-coco")

# Detection parameters
MIN_CONFIDENCE = 0.3
NMS_THRESHOLD = 0.3

# Social distance parameters
MIN_SAFE_DISTANCE = 2.0  # meters (WHO guideline)
REFERENCE_OBJECT_HEIGHT = 1.8  # Average person height in meters

# Visualization
SAFE_COLOR = (0, 255, 0)  # Green
VIOLATION_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White

# Video sources configuration
VIDEO_SOURCES = {
    "pedestrians": {
        "path": os.path.join(BASE_DIR, "input/pedestrians.mp4"),
        "calibration_points": None  # Will be loaded from file
    },
    "cctv": {
        "path": os.path.join(BASE_DIR, "input/cctv_footage.mp4"),
        "calibration_points": None
    }
}

# Calibration persistence
CALIBRATION_FILE = os.path.join(BASE_DIR, 'calibration.json')

def load_calibrations():
    """Load calibration data from JSON file"""
    if os.path.exists(CALIBRATION_FILE):
        with open(CALIBRATION_FILE) as f:
            try:
                calib_data = json.load(f)
                for source, points in calib_data.items():
                    if source in VIDEO_SOURCES:
                        VIDEO_SOURCES[source]["calibration_points"] = points
            except json.JSONDecodeError:
                pass

# Load calibrations on startup
load_calibrations()