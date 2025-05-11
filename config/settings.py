# YOLO model configuration
MODEL_PATH = "models/yolo-coco"
YOLO_WEIGHTS = "yolov3.weights"
YOLO_CONFIG = "yolov3.cfg"
COCO_NAMES = "coco.names"

# Detection parameters
MIN_CONFIDENCE = 0.3
NMS_THRESHOLD = 0.3

# Social distance parameters
MIN_DISTANCE_PIXELS = 50  # Minimum safe distance in pixels

# Visualization settings
SAFE_COLOR = (0, 255, 0)  # Green
VIOLATION_COLOR = (0, 0, 255)  # Red
TEXT_COLOR = (255, 255, 255)  # White