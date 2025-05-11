import cv2
import json
import os
from config.settings import (
    VIDEO_SOURCES,
    REFERENCE_OBJECT_HEIGHT,
    CALIBRATION_FILE,
    TEXT_COLOR
)

def show_instructions(frame):
    instructions = [
        "CALIBRATION INSTRUCTIONS:",
        "1. Click ground reference point (origin)",
        "2. Click 1m along ground plane (X-axis)",
        "3. Click 1m perpendicular (Y-axis)",
        "4. Click head height at origin (Z-axis)",
        "",
        "Press 'r' to reset points",
        "Press 's' to save calibration",
        "Press 'q' to quit"
    ]
    
    y_offset = 40
    for line in instructions:
        cv2.putText(frame, line, (20, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   TEXT_COLOR, 1)
        y_offset += 25

def calibrate_video_source(source_name):
    if source_name not in VIDEO_SOURCES:
        print(f"Error: Unknown video source '{source_name}'")
        return None
    
    video_path = VIDEO_SOURCES[source_name]["path"]
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        return None
    
    frame = cv2.resize(frame, (700, int(frame.shape[0] * 700/frame.shape[1])))
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            print(f"Added point {len(points)}: ({x}, {y})")
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(frame, points[-2], points[-1], (0,255,0), 2)
            for i, (px, py) in enumerate(points):
                cv2.putText(frame, str(i+1), (px+10, py+10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (255,255,255), 2)
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    while True:
        display_frame = frame.copy()
        show_instructions(display_frame)
        
        for i, (x, y) in enumerate(points):
            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display_frame, str(i+1), (x+10, y+10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (255,255,255), 2)
            if i > 0:
                cv2.line(display_frame, points[i-1], (x,y), (0,255,0), 2)
        
        cv2.imshow("Calibration", display_frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('r'):
            points = []
            print("Reset all calibration points")
        elif key == ord('s'):
            if len(points) == 4:
                break
            else:
                print("Need exactly 4 points for calibration")
        elif key == ord('q'):
            print("Calibration cancelled")
            cv2.destroyAllWindows()
            cap.release()
            return None
    
    cv2.destroyAllWindows()
    cap.release()
    return points

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Name of video source to calibrate")
    args = parser.parse_args()
    
    calib_points = calibrate_video_source(args.source)
    if calib_points:
        # Load existing calibrations
        calib_data = {}
        if os.path.exists(CALIBRATION_FILE):
            with open(CALIBRATION_FILE) as f:
                calib_data = json.load(f)
        
        # Update calibration
        calib_data[args.source] = calib_points
        
        # Save to file
        with open(CALIBRATION_FILE, 'w') as f:
            json.dump(calib_data, f, indent=4)
        
        print(f"Calibration saved for {args.source}")
        print("Calibration points:", calib_points)
    else:
        print("Calibration failed")