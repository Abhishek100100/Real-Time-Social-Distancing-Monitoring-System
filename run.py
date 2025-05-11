import argparse
import cv2
import os
from config.settings import (
    MODEL_PATH,
    VIDEO_SOURCES,
    MIN_SAFE_DISTANCE,
    REFERENCE_OBJECT_HEIGHT,
    SAFE_COLOR,
    VIOLATION_COLOR
)
from detector.yolo_people_detector import YoloPeopleDetector
from utils.distance_calculator import DistanceCalculator
from utils.visualization import Visualizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True,
                       help="Name of video source to use")
    parser.add_argument("-o", "--output", default="",
                       help="Output video path")
    parser.add_argument("-d", "--display", type=int, default=1,
                       help="Display output (1) or not (0)")
    args = parser.parse_args()

    # Verify video source
    if args.source not in VIDEO_SOURCES:
        print(f"Error: Unknown video source '{args.source}'")
        print("Available sources:", list(VIDEO_SOURCES.keys()))
        return

    video_info = VIDEO_SOURCES[args.source]
    if not video_info.get("calibration_points"):
        print(f"Error: Source '{args.source}' not calibrated")
        print(f"Please run: python calibrate.py {args.source}")
        return

    # Initialize components
    detector = YoloPeopleDetector(MODEL_PATH)
    distance_calc = DistanceCalculator(MIN_SAFE_DISTANCE, REFERENCE_OBJECT_HEIGHT)
    distance_calc.set_homography(video_info["calibration_points"])
    visualizer = Visualizer()

    # Initialize video
    cap = cv2.VideoCapture(video_info["path"])
    if not cap.isOpened():
        print(f"Error: Could not open video {video_info['path']}")
        return

    writer = None
    frame_width, frame_height = 700, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize while maintaining aspect ratio
        frame_height = int(frame.shape[0] * 700 / frame.shape[1])
        frame = cv2.resize(frame, (700, frame_height))
        
        # Detect people
        results = detector.detect(frame)
        
        if results:
            centroids = [r[2] for r in results]
            violations = distance_calc.find_violations(centroids)
            
            # Draw results
            visualizer.draw_detections(frame, results, violations)
            visualizer.draw_violation_info(frame, len(violations), len(results), True)
            visualizer.draw_timestamp(frame)

        # Display/output
        if args.display:
            cv2.imshow("Social Distance Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if args.output:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                writer = cv2.VideoWriter(
                    args.output, fourcc, 25,
                    (700, frame_height), True
                )
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()