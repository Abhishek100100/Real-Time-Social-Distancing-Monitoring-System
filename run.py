import argparse
import cv2
import imutils
import os
from config.settings import ( 
    MODEL_PATH,
    MIN_DISTANCE_PIXELS, 
    MIN_CONFIDENCE,
    NMS_THRESHOLD
)
from detector.yolo_people_detector import YoloPeopleDetector
from utils.distance_calculator import DistanceCalculator
from utils.visualization import Visualizer

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default="",
                      help="path to input video file")
    parser.add_argument("-o", "--output", type=str, default="",
                      help="path to output video file")
    parser.add_argument("-d", "--display", type=int, default=1,
                      help="whether to display output")
    args = vars(parser.parse_args())

    # Initialize components
    detector = YoloPeopleDetector(MODEL_PATH)
    distance_calculator = DistanceCalculator(MIN_DISTANCE_PIXELS)
    visualizer = Visualizer()

    # Initialize video stream
    vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
    writer = None

    while True:
        grabbed, frame = vs.read()
        if not grabbed:
            break

        # Resize frame and detect people
        frame = imutils.resize(frame, width=700)
        results = detector.detect(frame)
        centroids = [r[2] for r in results]
        
        # Find violations
        violations = distance_calculator.find_violations(centroids)

        # Visualize results
        visualizer.draw_detections(frame, results, violations)
        visualizer.draw_violation_info(frame, len(violations), len(results))
        visualizer.draw_timestamp(frame)

        # Display output
        if args["display"] > 0:
            cv2.imshow("Social Distance Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Write to output file
        if args["output"] and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args["output"], fourcc, 25,
                                    (frame.shape[1], frame.shape[0]), True)
        
        if writer is not None:
            writer.write(frame)

    # Cleanup
    vs.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()