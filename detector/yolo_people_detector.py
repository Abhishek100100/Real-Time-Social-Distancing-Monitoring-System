import cv2
import numpy as np
from config.settings import (
    MIN_CONFIDENCE, NMS_THRESHOLD, 
    SAFE_COLOR, VIOLATION_COLOR
)

class YoloPeopleDetector:
    def __init__(self, model_path):
        self.labels = open(f"{model_path}/coco.names").read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(
            f"{model_path}/yolov3.cfg",
            f"{model_path}/yolov3.weights"
        )
        self.layer_names = self._get_output_layers()
        self.person_idx = self.labels.index("person")

    def _get_output_layers(self):
        ln = self.net.getLayerNames()
        try:
            return [ln[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            return [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, frame):
        (H, W) = frame.shape[:2]
        
        # Create blob and perform forward pass
        blob = cv2.dnn.blobFromImage(
            frame, 1/255.0, (416, 416), 
            swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.layer_names)

        boxes = []
        centroids = []
        confidences = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if class_id == self.person_idx and confidence > MIN_CONFIDENCE:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))

        # Apply NMS
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                results.append((
                    confidences[i],
                    (x, y, x + w, y + h),
                    centroids[i]
                ))

        return results