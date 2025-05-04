# /home/ubuntu/persistent_watching_listening_en/advanced_models/object_detection/yolo_detector.py

import torch
import cv2
import numpy as np
from pathlib import Path

class YoloDetector:
    """Performs object detection using a pre-trained YOLOv5 model."""

    def __init__(self, model_name='yolov5s', confidence_threshold=0.25, iou_threshold=0.45):
        """Loads the YOLOv5 model from PyTorch Hub.

        Args:
            model_name (str): The name of the YOLOv5 model to load (e.g., 'yolov5s', 'yolov5m').
            confidence_threshold (float): Minimum confidence score for detections.
            iou_threshold (float): Intersection over Union threshold for Non-Maximum Suppression.
        """
        try:
            # Load the model from PyTorch Hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.conf = confidence_threshold
            self.model.iou = iou_threshold
            print(f"YOLOv5 model '{model_name}' loaded successfully.")
        except Exception as e:
            print(f"Error loading YOLOv5 model: {e}")
            print("Please ensure you have an internet connection and the ultralytics/yolov5 repository is accessible.")
            self.model = None

    def detect_objects(self, image_bgr):
        """Detects objects in a single image.

        Args:
            image_bgr: The input image in BGR format (from OpenCV).

        Returns:
            tuple: (results_df, annotated_image)
                   - results_df (pandas.DataFrame): DataFrame containing detection results 
                     (xmin, ymin, xmax, ymax, confidence, class, name).
                   - annotated_image (numpy.ndarray): The image with bounding boxes drawn.
        """
        if self.model is None:
            print("YOLOv5 model not loaded. Cannot perform detection.")
            return None, image_bgr

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Perform inference
        results = self.model(image_rgb)

        # Get results as pandas DataFrame
        results_df = results.pandas().xyxy[0] # Detections for image 0

        # Draw bounding boxes on the original BGR image
        annotated_image = results.render()[0] # Renders detections on the image
        annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        return results_df, annotated_image_bgr

# --- Example Usage --- 
if __name__ == '__main__':
    # Create a dummy image for testing
    dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(dummy_image, (100, 100), (200, 200), (0, 255, 0), 3) # Draw a green box
    cv2.circle(dummy_image, (400, 300), 50, (0, 0, 255), -1) # Draw a red circle

    output_dir = Path("/home/ubuntu/persistent_watching_listening_en/output/yolo_detection_examples")
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "dummy_input.png"), dummy_image)

    print("--- Testing YoloDetector --- ")
    detector = YoloDetector(model_name='yolov5s') # Use a small model for quick testing

    if detector.model:
        print("Processing dummy image...")
        detections_df, annotated_img = detector.detect_objects(dummy_image)

        if detections_df is not None:
            print("\nDetection Results (DataFrame):")
            print(detections_df)

            output_filename = output_dir / "dummy_annotated.png"
            cv2.imwrite(str(output_filename), annotated_img)
            print(f"\nAnnotated image saved to: {output_filename}")
        else:
            print("Detection failed.")
    else:
        print("Skipping detection due to model loading failure.")

    print("\nYOLO detector example finished.")

