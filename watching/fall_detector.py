# /home/ubuntu/persistent_watching_listening/watching/fall_detector.py

import cv2
import mediapipe as mp
import numpy as np
import math
from utils.dataset_utils import parse_yolo_label, FALL_DETECTION_CLASSES # Updated import

class FallDetector:
    """Detects falls based on human pose estimation from MediaPipe."""

    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        """Initializes MediaPipe Pose."""
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def _calculate_angle(self, a, b, c):
        """Calculates the angle between three points (e.g., elbow angle)."""
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def process_image(self, image_bgr):
        """Processes a single image to detect pose and determine fall status.

        Args:
            image_bgr: The input image in BGR format (from OpenCV).

        Returns:
            tuple: (bool, image_bgr)
                   - bool: True if a fall is detected, False otherwise.
                   - image_bgr: The image with pose landmarks drawn (if detected).
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False # Performance optimization
        results = self.pose.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        is_fall = False
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image_bgr.shape

            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image_bgr, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # --- Simple Static Fall Detection Logic (Example) ---
            # This is a very basic example based on body orientation.
            # A more robust method would analyze pose angles, aspect ratio, etc.
            try:
                # Get coordinates for shoulders and hips
                left_shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
                right_shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
                left_hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
                right_hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]

                # Calculate midpoint of shoulders and hips
                shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
                hip_mid_y = (left_hip[1] + right_hip[1]) / 2
                
                # Calculate midpoint of shoulders and hips x for angle calculation
                shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
                hip_mid_x = (left_hip[0] + right_hip[0]) / 2

                # Calculate the angle of the torso relative to the horizontal plane
                # Angle = 0 when horizontal, 90 when vertical upright
                dy = abs(shoulder_mid_y - hip_mid_y)
                dx = abs(shoulder_mid_x - hip_mid_x)
                
                if dx < 1e-6: # Avoid division by zero if torso is perfectly vertical
                    torso_angle = 90.0
                else:
                    torso_angle = math.degrees(math.atan(dy / dx))

                # Simple rule: If torso angle is less than 45 degrees, consider it a potential fall
                if torso_angle < 45:
                    is_fall = True
                    # print(f"Potential fall detected: Torso angle {torso_angle:.2f} degrees") # EN
                # else:
                    # print(f"Normal pose: Torso angle {torso_angle:.2f} degrees") # EN

            except Exception as e:
                print(f"Could not extract landmarks for fall detection logic: {e}") # EN
                is_fall = False # Default to not fallen if landmarks are missing
        else:
            # print("No pose landmarks detected in the image.") # EN
            is_fall = False # Cannot determine fall if no pose is detected

        # Add text overlay
        cv2.putText(image_bgr, f"Fall Detected: {is_fall}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255) if is_fall else (0, 255, 0), 2, cv2.LINE_AA)

        return is_fall, image_bgr

    def close(self):
        """Releases MediaPipe Pose resources."""
        self.pose.close()

# --- Example Usage (Updated for Kaggle Dataset) ---
if __name__ == '__main__':
    from ..utils.dataset_utils import load_kaggle_fall_detection_data, visualize_fall_detection_result
    from pathlib import Path

    # Define path to downloaded dataset
    fall_dataset_path = "/home/ubuntu/.cache/kagglehub/datasets/uttejkumarkandagatla/fall-detection-dataset/versions/1"
    output_dir = Path("/home/ubuntu/persistent_watching_listening/output/fall_detection_examples")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("--- Testing FallDetector with Kaggle Dataset (Validation Split) ---") # EN
    
    # Load validation dat    fall_val_data = load_kaggle_fall_detection_data(fall_dataset_path, split='val')

    if not fall_val_data:
        print("No validation data found. Exiting example.") # EN
    else:
        detector = FallDetector()
        max_examples = 5 # Process only a few examples
        processed_count = 0

        results_summary = []

        for i, item in enumerate(fall_val_data):
            if processed_count >= max_examples:
                break
            
            print(f"\nProcessing item {i+1}/{len(fall_val_data)}: {Path(item['image_path']).name}") # EN
            image = cv2.imread(item["image_path"])
            
            if image is None:
                print(f"  Warning: Could not read image {item['image_path']}") # EN
                continue

            # Get ground truth
            h, w, _ = image.shape
            ground_truth_labels = parse_yolo_label(item["label_path"], w, h)
            ground_truth_fall = any(label["class_id"] == 0 for label in ground_truth_labels) # 0 is 'Fall Detected'
            print(f"  Ground Truth: Fall Detected = {ground_truth_fall}") # EN

            # Perform detection
            is_fall_detected, annotated_image = detector.process_image(image.copy()) # Use copy to avoid modifying original
            print(f"  Prediction: Fall Detected = {is_fall_detected}") # EN

            # Save annotated image
            output_filename = output_dir / f"example_{i}_{Path(item['image_path']).stem}.jpg"
            cv2.imwrite(str(output_filename), annotated_image)
            print(f"  Annotated image saved to: {output_filename}") # EN

            results_summary.append({
                "filename": Path(item['image_path']).name,
                "ground_truth": ground_truth_fall,
                "prediction": is_fall_detected
            })
            
            processed_count += 1

        detector.close()
        
        print("\n--- Example Results Summary ---") # EN
        correct_predictions = 0
        for res in results_summary:
            print(f"  {res['filename']}: GT={res['ground_truth']}, Pred={res['prediction']}") # EN
            if res['ground_truth'] == res['prediction']:
                correct_predictions += 1
        
        if results_summary:
            accuracy = correct_predictions / len(results_summary)
            print(f"\nExample Accuracy ({len(results_summary)} images): {accuracy:.2f}") # EN
        else:
            print("No images processed.") # EN

    print("\nFall detector example finished.") # EN

