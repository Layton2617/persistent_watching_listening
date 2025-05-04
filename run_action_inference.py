# -*- coding: utf-8 -*-
"""Script to run action recognition inference on a sample video."""

import sys
import os

# Add the project root to the Python path
sys.path.append("/home/ubuntu/persistent_watching_listening_en")

from advanced_models.action_recognition.action_recognizer import ActionRecognizer

if __name__ == '__main__':
    sample_video_path = "/home/ubuntu/persistent_watching_listening_en/sample_videos/archery_official_sample.mp4"
    kinetics_labels_path = "/home/ubuntu/kinetics_classnames.json"

    if not os.path.exists(sample_video_path):
        print(f"Error: Sample video not found at {sample_video_path}")
        sys.exit(1)

    if not os.path.exists(kinetics_labels_path):
        print(f"Error: Kinetics labels file not found at {kinetics_labels_path}. Please ensure it is downloaded.")
        # Attempt to create dummy if absolutely necessary, but prefer real labels
        # print(f"Creating dummy {kinetics_labels_path}...")
        # dummy_labels = {f"Action_{i}": i for i in range(400)}
        # with open(kinetics_labels_path, 'w') as f:
        #     json.dump(dummy_labels, f)
        sys.exit(1)

    print("--- Running Action Recognition Inference --- ")
    try:
        # Initialize the recognizer (using slow_r50 for potentially faster inference on CPU)
        recognizer = ActionRecognizer(model_name="slow_r50")

        # Predict the action
        pred_label, pred_score = recognizer.predict(sample_video_path)

        if pred_label is not None:
            print(f"\nPrediction for '{os.path.basename(sample_video_path)}':")
            print(f"  Action: {pred_label}")
            print(f"  Confidence: {pred_score:.4f}")
        else:
            print(f"\nAction recognition failed for {sample_video_path}.")

    except Exception as e:
        print(f"\nError during action recognition inference: {e}")
        import traceback
        traceback.print_exc()

