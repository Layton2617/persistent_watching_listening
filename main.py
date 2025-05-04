# /home/ubuntu/persistent_watching_listening/main.py

import argparse
import os
from pathlib import Path
import cv2
import pandas as pd
import time

# Import modules from the project
from watching import FallDetector
from listening import KeywordDetector
from decision_logic import SimpleDecisionLogic
from utils import (
    load_kaggle_fall_detection_data,
    parse_yolo_label,
    load_kaggle_speech_emotion_data,
    visualize_fall_detection_result,
    calculate_binary_metrics,
    FALL_DETECTION_CLASSES,
    SPEECH_KEYWORDS
)

# Default dataset paths (adjust if kagglehub downloads elsewhere or if manually placed)
DEFAULT_FALL_DATASET_PATH = "/home/ubuntu/.cache/kagglehub/datasets/uttejkumarkandagatla/fall-detection-dataset/versions/1"
DEFAULT_SPEECH_DATASET_PATH = "/home/ubuntu/.cache/kagglehub/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls/versions/1"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/persistent_watching_listening/output"

def evaluate_fall_detection(fall_data, detector, output_dir, visualize=False, max_samples=None):
    """Evaluates the FallDetector on the provided dataset split."""
    print("\n--- Evaluating Fall Detection ---") # EN
    tp, fp, tn, fn = 0, 0, 0, 0
    results_list = []
    start_time = time.time()
    processed_count = 0

    for i, item in enumerate(fall_data):
        if max_samples is not None and processed_count >= max_samples:
            print(f"Reached max samples limit ({max_samples}). Stopping fall detection evaluation.") # EN
            break
        
        image_path = Path(item["image_path"])
        label_path = Path(item["label_path"])
        print(f"Processing image {i+1}/{len(fall_data)}: {image_path.name}") # EN

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not read image {image_path}") # EN
            continue

        # Get ground truth
        h, w, _ = image.shape
        ground_truth_labels = parse_yolo_label(label_path, w, h)
        # Ground truth is True if *any* detected object is 'Fall Detected' (class_id 0)
        ground_truth_fall = any(label["class_id"] == 0 for label in ground_truth_labels)
        print(f"  Ground Truth: Fall Detected = {ground_truth_fall}") # EN

        # Perform detection
        is_fall_predicted, annotated_image = detector.process_image(image.copy())
        print(f"  Prediction: Fall Detected = {is_fall_predicted}") # EN

        # Update confusion matrix
        if is_fall_predicted and ground_truth_fall:
            tp += 1
        elif is_fall_predicted and not ground_truth_fall:
            fp += 1
        elif not is_fall_predicted and not ground_truth_fall:
            tn += 1
        elif not is_fall_predicted and ground_truth_fall:
            fn += 1

        # Save visualization if requested
        if visualize:
            viz_filename = output_dir / "fall_viz" / f"eval_{i}_{image_path.stem}.png"
            viz_filename.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(viz_filename), annotated_image)
            # print(f"  Visualization saved to: {viz_filename}") # EN

        results_list.append({
            "filename": image_path.name,
            "ground_truth": ground_truth_fall,
            "prediction": is_fall_predicted
        })
        processed_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"--- Fall Detection Evaluation Finished ({processed_count} samples in {elapsed_time:.2f}s) ---") # EN

    # Calculate metrics
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Fall Detection Metrics:") # EN
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}") # EN
    if metrics:
        # Corrected lines using string keys
        print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}") # EN
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}") # EN
        print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}") # EN
        print(f"  F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}") # EN
    else:
        print("  Could not calculate metrics (no samples processed?).") # EN

    return metrics, results_list

def evaluate_keyword_detection(speech_data, detector, max_samples=None):
    """Evaluates the KeywordDetector on the provided dataset."""
    print("\n--- Evaluating Keyword Detection ---") # EN
    # We evaluate based on whether *any* correct keyword was found if expected,
    # and *no* keyword was found if none expected.
    # This is a simplification. A more rigorous eval would use word error rate or similar.
    tp, fp, tn, fn = 0, 0, 0, 0 # TP: Expected keyword & found; FP: No keyword expected & found; TN: No keyword expected & none found; FN: Expected keyword & none found/failed
    results_list = []
    start_time = time.time()
    processed_count = 0
    recognition_failures = 0

    for i, item in enumerate(speech_data):
        if max_samples is not None and processed_count >= max_samples:
            print(f"Reached max samples limit ({max_samples}). Stopping keyword detection evaluation.") # EN
            break
        
        audio_path = Path(item["audio_path"])
        print(f"Processing audio {i+1}/{len(speech_data)}: {audio_path.name}") # EN

        # Get ground truth
        ground_truth_keywords_present = bool(item["ground_truth_keywords"]) # True if list is not empty
        print(f"  Ground Truth: Keywords Expected = {ground_truth_keywords_present} ({item['ground_truth_keywords']})") # EN

        # Perform detection
        detected_keywords_list = detector.process_audio_file(str(audio_path))

        if detected_keywords_list is None: # Recognition failed
            print("  Prediction: Recognition Failed") # EN
            recognition_failures += 1
            # If keywords were expected, count as FN. If not, count as TN (as no keyword was falsely predicted).
            if ground_truth_keywords_present:
                fn += 1
            else:
                tn += 1 
            prediction_keywords_found = False
            prediction_details = "Recognition Failed" # EN
        else:
            prediction_keywords_found = bool(detected_keywords_list)
            print(f"  Prediction: Keywords Found = {prediction_keywords_found} ({detected_keywords_list})") # EN
            prediction_details = str(detected_keywords_list)
            # Update confusion matrix based on presence/absence
            if prediction_keywords_found and ground_truth_keywords_present:
                tp += 1 # Correctly found keywords when expected
            elif prediction_keywords_found and not ground_truth_keywords_present:
                fp += 1 # Falsely found keywords when none expected
            elif not prediction_keywords_found and not ground_truth_keywords_present:
                tn += 1 # Correctly found no keywords when none expected
            elif not prediction_keywords_found and ground_truth_keywords_present:
                fn += 1 # Failed to find keywords when expected

        results_list.append({
            "filename": audio_path.name,
            "ground_truth_expected": ground_truth_keywords_present,
            "ground_truth_keywords": item["ground_truth_keywords"],
            "prediction_found": prediction_keywords_found,
            "prediction_details": prediction_details
        })
        processed_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"--- Keyword Detection Evaluation Finished ({processed_count} samples in {elapsed_time:.2f}s) ---") # EN
    print(f"  Recognition Failures: {recognition_failures}") # EN

    # Calculate metrics (based on presence/absence detection)
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Keyword Detection Metrics (Presence/Absence):") # EN
    print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}") # EN
    if metrics:
        # Corrected lines using string keys
        print(f"  Accuracy:  {metrics.get('accuracy', 'N/A'):.4f}") # EN
        print(f"  Precision: {metrics.get('precision', 'N/A'):.4f}") # EN
        print(f"  Recall:    {metrics.get('recall', 'N/A'):.4f}") # EN
        print(f"  F1 Score:  {metrics.get('f1_score', 'N/A'):.4f}") # EN
    else:
        print("  Could not calculate metrics (no samples processed?).") # EN

    return metrics, results_list

def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Data ---
    print("Loading datasets...") # EN
    fall_data = load_kaggle_fall_detection_data(args.fall_dataset_path, split=args.split)
    speech_data = load_kaggle_speech_emotion_data(args.speech_dataset_path)

    if not fall_data:
        print(f"Warning: No fall detection data loaded for split '{args.split}'. Skipping fall evaluation.") # EN
    if not speech_data:
        print("Warning: No speech emotion data loaded. Skipping keyword evaluation.") # EN

    # --- Initialize Models ---
    print("\nInitializing models...") # EN
    fall_detector = FallDetector()
    keyword_detector = KeywordDetector() # Uses default keywords ["help", "ambulance", "dead", "shot", "fire"]
    # decision_maker = SimpleDecisionLogic() # Not used in evaluation loop directly

    # --- Run Evaluations ---
    fall_metrics, fall_results = None, []
    if fall_data:
        fall_metrics, fall_results = evaluate_fall_detection(
            fall_data,
            fall_detector,
            output_dir,
            visualize=args.visualize,
            max_samples=args.max_samples
        )

    keyword_metrics, keyword_results = None, []
    if speech_data:
        keyword_metrics, keyword_results = evaluate_keyword_detection(
            speech_data,
            keyword_detector,
            max_samples=args.max_samples
        )

    # --- Save Detailed Results --- 
    print("\nSaving detailed results to CSV...") # EN
    if fall_results:
        fall_df = pd.DataFrame(fall_results)
        fall_csv_path = output_dir / f"fall_detection_results_{args.split}.csv"
        fall_df.to_csv(fall_csv_path, index=False)
        print(f"Fall detection results saved to: {fall_csv_path}") # EN
    
    if keyword_results:
        keyword_df = pd.DataFrame(keyword_results)
        keyword_csv_path = output_dir / "keyword_detection_results.csv"
        keyword_df.to_csv(keyword_csv_path, index=False)
        print(f"Keyword detection results saved to: {keyword_csv_path}") # EN

    # --- Cleanup ---
    print("\nClosing models...") # EN
    fall_detector.close()

    print("\nEvaluation complete.") # EN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Persistent Watching & Listening components on Kaggle datasets.") # EN
    parser.add_argument("--fall_dataset_path", type=str, default=DEFAULT_FALL_DATASET_PATH, 
                        help="Path to the root directory of the extracted Kaggle Fall Detection dataset.") # EN
    parser.add_argument("--speech_dataset_path", type=str, default=DEFAULT_SPEECH_DATASET_PATH, 
                        help="Path to the root directory of the extracted Kaggle Speech Emotion dataset.") # EN
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, 
                        help="Directory to save evaluation results (CSV files, visualizations).") # EN
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], 
                        help="Which split of the fall detection dataset to evaluate.") # EN
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to process for each evaluation (for quick testing).") # EN
    parser.add_argument("--visualize", action="store_true", 
                        help="Save annotated images for fall detection evaluation.") # EN

    args = parser.parse_args()
    main(args)

