# /home/ubuntu/persistent_watching_listening/utils/evaluation_utils.py

import numpy as np

def calculate_binary_metrics(tp, fp, tn, fn):
    """Calculates precision, recall, f1-score, and accuracy for binary classification.

    Args:
        tp (int): True Positives
        fp (int): False Positives
        tn (int): True Negatives
        fn (int): False Negatives

    Returns:
        dict: A dictionary containing accuracy, precision, recall, and f1_score.
              Returns None for metrics that cannot be calculated (e.g., division by zero).
    """
    metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None
    }

    total = tp + fp + tn + fn
    if total > 0:
        metrics["accuracy"] = (tp + tn) / total

    # Precision
    if (tp + fp) > 0:
        metrics["precision"] = tp / (tp + fp)

    # Recall (Sensitivity)
    if (tp + fn) > 0:
        metrics["recall"] = tp / (tp + fn)

    # F1 Score
    precision = metrics["precision"]
    recall = metrics["recall"]
    if precision is not None and recall is not None and (precision + recall) > 0:
        metrics["f1_score"] = 2 * (precision * recall) / (precision + recall)

    return metrics

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    tp, fp, tn, fn = 10, 2, 30, 3
    print(f"Testing with TP={tp}, FP={fp}, TN={tn}, FN={fn}") # EN
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Calculated Metrics:", metrics) # EN

    tp, fp, tn, fn = 0, 0, 10, 0
    print(f"\nTesting with TP={tp}, FP={fp}, TN={tn}, FN={fn}") # EN
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Calculated Metrics (all negative):") # EN
    print(metrics)

    tp, fp, tn, fn = 10, 0, 0, 0
    print(f"\nTesting with TP={tp}, FP={fp}, TN={tn}, FN={fn}") # EN
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Calculated Metrics (all positive):") # EN
    print(metrics)

    tp, fp, tn, fn = 0, 5, 10, 0
    print(f"\nTesting with TP={tp}, FP={fp}, TN={tn}, FN={fn}") # EN
    metrics = calculate_binary_metrics(tp, fp, tn, fn)
    print("Calculated Metrics (no true positives):") # EN
    print(metrics)

