# Persistent Watching & Listening (Dataset Evaluation Version)

## 1. Project Overview

This project provides an AI-based status monitoring framework, enhanced to process and evaluate performance on public datasets. It analyzes poses in images to detect potential falls and analyzes corresponding audio files to identify help keywords, finally determining the urgency level and evaluating the performance of each component.

This version uses:
*   **Kaggle Fall Detection Dataset:** (`uttejkumarkandagatla/fall-detection-dataset`) for evaluating the visual fall detection component.
*   **Kaggle Speech Emotion Recognition for Emergency Calls Dataset:** (`anuvagoyal/speech-emotion-recognition-for-emergency-calls`) for evaluating the auditory keyword detection component.

Core Features:
*   **Watching Module:** Uses MediaPipe for pose estimation on images to detect fall poses (`watching/fall_detector.py`).
*   **Listening Module:** Uses SpeechRecognition to process audio files, perform speech-to-text, and identify preset keywords (`listening/keyword_detector.py`).
*   **Decision Logic Module:** Combines detection results (placeholder, not the focus of evaluation) (`decision_logic/simple_logic.py`).
*   **Dataset Utilities:** Functions for loading Kaggle datasets, parsing labels, and visualizing results (`utils/dataset_utils.py`).
*   **Evaluation Utilities:** Functions for calculating standard classification metrics (Accuracy, Precision, Recall, F1-Score) (`utils/evaluation_utils.py`).
*   **Main Program:** Takes dataset paths, runs evaluations on specified splits, calculates metrics, and saves detailed results (`main.py`).
*   **Report Generation:** Script to generate confusion matrix plots and a summary report from evaluation results (`generate_report.py`).

**Please Note:** This is a proof-of-concept demonstrating dataset integration and evaluation. The models used (MediaPipe pose for static images, Google Speech Recognition) are basic examples. Their performance on these datasets, as shown in the evaluation, reflects their inherent limitations. More sophisticated models and techniques are required for real-world application.

## 2. Project Structure

```
persistent_watching_listening/
├── watching/                 # Visual processing module
│   ├── __init__.py
│   └── fall_detector.py      # Fall detector (processes images)
├── listening/                # Auditory processing module
│   ├── __init__.py
│   └── keyword_detector.py   # Keyword detector (processes audio files)
├── decision_logic/           # Decision logic module
│   ├── __init__.py
│   └── simple_logic.py       # Simple decision logic
├── utils/                    # Utility classes/functions
│   ├── __init__.py
│   ├── dataset_utils.py      # Kaggle dataset loading/parsing utilities
│   └── evaluation_utils.py   # Metrics calculation utilities
├── output/                   # Default directory for results
│   ├── fall_detection_results_val.csv
│   ├── keyword_detection_results.csv
│   ├── fall_detection_confusion_matrix.png
│   ├── keyword_detection_confusion_matrix.png
│   ├── evaluation_report.md
│   └── fall_viz/             # Optional fall detection visualizations
│       └── ...
├── main.py                   # Main program entry point (runs evaluation)
├── generate_report.py        # Script to generate report from CSV results
├── requirements.txt          # Python dependency list (updated)
└── README.md                 # This document
```

## 3. Installation and Setup

**Environment Requirements:**
*   Python 3.8 or higher
*   macOS / Linux / Windows (Tested in sandbox Linux environment)
*   Internet connection (required for `kagglehub` download and Google Speech Recognition API)

**Installation Steps:**

1.  **Clone or Download Project:**
    Extract or clone the project files to your local machine.

2.  **Create Virtual Environment (Recommended):**
    Open a terminal, navigate to the project root directory, and execute:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate  # Windows
    ```

3.  **Install Dependencies:**
    In the activated virtual environment, run:
    ```bash
    pip install -r requirements.txt
    ```
    *Note:* This installs `opencv-python`, `mediapipe`, `numpy`, `SpeechRecognition`, `pandas`, `kagglehub`, `matplotlib`, `seaborn`, and `scikit-learn`.

4.  **Download Datasets (Automatic via `kagglehub`):**
    The first time you run `main.py` or `utils/dataset_utils.py`, the `kagglehub` library will attempt to download the required datasets automatically. You might need to authenticate with Kaggle if you haven't used `kagglehub` before (follow its prompts or configure `~/.kaggle/kaggle.json`). The datasets will typically be downloaded to `~/.cache/kagglehub/datasets/`.

## 4. Running the Evaluation

In the project root directory, with the virtual environment activated, run the main evaluation program using the following command:

```bash
python main.py [--fall_dataset_path <path>] [--speech_dataset_path <path>] [--output_dir <path>] [--split <train|val>] [--max_samples <N>] [--visualize]
```

**Command Line Arguments:**
*   `--fall_dataset_path` (Optional): Path to the root of the Kaggle Fall Detection dataset. Defaults to the typical `kagglehub` cache location.
*   `--speech_dataset_path` (Optional): Path to the root of the Kaggle Speech Emotion dataset. Defaults to the typical `kagglehub` cache location.
*   `--output_dir` (Optional): Directory to save evaluation results (CSV files, visualizations). Defaults to `./output`.
*   `--split` (Optional): Which split of the fall detection dataset to evaluate (`train` or `val`). Defaults to `val`.
*   `--max_samples` (Optional): Maximum number of samples to process for *each* evaluation (fall and keyword). Useful for quick testing. Processes all samples if omitted.
*   `--visualize` (Optional): If included, saves annotated images with pose and detection results for the fall detection evaluation to `<output_dir>/fall_viz/`.

**Example (Evaluate 50 samples from validation split, save visualizations):**
```bash
python main.py --split val --max_samples 50 --visualize
```

The program will:
1.  Load the specified split of the fall detection dataset and the full speech emotion dataset.
2.  Initialize the `FallDetector` and `KeywordDetector`.
3.  Iterate through the fall detection data (up to `max_samples`):
    *   Process each image.
    *   Compare prediction to ground truth label.
    *   Optionally save visualization.
4.  Calculate and print fall detection metrics (Accuracy, Precision, Recall, F1).
5.  Iterate through the speech emotion data (up to `max_samples`):
    *   Process each audio file.
    *   Compare predicted keyword presence to ground truth.
6.  Calculate and print keyword detection metrics (based on presence/absence).
7.  Save detailed results for both evaluations to CSV files in the output directory.

## 5. Generating the Report

After running the evaluation (`main.py`), you can generate a summary report with confusion matrix plots:

```bash
python generate_report.py
```

This script reads the CSV files generated by `main.py` from the `./output` directory (or the directory specified via `--output_dir` if you used that) and creates:
*   `evaluation_report.md`: A markdown file summarizing the results with embedded confusion matrices.
*   `fall_detection_confusion_matrix.png`: Confusion matrix plot for fall detection.
*   `keyword_detection_confusion_matrix.png`: Confusion matrix plot for keyword detection (presence/absence).

## 6. Evaluation Results (Example with `max_samples=10`)

*(Note: These are results from a small sample run and may vary. Run the evaluation on the full dataset for more reliable metrics.)*

**Fall Detection (Validation Split):**
*   TP: 7, FP: 0, TN: 0, FN: 3
*   Accuracy: 0.7000
*   Precision: 1.0000
*   Recall: 0.7000
*   F1 Score: 0.8235

**Keyword Detection (Presence/Absence):**
*   TP: 4, FP: 4, TN: 1, FN: 1 (Recognition Failures: 0)
*   Accuracy: 0.5000
*   Precision: 0.5000
*   Recall: 0.8000
*   F1 Score: 0.6154

See the generated `evaluation_report.md` and associated PNG images in the output directory for more details.

## 7. Limitations and Future Development

*   **Model Simplicity:** The current models (MediaPipe static pose, basic keyword spotting via ASR) are very basic and have limited accuracy, as reflected in the evaluation metrics. More advanced models (e.g., video-based action recognition, dedicated keyword spotting models) are needed for better performance.
*   **Keyword Detection Evaluation:** The current keyword evaluation only checks for the *presence* or *absence* of *any* target keyword, not the specific keyword accuracy or word error rate, which would be more informative.
*   **Dataset Limitations:** The speech dataset contains emotional speech but not necessarily specific distress calls like "help me I've fallen". The fall dataset contains images, limiting the evaluation of temporal fall detection methods.
*   **Real-world Applicability:** This setup is purely for demonstrating dataset processing and evaluation, not for deployment.

**Potential Future Extensions:**
*   Integrate video-based action recognition models for fall detection.
*   Use dedicated keyword spotting or more robust ASR models.
*   Implement more sophisticated evaluation metrics (e.g., Word Error Rate for ASR).
*   Explore other relevant datasets.
*   Combine visual and auditory information using multi-modal models.

## 8. Disclaimer

This project is intended for research and learning purposes only and cannot replace professional medical or caregiving equipment. The developer assumes no responsibility for any consequences arising from the use of this project.

