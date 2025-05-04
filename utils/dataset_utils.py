# /home/ubuntu/persistent_watching_listening/utils/dataset_utils.py

import os
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

# --- Kaggle Fall Detection Dataset --- 

FALL_DETECTION_CLASSES = {0: "Fall Detected", 1: "Walking", 2: "Sitting"}

def parse_yolo_label(label_path, img_width, img_height):
    """Parses a YOLO format label file.

    Args:
        label_path (str or Path): Path to the label file (.txt).
        img_width (int): Width of the corresponding image.
        img_height (int): Height of the corresponding image.

    Returns:
        list: A list of dictionaries, each containing 'class_id', 'class_name', 
              'bbox' (list: [xmin, ymin, xmax, ymax] in absolute pixel coordinates).
              Returns an empty list if the file doesn't exist or is empty.
    """
    labels = []
    label_path = Path(label_path)
    if not label_path.is_file():
        return labels

    with open(label_path, 'r') as f:
        for line in f.readlines():
            try:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center_rel, y_center_rel, width_rel, height_rel = map(float, parts[1:])
                    
                    # Convert relative coordinates to absolute pixel coordinates
                    x_center_abs = x_center_rel * img_width
                    y_center_abs = y_center_rel * img_height
                    width_abs = width_rel * img_width
                    height_abs = height_rel * img_height
                    
                    xmin = int(x_center_abs - width_abs / 2)
                    ymin = int(y_center_abs - height_abs / 2)
                    xmax = int(x_center_abs + width_abs / 2)
                    ymax = int(y_center_abs + height_abs / 2)
                    
                    labels.append({
                        'class_id': class_id,
                        'class_name': FALL_DETECTION_CLASSES.get(class_id, "Unknown"),
                        'bbox': [xmin, ymin, xmax, ymax]
                    })
            except Exception as e:
                print(f"  Warning: Could not parse line 	{line.strip()}	 in 	{label_path.name}	: {e}") # EN
    return labels

def load_kaggle_fall_detection_data(dataset_base_path, split='train'):
    """Loads image paths and corresponding label paths from the Kaggle Fall Detection dataset.

    Args:
        dataset_base_path (str or Path): Path to the extracted dataset root 
                                         (e.g., '/home/ubuntu/.cache/kagglehub/datasets/.../1').
        split (str): Which split to load ('train' or 'val').

    Returns:
        list: A list of dictionaries, each containing 'image_path' and 'label_path'.
    """
    dataset_base_path = Path(dataset_base_path)
    image_dir = dataset_base_path / "fall_dataset" / "images" / split
    label_dir = dataset_base_path / "fall_dataset" / "labels" / split
    data_items = []

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"Error: Kaggle Fall Detection dataset directories not found for split 	{split}	 at 	{dataset_base_path}") # EN
        return data_items

    for img_path in image_dir.glob("*.jpg"): # Assuming jpg, adjust if other formats exist
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.is_file():
            data_items.append({
                'image_path': str(img_path),
                'label_path': str(label_path)
            })
        # else: # Optional warning
        #     print(f"Warning: No label file found for image {img_path.name}") # EN
            
    print(f"Found {len(data_items)} image/label pairs in split 	{split}	.") # EN
    return data_items

# --- Kaggle Speech Emotion Dataset --- 

SPEECH_EMOTION_CLASSES = {1: "Angry", 2: "Drunk", 3: "Painful", 4: "Stressful"}
SPEECH_SENTENCES = {
    1: "We need an ambulance as soon as possible.",
    2: "Someone has been lying dead on the street.",
    3: "A neighbor of mine is shot dead.",
    4: "This place is on fire. Please send help."
}
SPEECH_KEYWORDS = { # Define which sentences contain which keywords for evaluation
    1: ["ambulance"],
    2: [],
    3: [],
    4: ["help"]
}

def parse_speech_filename(filename):
    """Parses the filename from the Kaggle Speech Emotion dataset.
    Filename format: EmotionNumber_SentenceNumber_Gender_Synthetic/Natural_SpeakerNumber.wav
    """
    try:
        parts = filename.stem.split('_')
        if len(parts) == 5:
            emotion_id = int(parts[0])
            sentence_id = int(parts[1])
            # gender = int(parts[2]) # 01 Female, 02 Male
            # type = int(parts[3]) # 01 Natural, 02 Synthetic
            # speaker = int(parts[4])
            return {
                'emotion_id': emotion_id,
                'emotion_name': SPEECH_EMOTION_CLASSES.get(emotion_id, "Unknown"),
                'sentence_id': sentence_id,
                'sentence_text': SPEECH_SENTENCES.get(sentence_id, ""),
                'ground_truth_keywords': SPEECH_KEYWORDS.get(sentence_id, [])
            }
    except Exception as e:
        print(f"Warning: Could not parse filename 	{filename.name}	: {e}") # EN
    return None

def load_kaggle_speech_emotion_data(dataset_base_path):
    """Loads audio paths and metadata from the Kaggle Speech Emotion dataset.

    Args:
        dataset_base_path (str or Path): Path to the extracted dataset root 
                                         (e.g., '/home/ubuntu/.cache/kagglehub/datasets/.../1').

    Returns:
        list: A list of dictionaries, each containing 'audio_path' and metadata 
              parsed from the filename (emotion, sentence, keywords).
    """
    dataset_base_path = Path(dataset_base_path)
    audio_root_dir = dataset_base_path / "CUSTOM_DATASET"
    data_items = []

    if not audio_root_dir.is_dir():
        print(f"Error: Kaggle Speech Emotion dataset directory not found at 	{dataset_base_path}") # EN
        return data_items

    # Use glob to find all .wav files recursively within CUSTOM_DATASET
    audio_files = glob.glob(str(audio_root_dir / "**" / "*.wav"), recursive=True)

    for audio_path_str in audio_files:
        audio_path = Path(audio_path_str)
        metadata = parse_speech_filename(audio_path)
        if metadata:
            data_items.append({
                'audio_path': str(audio_path),
                **metadata # Unpack metadata dictionary
            })
            
    print(f"Found {len(data_items)} audio files.") # EN
    return data_items

# --- Visualization (Keep as is for now) --- 

def visualize_fall_detection_result(image_bgr, is_fall_detected, title=""):
    """Visualizes the image with fall detection results using Matplotlib."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img_rgb)
    plt.title(f"{title} - Fall Detected: {is_fall_detected}") # EN
    plt.axis("off")
    # plt.show() # Avoid showing plots in non-interactive environment
    # Instead, save the plot
    output_path = f"/home/ubuntu/persistent_watching_listening/output/viz_{Path(title).stem}.png"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close() # Close the figure to free memory
    print(f"  Visualization saved to: {output_path}") # EN

# --- Old find_data_pairs (Commented out or remove) --- 
# def find_data_pairs(image_dir, audio_dir, img_extensions=(".jpg", ".png", ".jpeg"), audio_extensions=(".wav", ".mp3")):
#     ...

# --- Example Usage (Updated for new functions) ---
if __name__ == '__main__':
    # Define paths to downloaded datasets (adjust if necessary)
    fall_dataset_path = "/home/ubuntu/.cache/kagglehub/datasets/uttejkumarkandagatla/fall-detection-dataset/versions/1"
    speech_dataset_path = "/home/ubuntu/.cache/kagglehub/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls/versions/1"

    print("--- Testing Kaggle Fall Detection Loader ---") # EN
    fall_train_data = load_kaggle_fall_detection_data(fall_dataset_path, split='train')
    if fall_train_data:
        print(f"Loaded {len(fall_train_data)} training items. First item:") # EN
        print(fall_train_data[0])
        # Try parsing the first label
        try:
            img = cv2.imread(fall_train_data[0]['image_path'])
            if img is not None:
                h, w, _ = img.shape
                labels = parse_yolo_label(fall_train_data[0]['label_path'], w, h)
                print("Parsed labels for first item:", labels) # EN
            else:
                print("Could not read first image to get dimensions.") # EN
        except Exception as e:
            print(f"Error parsing first label: {e}") # EN

    fall_val_data = load_kaggle_fall_detection_data(fall_dataset_path, split='val')
    if fall_val_data:
        print(f"\nLoaded {len(fall_val_data)} validation items.") # EN

    print("\n--- Testing Kaggle Speech Emotion Loader ---") # EN
    speech_data = load_kaggle_speech_emotion_data(speech_dataset_path)
    if speech_data:
        print(f"Loaded {len(speech_data)} audio items. First item:") # EN
        print(speech_data[0])
        print("\nLast item:") # EN
        print(speech_data[-1])

    print("\nDataset utility tests finished.") # EN

