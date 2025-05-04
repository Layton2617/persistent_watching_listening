# /home/ubuntu/persistent_watching_listening/listening/keyword_detector.py

import speech_recognition as sr
import os
from pathlib import Path

class KeywordDetector:
    """Detects specific keywords in spoken audio using SpeechRecognition."""

    def __init__(self, keywords=None):
        """Initializes the recognizer and sets target keywords."""
        self.recognizer = sr.Recognizer()
        if keywords is None:
            # Default keywords, including those relevant to the Kaggle dataset sentences
            self.keywords = ["help", "ambulance", "dead", "shot", "fire"] # EN
        else:
            self.keywords = [kw.lower() for kw in keywords]
        print(f"KeywordDetector initialized. Target keywords: {self.keywords}") # EN

    def process_audio_file(self, audio_path):
        """Processes an audio file to detect keywords.

        Args:
            audio_path (str): Path to the audio file (e.g., .wav).

        Returns:
            list: A list of detected keywords found in the audio, or None if recognition fails.
        """
        if not Path(audio_path).is_file():
            print(f"Error: Audio file not found at {audio_path}") # EN
            return None

        detected_keywords = []
        try:
            with sr.AudioFile(audio_path) as source:
                # print(f"  Adjusting for ambient noise from file...") # EN
                # self.recognizer.adjust_for_ambient_noise(source) # Less critical for file processing
                print(f"  Reading audio file: {Path(audio_path).name}") # EN
                audio_data = self.recognizer.record(source)

            # Recognize speech using Google Web Speech API (requires internet)
            # Other engines like Sphinx (offline) can be used but require setup
            print("  Recognizing speech...") # EN
            text = self.recognizer.recognize_google(audio_data)
            print(f"  Recognized text: 	{text}	") # EN
            text_lower = text.lower()
            for keyword in self.keywords:
                if keyword in text_lower:
                    detected_keywords.append(keyword)
            
            if not detected_keywords:
                 print("  No target keywords detected.") # EN
            else:
                 print(f"  Detected keywords: {detected_keywords}") # EN
            return detected_keywords

        except sr.UnknownValueError:
            print("  Speech Recognition could not understand audio") # EN
            return None # Return None to indicate recognition failure
        except sr.RequestError as e:
            print(f"  Could not request results from Speech Recognition service; {e}") # EN
            return None # Return None to indicate service error
        except Exception as e:
            print(f"  An unexpected error occurred during audio processing: {e}") # EN
            return None

# --- Example Usage (Updated for Kaggle Dataset) ---
if __name__ == '__main__':
    from utils.dataset_utils import load_kaggle_speech_emotion_data

    # Define path to downloaded dataset
    speech_dataset_path = "/home/ubuntu/.cache/kagglehub/datasets/anuvagoyal/speech-emotion-recognition-for-emergency-calls/versions/1"
    
    print("--- Testing KeywordDetector with Kaggle Dataset ---") # EN

    # Load audio data
    speech_data = load_kaggle_speech_emotion_data(speech_dataset_path)

    if not speech_data:
        print("No audio data found. Exiting example.") # EN
    else:
        detector = KeywordDetector() # Use default keywords
        max_examples = 5 # Process only a few examples
        processed_count = 0
        results_summary = []

        for i, item in enumerate(speech_data):
            if processed_count >= max_examples:
                break
            
            print(f"\nProcessing item {i+1}/{len(speech_data)}: {Path(item['audio_path']).name}") # EN
            print(f"  Ground Truth Sentence: 	{item['sentence_text']}	") # EN
            print(f"  Ground Truth Keywords: {item['ground_truth_keywords']}") # EN

            # Perform keyword detection
            detected_keywords = detector.process_audio_file(item["audio_path"])
            
            # Handle cases where recognition failed (detected_keywords is None)
            if detected_keywords is None:
                print("  Prediction: Recognition failed") # EN
                predicted_keywords_set = set()
                recognition_successful = False
            else:
                print(f"  Prediction: Detected Keywords = {detected_keywords}") # EN
                predicted_keywords_set = set(detected_keywords)
                recognition_successful = True

            ground_truth_keywords_set = set(item['ground_truth_keywords'])
            
            # Simple comparison (exact match of sets)
            match = (predicted_keywords_set == ground_truth_keywords_set) if recognition_successful else False

            results_summary.append({
                "filename": Path(item["audio_path"]).name,
                "ground_truth_sentence": item['sentence_text'],
                "ground_truth_keywords": ground_truth_keywords_set,
                "predicted_keywords": predicted_keywords_set if recognition_successful else "Recognition Failed", # EN
                "match": match,
                "recognition_successful": recognition_successful
            })
            
            processed_count += 1

        print("\n--- Example Results Summary ---") # EN
        correct_matches = 0
        recognition_failures = 0
        
        for res in results_summary:
            print(f"  File: {res['filename']}") # EN
            print(f"    GT Sentence: {res['ground_truth_sentence']}") # EN
            print(f"    GT Keywords: {res['ground_truth_keywords']}") # EN
            print(f"    Predicted Keywords: {res['predicted_keywords']}") # EN
            print(f"    Match: {res['match']}") # EN
            if not res['recognition_successful']:
                recognition_failures += 1
            elif res['match']:
                correct_matches += 1
        
        num_successful_recognitions = len(results_summary) - recognition_failures
        if num_successful_recognitions > 0:
            accuracy = correct_matches / num_successful_recognitions
            print(f"\nExample Accuracy (on {num_successful_recognitions} successful recognitions): {accuracy:.2f}") # EN
        elif len(results_summary) > 0:
             print("\nNo successful recognitions in the processed examples.") # EN
        else:
            print("No audio files processed.") # EN

    print("\nKeyword detector example finished.") # EN

