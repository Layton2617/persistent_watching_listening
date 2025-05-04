# -*- coding: utf-8 -*-
"""
Action Recognition Module using PyTorchVideo (e.g., SlowFast or 3D ResNet).
"""

import torch
import json
from pytorchvideo.data.encoded_video import EncodedVideo

from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)

class ActionRecognizer:
    def __init__(self, model_name="slowfast_r50", device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initializes the Action Recognizer.

        Args:
            model_name (str): Name of the model to load (e.g., "slowfast_r50", "slow_r50").
            device (str): Device to run the model on ("cuda" or "cpu").
        """
        self.device = device
        self.model_name = model_name # Store model name as an attribute
        print(f"Loading action recognition model ({model_name}) onto {self.device}...")

        try:
            # Load the model
            self.model = torch.hub.load("facebookresearch/pytorchvideo", model=model_name, pretrained=True)
            self.model = self.model.eval()
            self.model = self.model.to(self.device)

            # Load the class labels (Kinetics 400)
            # You might need to download this file or adjust the path
            # Example source: https://dl.fbaipublicfiles.com/pytorchvideo/data/class_names/kinetics_classnames.json
            # For simplicity, let's assume it's downloaded or define a placeholder
            try:
                # Attempt to load from a standard location if available
                # In a real scenario, ensure this file exists
                with open("/home/ubuntu/kinetics_classnames.json", "r") as f:
                    # This dictionary maps ID (string) to Class Name (string)
                    self.id_to_classname = json.load(f)
            except FileNotFoundError:
                print("Warning: Kinetics class names file not found. Using placeholder indices.")
                # Create dummy labels - replace with actual loading
                self.id_to_classname = {str(i): f"Action_{i}" for i in range(400)} # Placeholder for Kinetics 400

            # Map class names to indices - REMOVED redundant/incorrect mapping
            # self.kinetics_id_to_classname = {str(v): k for k, v in self.kinetics_classnames.items()}

            # Define the video transform
            self._define_transform()
            print("Action recognition model loaded successfully.")

        except Exception as e:
            print(f"Error loading action recognition model: {e}")
            raise

    def _define_transform(self):
        """ Defines the video preprocessing transform based on the model type. """
        # Common parameters (adjust as needed for specific models)
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        num_frames = 32 # Example for SlowFast, adjust for others
        sampling_rate = 2 # Example for SlowFast, adjust for others
        frames_per_second = 30
        alpha = 4 # Example for SlowFast, adjust for others

        if "slowfast" in self.model_name:
            num_frames = 32
            sampling_rate = 2
            alpha = 4
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                ),
            )
            self.clip_duration = (num_frames * sampling_rate) / frames_per_second

        elif "slow" in self.model_name:
            num_frames = 8
            sampling_rate = 8
            alpha = None # Not used for Slow
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size)
                    ]
                ),
            )
            self.clip_duration = (num_frames * sampling_rate) / frames_per_second
        else: # Add more model types if needed (e.g., X3D)
             # Defaulting to SlowFast transform for now
            print(f"Warning: Transform for model {self.model_name} not explicitly defined. Using SlowFast defaults.")
            num_frames = 32
            sampling_rate = 2
            alpha = 4
            self.transform = ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x / 255.0),
                        NormalizeVideo(mean, std),
                        ShortSideScale(size=side_size),
                        CenterCropVideo(crop_size=(crop_size, crop_size))
                    ]
                ),
            )
            self.clip_duration = (num_frames * sampling_rate) / frames_per_second


    def predict(self, video_path):
        """
        Predicts the action in a given video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tuple: (predicted_action_label, confidence_score) or (None, None) if error.
        """
        try:
            print(f"Processing video: {video_path}")
            # Select the duration of the clip to load by specifying the start and end duration
            # The start_sec should correspond to where the action occurs in the video.
            # For demonstration, we'll load the first few seconds.
            start_sec = 0
            end_sec = start_sec + self.clip_duration

            # Initialize an EncodedVideo helper class
            video = EncodedVideo.from_path(video_path, decode_audio=False)

            # Load the desired clip
            video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

            if video_data is None or video_data.get('video') is None:
                 print(f"Warning: Could not load video clip from {video_path} between {start_sec}s and {end_sec}s.")
                 # Try loading the whole video if clip fails and duration is reasonably short
                 video_duration = video.duration
                 max_reasonable_duration = 12.0 # Define a max duration (e.g., 12 seconds, increased from 10)
                 if video_duration <= max_reasonable_duration:
                     print(f"Attempting to load full video (duration: {float(video_duration):.2f}s) as clip loading failed.")
                     video_data = video.get_clip(start_sec=0, end_sec=video_duration)
                     if video_data is None or video_data.get('video') is None:
                         print(f"Error: Failed to load any video data from {video_path}, even when trying full video.")
                         return None, None
                     print("Successfully loaded full video.")
                 else:
                     print(f"Error: Video clip is empty or invalid, and full video duration ({float(video_duration):.2f}s) exceeds reasonable limit ({max_reasonable_duration}s).")
                     return None, None

            # Apply transforms
            video_data = self.transform(video_data)

            # Move the inputs to the desired device
            inputs = video_data["video"]
            inputs = [i.to(self.device)[None, ...] for i in inputs] if isinstance(inputs, list) else inputs.to(self.device)[None, ...]

            # Pass the input clip through the model
            with torch.no_grad():
                preds = self.model(inputs)

            # Get the predicted classes
            post_act = torch.nn.Softmax(dim=1)
            preds = post_act(preds)
            pred_classes = preds.topk(k=5).indices[0]
            pred_scores = preds.topk(k=5).values[0]

            # Map the predicted classes to the label names
            pred_class_names = [self.id_to_classname[str(i.item())] for i in pred_classes]

            print(f"Top 5 predictions for {video_path}:")
            for i in range(len(pred_class_names)):
                print(f"  {pred_class_names[i]}: {pred_scores[i].item():.4f}")

            # Return the top prediction
            top_prediction_label = pred_class_names[0]
            top_prediction_score = pred_scores[0].item()

            return top_prediction_label, top_prediction_score

        except FileNotFoundError:
            print(f"Error: Video file not found at {video_path}")
            return None, None
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # This example requires a sample video file and the kinetics class names json
    # Create dummy files for basic testing if they don't exist
    import os
    dummy_video_path = "/home/ubuntu/dummy_video.mp4"
    dummy_json_path = "/home/ubuntu/kinetics_classnames.json"

    if not os.path.exists(dummy_json_path):
        print(f"Creating dummy {dummy_json_path}...")
        dummy_labels = {f"Action_{i}": i for i in range(400)}
        with open(dummy_json_path, 'w') as f:
            json.dump(dummy_labels, f)

    if not os.path.exists(dummy_video_path):
        print(f"Creating dummy video {dummy_video_path} (requires ffmpeg)...")
        # Create a short, silent, black video using ffmpeg
        os.system(f"ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=5 -vf 'fps=30' -c:v libx264 -tune zerolatency -pix_fmt yuv420p {dummy_video_path} > /dev/null 2>&1")

    if os.path.exists(dummy_video_path) and os.path.exists(dummy_json_path):
        print("\n--- Testing ActionRecognizer --- ")
        try:
            recognizer = ActionRecognizer(model_name="slow_r50") # Use a smaller model for faster testing
            pred_label, pred_score = recognizer.predict(dummy_video_path)

            if pred_label is not None:
                print(f"\nTop Prediction for dummy video: {pred_label} (Score: {pred_score:.4f})")
            else:
                print("\nAction recognition failed for dummy video.")
        except Exception as e:
            print(f"\nError during ActionRecognizer test: {e}")
        finally:
            # Clean up dummy files (optional)
            # print("Cleaning up dummy files...")
            # if os.path.exists(dummy_video_path):
            #     os.remove(dummy_video_path)
            # if os.path.exists(dummy_json_path):
            #     os.remove(dummy_json_path)
            pass # Add pass to make the block valid    else:
        print("Skipping ActionRecognizer test: Dummy files could not be created.")

