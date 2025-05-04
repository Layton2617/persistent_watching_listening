# /home/ubuntu/persistent_watching_listening/utils/video_utils.py

import cv2
import time

class VideoStream:
    """Handles video input from webcam or file."""
    def __init__(self, source=0):
        """Initializes the video stream.

        Args:
            source: Video source. Can be an integer for webcam index (e.g., 0)
                    or a string path to a video file.
        """
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Video source opened: {source} ({self.width}x{self.height} @ {self.fps:.2f} FPS)")

    def read_frame(self):
        """Reads the next frame from the video stream.

        Returns:
            A tuple: (success, frame)
            success: Boolean indicating if a frame was successfully read.
            frame: The video frame (NumPy array) or None if reading failed.
        """
        success, frame = self.cap.read()
        return success, frame

    def release(self):
        """Releases the video capture object."""
        if self.cap:
            self.cap.release()
            print(f"Video source released: {self.source}")

    def is_opened(self):
        """Checks if the video source is opened."""
        return self.cap.isOpened()

# --- Example Usage --- 
if __name__ == '__main__':
    # Example using webcam
    try:
        video_stream = VideoStream(source=0) # Use 0 for default webcam
    except IOError as e:
        print(e)
        exit()

    frame_count = 0
    start_time = time.time()

    while video_stream.is_opened():
        success, frame = video_stream.read_frame()
        if not success:
            print("End of video stream or error.")
            break

        frame_count += 1

        # Display the frame (optional)
        cv2.imshow('Video Stream Test', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'): # Use waitKey(1) for video feel
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    calculated_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({calculated_fps:.2f} FPS).")

    video_stream.release()
    cv2.destroyAllWindows()

