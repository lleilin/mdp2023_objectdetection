import cv2
from ultralytics import YOLO
import time
import pyaudio
import wave
import numpy as np

# Load the YOLOv8 model
model = YOLO('best.pt')

# Open the video file
cap = cv2.VideoCapture(0)

# Set the resolution to 480x640
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize a flag to False
object_detected = False

# Initialize a timer
last_detection_time = time.time()

# Initialize audio recording parameters
audio_output_filename = "audio_output.wav"
audio_format = pyaudio.paInt16
audio_channels = 1
audio_rate = 11025  # Reduced audio rate
audio_chunk_size = 2048

# Initialize audio recording variables
audio_frames = []
recording = False

# Create a PyAudio object
p = pyaudio.PyAudio()

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, conf=0.7, persist=True)

        # Check if objects have been detected
        if len(results[0]) > 0:
            object_detected = True
            last_detection_time = time.time()

            # Start recording audio if not already recording
            if not recording:
                recording = True
                audio_frames = []
                audio_stream = p.open(format=audio_format, channels=audio_channels,
                                      rate=audio_rate, input=True, frames_per_buffer=audio_chunk_size)
                print("Recording audio...")

        # Check if one second has elapsed since the last detection
        if time.time() - last_detection_time >= 1:
            object_detected = False

            # Stop recording audio if currently recording
            if recording:
                recording = False
                audio_stream.stop_stream()
                audio_stream.close()
                p.terminate()

                # Save recorded audio as a WAV file
                wf = wave.open(audio_output_filename, 'wb')
                wf.setnchannels(audio_channels)
                wf.setsampwidth(p.get_sample_size(audio_format))
                wf.setframerate(audio_rate)
                wf.writeframes(b''.join(audio_frames))
                wf.close()
                print(f"Audio saved as {audio_output_filename}")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Print the flag value (True if detected within the last second, otherwise False)
        print("Object Detected:", object_detected)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    if recording:
        # Record audio while an object is detected
        audio_data = audio_stream.read(audio_chunk_size)
        audio_frames.append(audio_data)

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
