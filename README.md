# YOLOv8 Object Tracking with Audio Recording

This Python script demonstrates an application that uses the YOLOv8 model for real-time object tracking in video frames captured from a webcam. It also includes functionality for audio recording when an object is detected, saving the recorded audio as a WAV file. The application uses OpenCV, Ultralytics YOLO, PyAudio, and multithreading to accomplish these tasks.

## Prerequisites

Before running the script, ensure you have the following Python libraries installed:

- `cv2` (OpenCV)
- `ultralytics` (YOLO model)
- `pyaudio` (audio recording)

You can install these libraries using pip:

```bash
pip install opencv-python-headless ultralytics pyaudio
```

To run the script

```bash
python3 main.py
```

## Output
Annotated video frames are displayed in real-time with detected objects.
If an object is detected, audio recording starts, and the recorded audio is saved as "audio_output.wav" in the script's directory.
Press 'q' to exit the script gracefully.