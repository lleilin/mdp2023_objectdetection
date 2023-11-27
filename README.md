# YOLOv8 Object Tracking with Audio Recording

This Python script demonstrates an application that uses the YOLOv8 model for real-time object tracking in video frames captured from a webcam. It also includes functionality for audio recording when an object is detected, saving the recorded audio as a WAV file.

## Installation
1. Install Python 3.x 
2. Install required Python packages by running:
```
pip3 install -r requirements.txt
```

## Usage
Run the script using the following command:

```
python3 script_name.py [source]
```
source (optional): Video source. Default is 0 (camera). You can specify a video file or a different camera source.

The script will display the video stream with face detection. Audio recording will be triggered when a face is detected, and the recorded audio will be saved as a WAV file.

Press Ctrl+C to exit the script.

## Configuration
- FPS: Frames per second for the video stream.
- chunk, sample_format, channels, fs, seconds: Audio recording settings.
- stop_threshold: The number of frames without face detection to stop recording.
