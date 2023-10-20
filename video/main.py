import cv2
import numpy as np
from ultralytics import YOLO
import pyaudio
import wave

# Initialize the YOLO model
model = YOLO("best.pt")

# Initialize the audio recording settings
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1  # Changed to 1 (mono)
fs = 44100
seconds = 5

p = pyaudio.PyAudio()

# Function to record audio
def record_audio(output_filename):
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)

    frames = []

    print("Recording...")
    for _ in range(0, int(fs / chunk * seconds)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

# Main function
def main():
    for result in model.track(source=0, show=True, stream=True, agnostic_nms=True, save_crop=True):
        if len(result):
            print("DETECTED")

            # Start recording audio
            record_audio("detected_audio.wav")

if __name__ == "__main__":
    main()

# Terminate the audio stream
p.terminate()
