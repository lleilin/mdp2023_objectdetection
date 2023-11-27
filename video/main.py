from ultralytics import YOLO
import pyaudio
import wave
import threading
import time
import sys

FPS = 5

# Initialize the YOLO model
model = YOLO("yolov8n.pt")

# Initialize the audio recording settings
chunk = 1024
sample_format = pyaudio.paInt16
channels = 1  # Changed to 1 (mono)
fs = 44100
seconds = 5

p = pyaudio.PyAudio()

# Create a thread for audio recording
audio_thread = None

# Flag to indicate whether audio recording is in progress
is_recording = False

# Frame counter and threshold to stop recording
frame_counter = 0
stop_threshold = 10


# Function to record audio with a unique filename
def record_audio(output_filename):
    global is_recording
    global frame_counter
    is_recording = True

    stream = p.open(
        format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=chunk,
        input=True,
    )

    frames = []

    print("Recording...")
    # for _ in range(0, int(fs / chunk * seconds)):
    while frame_counter <= stop_threshold:
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()

    wf = wave.open(output_filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()

    is_recording = False


# Function to start audio recording thread with a unique filename
def start_audio_recording():
    global audio_thread
    if not is_recording:
        current_time = time.strftime("%Y%m%d-%H%M%S")
        audio_filename = f"audio_{current_time}.wav"
        audio_thread = threading.Thread(target=record_audio, args=(audio_filename,))
        audio_thread.start()


# Main function
def main():
    global frame_counter
    source = 0
    if len(sys.argv) > 1:
        source = sys.argv[1]
    try:
        start_time = time.time()
        for result in model.track(
            source=source, show=True, stream=True, agnostic_nms=True, save_crop=True, classes=0
        ):
            if len(result):
                print("DETECTED")
                frame_counter = 0  # Reset the frame counter when a face is detected
                # Start recording audio in a separate thread
                start_audio_recording()
            else:
                frame_counter += 1  # Increment the frame counter

            elapsed_time = time.time() - start_time
            delay = max(0, 1 / FPS - elapsed_time)
            time.sleep(delay)
            start_time = time.time()

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting...")
        frame_counter=11
    finally:
        terminate_audio_stream()


# Terminate the audio stream
def terminate_audio_stream():
    global audio_thread
    if audio_thread:
        audio_thread.join()  # Wait for the audio recording thread to finish
    p.terminate()


if __name__ == "__main__":
    try:
        main()
    finally:
        terminate_audio_stream()
