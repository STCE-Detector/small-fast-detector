import time
from deffcode import FFdecoder

# Path to the video file
video_path = "/Users/johnny/Projects/small-fast-detector/tracker/videos/demo.mp4"

# Initialize and formulate the decoder
decoder = FFdecoder(video_path).formulate()

# Initialize a list to store frame timestamps
frame_times = []

# Generate and process frames
try:
    start_time = time.time()  # Record the start time of the frame processing

    for frame in decoder.generateFrame():
        if frame is None:
            break

        # Process the frame (e.g., display or analyze)
        # {do something with the frame here}
        print(frame.shape)  # For example, print the shape of the frame (1080, 1920, 3)

        # Record the timestamp after processing each frame
        frame_times.append(time.time())

finally:
    # Always ensure the decoder is properly terminated
    decoder.terminate()

# Calculate and print the FPS
if frame_times:
    total_time = frame_times[-1] - start_time
    num_frames = len(frame_times)
    fps = num_frames / total_time
    print(f"Processed {num_frames} frames in {total_time:.2f} seconds, resulting in an FPS of {fps:.2f}")
else:
    print("No frames were processed.")
