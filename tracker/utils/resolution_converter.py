import os
from moviepy.editor import VideoFileClip


def resize_video(input_path, output_path, max_width=1920, max_height=1080):
    # Load the video
    clip = VideoFileClip(input_path)

    # Get the original dimensions
    width, height = clip.size

    # Calculate the new dimensions while preserving aspect ratio
    if width <= max_width and height <= max_height:
        new_width, new_height = width, height
    else:
        aspect_ratio = width / height
        if width > height:
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)

    # Resize the video
    resized_clip = clip.resize((new_width, new_height))

    # Write the resized video to a new file
    resized_clip.write_videofile(output_path)


def batch_resize_videos(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp4"):  # Assuming all videos are in mp4 format
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            resize_video(input_path, output_path)


# Specify the input and output folders
input_folder = "/Users/inaki-eab/Desktop/TAR_data/videos_30fps"
output_folder = "/Users/inaki-eab/Desktop/TAR_data/videos_30fps_resized_2ND"

# Resize all videos in the input folder and save them to the output folder
batch_resize_videos(input_folder, output_folder)