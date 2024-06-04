from moviepy.editor import VideoFileClip
import os

root = "/Users/inaki-eab/Desktop/TAR_data/"
original_videos = root + "2ndround/"
original_videos_list = os.listdir(original_videos)

for video in original_videos_list:
    if video.endswith(".mp4"):
        new_fps = 30
        clip = VideoFileClip(original_videos + video)
        original_fps = clip.fps
        if original_fps <= new_fps:
            new_fps = original_fps
        clip = clip.set_fps(new_fps)
        clip.write_videofile(root + "videos_30fps/" + video)
