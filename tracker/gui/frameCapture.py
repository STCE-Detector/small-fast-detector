import time

import cv2
from vidgear.gears import VideoGear, CamGear
import supervision as sv


class FrameCapture:
    """
    Class to capture frames from a video source (webcam or video file) using queue and threading.
    """
    
    def __init__(self, source=0, stabilize=False, stream_mode=False, logging=False):
        self.source = source
        if isinstance(self.source, int):
            # Use CamGear for optimized live stream handling
            self.vcap = CamGear(source=self.source, stream_mode=stream_mode, logging=logging, time_delay=2)
            height, width, _ = self.vcap.frame.shape
            self.fps = self.vcap.framerate
        else:
            # Use VideoGear for general video file handling
            self.vcap = VideoGear(source=self.source, stabilize=stabilize, stream_mode=stream_mode, logging=logging,
                                  time_delay=2)
            height, width, _ = self.vcap.stream.frame.shape
            self.fps = self.vcap.stream.framerate
            self.video_info = sv.VideoInfo.from_video_path(self.source)
        self.frame_count = 0
        self.frame_size = (width, height)
        self.streaming = False

    def start(self):
        self.streaming = True
        self.vcap.start()

    def Capture(self):
        if self.streaming:
            frame = self.vcap.read()
            if frame is None:
                self.Close()
                return None
            self.frame_count += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def GetFrameCount(self):
        return self.frame_count

    def GetWidth(self):
        return self.frame_size[0]

    def GetHeight(self):
        return self.frame_size[1]

    def GetFrameRate(self):
        return self.fps

    def get_frame_size(self):
        return self.frame_size

    def IsStreaming(self):
        return self.streaming

    @property
    def eos(self):
        """
        Returns true if the stream is currently closed (EOS has been reached)
        """
        return not self.vcap.stream.stream

    def Close(self):
        # Safely close the video stream
        self.frame_count = 0
        # self.vcap.release()
        self.vcap.stop()
        self.streaming = False