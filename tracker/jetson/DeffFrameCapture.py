import platform
import time
from threading import Thread

import cv2
import numpy as np
from deffcode import FFdecoder, Sourcer

from ultralytics.utils import IS_JETSON

if IS_JETSON:
    from jetson_utils import (cudaFromNumpy, cudaAllocMapped, cudaConvertColor,cudaDeviceSynchronize)


def check_os():
    os = platform.system()
    if os == "Darwin":
        return "MacOS"
    elif os == "Linux":
        return "Linux"
    else:
        return "Unknown OS"


class DeffFrameCapture:
    """ Class to capture frames from a video source (webcam or video file) using FFmpeg.
        With FFMPEG you can use hardware acceleration to decode the video stream.
    """
    def __init__(self, source, frame_format='rgb8', verbose=False):
        self.source = source
        self.frame_format = frame_format
        self.verbose = verbose
        self.decoder = None
        self.stopped = True
    def start(self):
        operating_system = check_os()
        if IS_JETSON:
            ffparams = {
                "-vcodec": "h264_nvv4l2dec",  # use H.264 CUVID Video-decoder
            }
        elif operating_system == "MacOS":
            ffparams = {
                # "-vcodec": "h264_videotoolbox",  # use H.264 VideoToolbox decoder
                # "-hwaccel": "videotoolbox",  # accelerator
            }
        else:
            ffparams = {
                "-vcodec": None,  # skip source decoder and let FFmpeg chose
                "-ffprefixes": [
                    "-vsync",
                    "0",  # prevent duplicate frames
                    "-hwaccel",
                    "cuda",  # accelerator
                    "-hwaccel_output_format",
                    "cuda",  # output accelerator
                ],
                "-custom_resolution": "null",  # discard source `-custom_resolution`
                "-framerate": "null",  # discard source `-framerate`
            }

        self.decoder = FFdecoder(self.source, frame_format=self.frame_format, verbose=self.verbose, **ffparams)
        self.fps = int(Sourcer(self.source).probe_stream().retrieve_metadata().get('source_video_framerate'))
        self.decoder = self.decoder.formulate()
        self.streaming = True
        self.frame_count = 0
        self.thread = Thread(target=self.Capture)
        self.thread.start()

    def Capture(self):
        if self.streaming:
            for frame in self.decoder.generateFrame():
                if frame is None:
                    self.Close()
                    return None
                self.frame_count += 1
                cuda_array = cudaFromNumpy(frame, isBGR=False) if IS_JETSON else np.array(frame)
                if IS_JETSON:
                    cudaDeviceSynchronize()
                return cuda_array
        return None

    def stop(self):
        self.streaming = False
        self.frame_count = 0
        self.decoder.terminate()
        self.thread.join()

    def Close(self):
        self.stop()

    def GetFrameRate(self):
        return self.fps

    def GetFrameCount(self):
        return self.frame_count

    def GetWidth(self):
        return self.decoder.width

    def GetHeight(self):
        return self.decoder.height

    def IsStreaming(self):
        return self.streaming