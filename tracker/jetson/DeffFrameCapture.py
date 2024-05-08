import time
from threading import Thread

import cv2
import numpy as np
from deffcode import FFdecoder
from ultralytics.utils import IS_JETSON
import platform
if IS_JETSON:
    from jetson_utils import cudaFromNumpy
def check_os():
    os = platform.system()
    if os == "Darwin":
        return "MacOS"
    elif os == "Linux":
        return "Linux"
    else:
        return "Unknown OS"

class DeffFrameCapture:
    def __init__(self, source, frame_format='bgr24', verbose=False):
        self.source = source
        self.frame_format = frame_format
        self.verbose = verbose
        self.frame_count = 0
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
        self.decoder = self.decoder.formulate()
        self.stopped = False
        self.thread = Thread(target=self.Capture)
        self.thread.start()

    def Capture(self):
        if not self.stopped:
            for frame in self.decoder.generateFrame():
                if frame is None:
                    self.stop()
                    return None
                self.frame_count += 1
                cuda_array = cudaFromNumpy(np.array(frame)) if IS_JETSON else np.array(frame)
                return cuda_array
        return None

    def stop(self):
        if not self.stopped:
            self.stopped = True
            self.decoder.terminate()
            self.thread.join()

    def Close(self):
        self.stop()

    def GetFrameRate(self):
        return self.decoder.fps

    def GetFrameCount(self):
        return self.frame_count

    def GetWidth(self):
        return self.decoder.width

    def GetHeight(self):
        return self.decoder.height

    def IsStreaming(self):
        return not self.stopped

if __name__ == '__main__':
    video_path = "/Users/johnny/Projects/small-fast-detector/tracker/videos/demo.mp4"
    capture = DeffFrameCapture(video_path)
    capture.start()
    print(capture.decoder.metadata)
    start_time = time.time()
    try:
        while True:
            frame = capture.Capture()
            if frame is None:
                break
            cv2.imshow("Frame", frame)
            elapsed_time = time.time() - start_time
            fps = capture.GetFrameCount() / elapsed_time if elapsed_time > 0 else 0
            print(f"Current FPS: {fps:.2f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Exiting program")