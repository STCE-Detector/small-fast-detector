import time

import cv2
from deffcode import FFdecoder
from ultralytics.utils import IS_JETSON


class FFmpegFrameCapture:
    def __init__(self, source, frame_format='bgr24', verbose=False):
        self.source = source
        self.frame_format = frame_format
        self.verbose = verbose
        self.frame_count = 0
        self.decoder = None
        self.stopped = True

    def start(self):

        if IS_JETSON:
            ffparams = {
                "-vcodec": "h264_nvv4l2dec",  # use H.264 CUVID Video-decoder
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

    def capture(self):
        if not self.stopped:
            for frame in self.decoder.generateFrame():
                if frame is None:
                    self.stop()
                    return None
                self.frame_count += 1
                return frame
        return None

    def stop(self):
        if not self.stopped:
            self.decoder.terminate()
            self.stopped = True

    def get_frame_count(self):
        return self.frame_count

    def is_streaming(self):
        return not self.stopped

if __name__ == '__main__':
    video_path = "/home/johnny/Projects/small-fast-detector/tracker/videos/output.mp4"
    capture = FFmpegFrameCapture(video_path)
    capture.start()
    print(capture.decoder.metadata)
    start_time = time.time()
    try:
        while True:
            frame = capture.capture()
            if frame is None:
                break
            cv2.imshow("Frame", frame)
            elapsed_time = time.time() - start_time
            fps = capture.get_frame_count() / elapsed_time if elapsed_time > 0 else 0
            print(f"Current FPS: {fps:.2f}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        capture.stop()
        cv2.destroyAllWindows()
        print("Exiting program")