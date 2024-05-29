import cv2
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst
import numpy as np
import threading
import time
import supervision as sv

Gst.init(None)


class GStreamerFrameCapture:
    def __init__(self, source):
        self.source = source
        self.pipeline = None
        self.appsink = None
        self.stop_event = threading.Event()
        self.frame_queue = []
        self.frame_count = 0
        self.video_info = sv.VideoInfo.from_video_path(self.source)
        self.setup_pipeline()
        self.frame_size = (self.video_info.width, self.video_info.height)
        self.frame_rate = self.video_info.fps

    def setup_pipeline(self):
        pipeline_str = (f"filesrc location={self.source} ! decodebin ! videoconvert ! "
                        "video/x-raw,format=RGB ! videoscale ! appsink name=sink emit-signals=True sync=True")
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.appsink = self.pipeline.get_by_name('sink')

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def run(self):
        self.pipeline.set_state(Gst.State.PLAYING)
        while not self.stop_event.is_set():
            frame = self.capture()
            if frame is not None:
                self.frame_queue.append(frame)
            else:
                break
        self.pipeline.set_state(Gst.State.NULL)

    def GetFrameCount(self):
        return self.frame_count

    def GetWidth(self):
        return self.frame_size[0]

    def GetHeight(self):
        return self.frame_size[1]

    def GetFrameRate(self):
        return self.frame_rate

    def IsStreaming(self):
        return not self.stop_event.is_set()

    def capture(self):
        sample = self.appsink.emit('pull-sample')
        if sample:
            buf = sample.get_buffer()
            caps = sample.get_caps()
            try:
                width = caps.get_structure(0).get_value('width')
                height = caps.get_structure(0).get_value('height')
                # Handle format conversion here
                # Example conversion assuming the buffer is RGB
                frame = np.ndarray((height, width, 3), buffer=buf.extract_dup(0, buf.get_size()), dtype=np.uint8)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility
                return frame
            except Exception as e:
                print(f"Error extracting frame: {e}")
                return None
        else:
            print("No more frames available or pipeline error.")
            return None

    def Capture(self):
        if self.frame_queue:
            return self.frame_queue.pop(0)
        return None

    def Stop(self):
        self.frame_count = 0
        self.stop_event.set()

    def Close(self):
        self.Stop()


if __name__ == "__main__":
    video_path = '../videos/demo.mp4'
    capturer = GStreamerFrameCapture(video_path)
    capturer.thread = threading.Thread(target=capturer.run)
    capturer.thread.start()

    frame_count = 0
    start_time = time.time()

    try:
        while capturer.thread.is_alive():
            frame = capturer.get_next_frame()
            if frame is not None:
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                print(f"Frame {frame_count}: FPS={fps:.2f}")
    finally:
        capturer.close()
        print("Video capture stopped.")
