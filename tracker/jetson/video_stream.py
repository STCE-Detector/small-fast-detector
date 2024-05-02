#!/usr/bin/env python3
import argparse
import logging
import sys

from tracker.gui.jetson import VideoSource, VideoOutput
from tracker.jetson.agent import Agent


class VideoStream(Agent):
    """
    Relay, view, or test a video stream.  Use the ``--video-input`` and ``--video-output`` arguments
    to set the video source and output protocols used from `jetson_utils <https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md>`_
    like V4L2, CSI, RTP/RTSP, WebRTC, or static video files.

    For example, this will capture a V4L2 camera and serve it via WebRTC with H.264 encoding:

    .. code-block:: text

        python3 -m nano_llm.agents.video_stream \
           --video-input /dev/video0 \
           --video-output webrtc://@:8554/output

    It's also used as a basic test of video streaming before using more complex agents that rely on it.
    """

    def __init__(self, video_input=None, video_output=None, video_output_bool=False, **kwargs):
        """
        Args:
          video_input (Plugin|str): the VideoSource plugin instance, or URL of the video stream or camera device.
          video_output (Plugin|str): the VideoOutput plugin instance, or output stream URL / device ID.
        """
        super().__init__()

        self.video_source = VideoSource(video_input, **kwargs)
        self.video_output = VideoOutput(video_output, **kwargs)
        self.video_output_bool = video_output_bool

        self.video_source.add(self.on_video, threaded=False)
        if not self.video_output_bool:
            self.video_source.add(self.video_output)

    def frame(self):
        self.video_source.open()
        self.video_output.open()
    def on_video(self, image):
        logging.debug(f"captured {image.width}x{image.height} frame from {self.video_source.resource}")

    def on_exit(self):
        self.video_source.close()
        self.video_output.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    args = parser.parse_known_args()[0]

    # Setup logging
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    try:
        # Create and start video stream
        video_stream = VideoStream(video_input=args.video_input, video_output=args.video_output)
        video_stream.video_source.run()  # Begin processing the video stream
    except Exception as e:
        logging.error("Failed to start video stream: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
