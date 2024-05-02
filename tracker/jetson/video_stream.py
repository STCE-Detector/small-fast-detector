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

    def __init__(self, video_input=None, video_output=None, **kwargs):
        """
        Args:
          video_input (Plugin|str): the VideoSource plugin instance, or URL of the video stream or camera device.
          video_output (Plugin|str): the VideoOutput plugin instance, or output stream URL / device ID.
        """
        super().__init__()

        self.video_source = VideoSource(video_input, **kwargs)
        self.video_output = VideoOutput(video_output, **kwargs)

        self.video_source.add(self.on_video, threaded=False)
        self.video_source.add(self.video_output)

        self.pipeline = [self.video_source]

    def on_video(self, image):
        logging.debug(f"captured {image.width}x{image.height} frame from {self.video_source.resource}")

class ArgParser(argparse.ArgumentParser):
    Video = ['video_input', 'video_output']

    def __init__(self, extras=Video, **kwargs):
        super().__init__(formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs)
        if 'video_input' in extras:
            self.add_argument("--video-input", type=str, default=None,
                              help="video camera device name, stream URL, file/dir path")
            self.add_argument("--video-input-width", type=int, default=None,
                              help="manually set the resolution of the video input")
            self.add_argument("--video-input-height", type=int, default=None,
                              help="manually set the resolution of the video input")
            self.add_argument("--video-input-codec", type=str, default=None,
                              choices=['h264', 'h265', 'vp8', 'vp9', 'mjpeg'],
                              help="manually set the input video codec to use")
            self.add_argument("--video-input-framerate", type=int, default=None,
                              help="set the desired framerate of input video")
            self.add_argument("--video-input-save", type=str, default=None,
                              help="path to video file to save the incoming video feed to")

        if 'video_output' in extras:
            self.add_argument("--video-output", type=str, default=None, help="display, stream URL, file/dir path")
            self.add_argument("--video-output-codec", type=str, default=None,
                              choices=['h264', 'h265', 'vp8', 'vp9', 'mjpeg'], help="set the output video codec to use")
            self.add_argument("--video-output-bitrate", type=int, default=None, help="set the output bitrate to use")
            self.add_argument("--video-output-save", type=str, default=None,
                              help="path to video file to save the outgoing video stream to")

    def parse_args(self, **kwargs):
        """
        Override for parse_args() that does some additional configuration
        """
        args = super().parse_args(**kwargs)
        logging.debug(f"{args}")
        return args

def main():
    parser = ArgParser()
    parser.add_argument('--video-input', type=str, default='/dev/video0',
                        help='URL of the video stream or camera device.')
    parser.add_argument('--video-output', type=str, default='display://',
                        help='Output stream URL or device ID.')

    args = parser.parse_args()

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
