import cv2


class VideoInfo:
    """
    Class to store video information
    """
    def __init__(self, source=None):
        """
        Constructor
        :param source: str, path to video file
        """
        if source is not None:
            cap = cv2.VideoCapture(source)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            self.total_frames = total_frames
            self.fps = fps
            self.resolution_wh = (width, height)
        else:
            self.total_frames = None
            self.fps = None
            self.resolution_wh = None

    def manual_init(self, total_frames, fps, resolution_wh):
        """
        Manual initialization
        :param total_frames: int, total number of frames
        :param fps: float, frames per second
        :param resolution_wh: tuple, resolution (width, height)
        """
        self.total_frames = total_frames
        self.fps = fps
        self.resolution_wh = resolution_wh




