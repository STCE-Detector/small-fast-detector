import cv2


class VideoInfo:
    """
    Class to store video information
    """
    def __init__(self, source):
        cap = cv2.VideoCapture(source)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        self.total_frames = total_frames
        self.fps = fps
        self.resolution_wh = (width, height)