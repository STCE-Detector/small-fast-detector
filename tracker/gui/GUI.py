import argparse
import sys
import cv2
import numpy as np
from time import time

import os

from PySide6.QtCore import QObject, QThread, QTimer
from PySide6.QtGui import QImage, QPainter, QFont, QColor, Qt, QPixmap, QKeyEvent
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PySide6.QtCore import Signal


os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"


class FrameWorker(QObject):
    frame_ready = Signal(QImage, float)  # Emit both the QImage and the calculated FPS

    def __init__(self, video_path):
        super(FrameWorker, self).__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Video cannot be opened")
        self.paused = False
        self.last_timestamp = time()

    def process_frames(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                current_timestamp = time()
                elapsed_time = current_timestamp - self.last_timestamp
                fps = 1 / elapsed_time if elapsed_time else 0
                self.last_timestamp = current_timestamp

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(q_image, fps)
            else:
                self.cap.release()

    def toggle_pause(self):
        self.paused = not self.paused


class VideoDisplay(QGraphicsView):
    def __init__(self, processor, sync_fps=False):
        super(VideoDisplay, self).__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setMinimumSize(640, 360)
        self.sync_fps = sync_fps

        self.thread = QThread()
        self.worker = processor
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self.update_display)
        self.thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.worker.process_video)
        self.timer.start(int(1000 / self.worker.frame_capture.get_fps() if self.sync_fps else 0))

    def update_display(self, q_image, fps):
        painter = QPainter(q_image)
        painter.setFont(QFont("Arial", 38))
        painter.setPen(QColor("yellow"))
        painter.drawText(q_image.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"FPS: {fps:.2f}")
        painter.end()

        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

        if self.worker.frame_capture.get_stop():
            self.timer.stop()
            self.thread.quit()
            self.close()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_P:
            self.worker.toggle_pause()
        elif event.key() == Qt.Key.Key_F:
            self.sync_fps = not self.sync_fps
            self.timer.start(int(1000 / self.worker.frame_capture.get_fps() if self.sync_fps else 0))
        elif event.key() == Qt.Key.Key_Q:
            self.close()
            self.worker.cleanup()
            self.thread.quit()
