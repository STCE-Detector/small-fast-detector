import sys
import os
import cv2
import csv
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QFileDialog, QHBoxLayout,
                             QTableWidget, QTableWidgetItem, QSizePolicy, QScrollArea, QCheckBox, QSplitter, QSlider, QStyleFactory, QMessageBox, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from collections import deque


class VideoAnnotationTool(QWidget):
    """
    A simple video annotation tool for annotating bounding boxes and actions in a video sequence.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('📹 Video Annotation Tool 📝')
        self.setGeometry(100, 100, 1200, 1000)
        self.setMinimumSize(1400, 800)
        self.fa_distance_threshold = 4
        self.layout = QVBoxLayout()

        # Paths
        self.img_seq_path = ""
        self.gt_file_path = ""

        # Image and GT loading
        self.image_files = []
        self.gt_coordinates_dict = {}
        self.current_frame = 0
        self.frame_rate = 30  # Assuming 30 fps for calculation

        # Color state for each tracker ID
        self.tracker_color_state = {}

        # Scroll area for the image
        self.scroll_area = QScrollArea()
        self.image_label = QLabel(self)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # Button layout
        self.button_layout = QHBoxLayout()

        # Buttons for loading new paths and navigation
        self.btn_load_img_seq = QPushButton('Load Image Sequence')
        self.btn_load_img_seq.clicked.connect(lambda: self.load_new_img_seq(True))
        self.btn_load_img_seq.setStyleSheet("background-color: #87CEEB; font-weight: bold;")
        self.button_layout.addWidget(self.btn_load_img_seq)

        self.btn_load_gt = QPushButton('Load GT File')
        self.btn_load_gt.clicked.connect(lambda: self.load_new_gt(True))
        self.btn_load_gt.setStyleSheet("background-color: #87CEEB; font-weight: bold;")
        self.button_layout.addWidget(self.btn_load_gt)

        self.add_annotation_button = QPushButton('Add Annotation')
        self.add_annotation_button.clicked.connect(self.add_annotation)
        self.add_annotation_button.setStyleSheet("background-color: #32CD32; font-weight: bold;")
        self.button_layout.addWidget(self.add_annotation_button)

        self.export_button = QPushButton('Export Annotations')
        self.export_button.clicked.connect(self.export_annotations)
        self.export_button.setStyleSheet("background-color: #FFD700; font-weight: bold;")
        self.button_layout.addWidget(self.export_button)

        self.import_button = QPushButton('Import Annotations')
        self.import_button.clicked.connect(self.import_annotations)
        self.import_button.setStyleSheet("background-color: #FFD700; font-weight: bold;")
        self.button_layout.addWidget(self.import_button)

        self.repeat_checkbox = QCheckBox('Repeat Video')
        self.repeat_checkbox.setStyleSheet("font-weight: bold;")
        self.button_layout.addWidget(self.repeat_checkbox)

        self.slow_motion_checkbox = QCheckBox('Slow Motion')
        self.slow_motion_checkbox.setStyleSheet("font-weight: bold;")
        self.button_layout.addWidget(self.slow_motion_checkbox)

        self.show_actions_checkbox = QCheckBox('Show Actions')
        self.show_actions_checkbox.setChecked(False)
        self.show_actions_checkbox.setStyleSheet("font-weight: bold;")
        self.show_actions_checkbox.setEnabled(False)  # Initially disabled
        self.button_layout.addWidget(self.show_actions_checkbox)

        self.show_only_actions_checkbox = QCheckBox('Only Actions')
        self.show_only_actions_checkbox.setChecked(False)
        self.show_only_actions_checkbox.setStyleSheet("font-weight: bold;")
        self.show_only_actions_checkbox.setEnabled(False)  # Initially disabled
        self.button_layout.addWidget(self.show_only_actions_checkbox)

        self.slow_motion_checkbox.stateChanged.connect(self.adjust_playback_speed)
        self.repeat_checkbox.stateChanged.connect(self.adjust_playback_speed)

        # Add gathering Annotation Button
        self.btn_add_gathering_annotation = QPushButton('Add gathering Annotation')
        self.btn_add_gathering_annotation.setEnabled(False)  # Initially disabled
        self.btn_add_gathering_annotation.clicked.connect(self.add_gathering_annotation)
        self.btn_add_gathering_annotation.setStyleSheet("background-color: #32CD32; font-weight: bold;")
        self.button_layout.addWidget(self.btn_add_gathering_annotation)

        # Slider for video navigation
        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setMinimum(0)
        self.video_slider.sliderMoved.connect(self.slider_changed)
        self.layout.addLayout(self.button_layout)
        self.layout.addWidget(self.video_slider)

        # Fast approaching zone
        self.draw_half_circle = True  # Control if draw or not the zone

        # Boundary lines
        self.boundaries = {
            'ob_1': [802, 676, 822, 658, "right"],
            'ob_2': [831, 550, 857, 552, "left"],
            'ob_3': [927, 636, 1169, 638, "bottom"],
            'ob_4': [748, 549, 780, 555, "left"],
            'ob_5': [624, 841, 869, 842, "bottom"],
        }

        # Tables for displaying file paths and annotations
        self.table_widget = QTableWidget()
        self.table_widget.setRowCount(1)
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(['Image Sequence Path', 'GT File Path', 'FR', 'Frame', 'Detections', 'Time'])
        self.table_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.annotations_table = QTableWidget()
        self.annotations_table.setRowCount(0)
        self.annotations_table.setColumnCount(8)  # ID, Start Frame, End Frame, SS, SR, FA, G, OB
        self.annotations_table.setHorizontalHeaderLabels(['ID', 'Start Frame', 'End Frame', 'SS', 'SR', 'FA', 'G', 'OB'])
        self.annotations_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        # New table for tracker IDs
        self.tracker_table = QTableWidget()
        self.tracker_table.setRowCount(0)
        self.tracker_table.setColumnCount(1)  # Tracker IDs
        self.tracker_table.setHorizontalHeaderLabels(['Tracker IDs'])
        self.tracker_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.tracker_table.cellClicked.connect(self.highlight_tracker_id)

        # Using QSplitter to allow dynamic resizing
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.scroll_area)  # Add scroll area to splitter
        self.tables_splitter = QSplitter(Qt.Horizontal)
        self.tables_splitter.addWidget(self.table_widget)
        self.tables_splitter.addWidget(self.annotations_table)
        self.tables_splitter.addWidget(self.tracker_table)  # Add the tracker table
        self.splitter.addWidget(self.tables_splitter)  # Add tables splitter to main splitter

        self.layout.addWidget(self.splitter)  # Add splitter to the main layout
        self.splitter.setSizes([700, 300])  # Initial size allocation, adjust as necessary

        # Navigation button layout
        self.nav_button_layout = QHBoxLayout()
        self.btn_prev_frame = QPushButton('⬅️ Previous Frame')
        self.btn_prev_frame.setStyleSheet("background-color: #ADD8E6; font-weight: bold;")
        self.btn_prev_frame.setFixedSize(150, 40)
        self.btn_prev_frame.clicked.connect(self.prev_frame)
        self.nav_button_layout.addWidget(self.btn_prev_frame)

        self.play_button = QPushButton('Play/Pause [Space]')
        self.play_button.setStyleSheet("background-color: #90EE90; font-weight: bold;")
        self.play_button.setFixedSize(150, 40)
        self.play_button.clicked.connect(self.play_pause_video)
        self.nav_button_layout.addWidget(self.play_button)

        self.btn_next_frame = QPushButton('Next Frame ➡️')
        self.btn_next_frame.setStyleSheet("background-color: #ADD8E6; font-weight: bold;")
        self.btn_next_frame.setFixedSize(150, 40)
        self.btn_next_frame.clicked.connect(self.next_frame)
        self.nav_button_layout.addWidget(self.btn_next_frame)

        self.btn_exit = QPushButton('Exit')
        self.btn_exit.setStyleSheet("background-color: #FF0000; color: white; font-weight: bold;")
        self.btn_exit.setFixedSize(150, 40)
        self.btn_exit.clicked.connect(self.close)
        self.nav_button_layout.addWidget(self.btn_exit, alignment=Qt.AlignRight)

        self.layout.addLayout(self.nav_button_layout)

        self.setLayout(self.layout)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.playing = False
        self.slider_busy = False

        # Keyboard shortcuts
        self.setFocusPolicy(Qt.StrongFocus)

        # Initialize with the first frame
        self.update_frame()

        # Trace checkbox and frame input
        self.trace_checkbox = QCheckBox('Trace bbx')
        self.trace_checkbox.setStyleSheet("font-weight: bold;")
        self.button_layout.addWidget(self.trace_checkbox)

        self.trace_frames_input = QLineEdit()
        self.trace_frames_input.setPlaceholderText('Trace Frames')
        self.trace_frames_input.setFixedWidth(100)
        self.button_layout.addWidget(self.trace_frames_input)

        # Initialize trajectory storage
        self.trajectories = {}
        self.max_trace_frames = 30

    def load_new_img_seq(self, change_color=False):
        """
        Load a new image sequence from a directory.
        Args:
            change_color (bool): Change the button color to green if True.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            self.img_seq_path = folder
            self.image_files = []
            self.load_image_sequence(self.img_seq_path)
            self.current_frame = 0
            if change_color:
                self.btn_load_img_seq.setStyleSheet("background-color: lightgreen;")
            self.update_table()
            self.video_slider.setMaximum(len(self.image_files) - 1)  # Update slider range

    def load_new_gt(self, change_color=False):
        """
        Load a new ground truth file.
        Args:
            change_color (bool): Change the button color to green if True.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select GT File", "", "Text Files (*.txt)")
        if file_path:
            self.gt_file_path = file_path
            self.gt_coordinates_dict, self.has_actions = self.load_gt_coordinates(self.gt_file_path)
            if change_color:
                self.btn_load_gt.setStyleSheet("background-color: lightgreen;")
            self.update_table()

            # Clear the annotations table
            self.clear_annotations_table()

            if not self.has_actions:
                self.show_actions_checkbox.setChecked(False)
                self.show_actions_checkbox.setEnabled(False)
                self.show_only_actions_checkbox.setChecked(False)
                self.show_only_actions_checkbox.setEnabled(False)
                QMessageBox.warning(self, "GT File Warning", "The GT file does not contain action columns.")
            else:
                self.show_actions_checkbox.setEnabled(True)
                self.show_actions_checkbox.setChecked(True)
                self.show_only_actions_checkbox.setEnabled(True)
                self.show_only_actions_checkbox.setChecked(False)

    def clear_annotations_table(self):
        """
        Clear the annotations table.
        """
        self.annotations_table.setRowCount(0)

    def load_image_sequence(self, image_sequence_path):
        """
        Load an image sequence from a directory.
        Args:
            image_sequence_path (str): Path to the image sequence directory.
        """
        self.image_files = sorted([os.path.join(image_sequence_path, f) for f in os.listdir(image_sequence_path) if f.endswith('.jpg') or f.endswith('.png')])

    def load_gt_coordinates(self, gt_file_path):
        """
        Load ground truth coordinates from a file.
        Args:
            gt_file_path (str): Path to the ground truth file.
        Returns:
            dict: Dictionary containing the ground truth coordinates for each frame.
        """
        df = pd.read_csv(gt_file_path, header=None)
        has_actions = False
        num_cols = df.shape[1]
        col_names = ['frame', 'id', 'x', 'y', 'w', 'h', 'x/conf', 'y/class', 'z/vis', 'SS', 'SR', 'FA', 'G', 'OB']
        df.columns = col_names[:num_cols]

        if num_cols > 11:
            has_actions = True
        else:
            # Select only the first 9 columns
            df = df.iloc[:, :9] if num_cols > 9 else df     # Correct Hieve 10th column

        # Clean up the dataframe if coming from MOT
        # TODO: maybe do not consider last two classes (distractor:8 and reflection:12)
        if 'MOT' in gt_file_path:
            correct_classes = [1, 2, 7]
            df = df[df['y/class'].isin(correct_classes)].copy()

        # Convert to dictionary
        if has_actions:
            self.active_actions = ['SS', 'SR', 'FA', 'G', 'OB'] if 'OB' in df.columns else ['SS', 'SR', 'FA', 'G']
            gt_coordinates_dict = {frame: group[['x', 'y', 'w', 'h', 'id'] + self.active_actions].values.tolist()
                                   for frame, group in df.groupby('frame')}
        else:
            gt_coordinates_dict = {frame: group[['x', 'y', 'w', 'h', 'id']].values.tolist()
                                   for frame, group in df.groupby('frame')}
        return gt_coordinates_dict, has_actions

    def play_pause_video(self):
        """
        Play or pause the video sequence.
        """
        if self.playing:
            self.timer.stop()
            self.playing = False
        else:
            if self.slow_motion_checkbox.isChecked():
                self.timer.start(2000 // self.frame_rate)  # Double the interval for half-speed
            else:
                self.timer.start(1000 // self.frame_rate)
            self.playing = True

    def adjust_playback_speed(self):
        """
        Adjust the playback speed based on the checkbox states.
        """
        if self.playing:  # Check if the video is playing
            self.timer.stop()
            if self.slow_motion_checkbox.isChecked():
                self.timer.start(2000 // self.frame_rate)
            else:
                self.timer.start(1000 // self.frame_rate)

    def prev_frame(self):
        """
        Go to the previous frame.
        """
        if self.current_frame > 0:
            self.current_frame -= 1
            self.update_frame()
        elif self.repeat_checkbox.isChecked():
            self.current_frame = len(self.image_files) - 1
            self.update_frame()
        self.video_slider.setValue(self.current_frame)  # Update slider

    def next_frame(self):
        """
        Go to the next frame.
        """
        if self.current_frame < len(self.image_files) - 1:
            self.current_frame += 1
            self.update_frame()
        elif self.repeat_checkbox.isChecked():
            self.current_frame = 0
            self.update_frame()
        self.video_slider.setValue(self.current_frame)  # Update slider

    def update_frame(self):
        """
        Update the frame based on the current frame number.
        """
        if self.current_frame < len(self.image_files) and len(self.image_files) > 0:
            image_file = self.image_files[self.current_frame]
            image = cv2.imread(image_file)
            original_height, original_width, _ = image.shape

            # Define the size to which images should be resized
            target_height = 600
            r = target_height / original_height
            target_width = int(original_width * r)

            # Resize the image to the target size
            image = cv2.resize(image, (target_width, target_height))

            # Calculate the radius and center for the circle based on the resized image dimensions
            self.interest_point = np.array([target_width // 2, target_height])  # Bottom center of the frame
            self.trigger_radius = self.interest_point[1] // self.fa_distance_threshold  # Adjust radius calculation

            # Draw full circle at the bottom if enabled
            if self.draw_half_circle:
                cv2.circle(image, tuple(self.interest_point), int(self.trigger_radius), (0, 255, 0), thickness=2)

            # Draw boundaries if needed
            seq_name = self.img_seq_path.split('/')[-2]
            if seq_name in self.boundaries.keys():
                xl, yl, xr, yr, side = self.boundaries[seq_name]
                # Translate coordinates to the resized image
                xl, yl, xr, yr = int(xl*r), int(yl*r), int(xr*r), int(yr*r)
                cv2.line(image, (xl, yl), (xr, yr), (0, 0, 255), thickness=2)

            # Update max_trace_frames from user input
            try:
                self.max_trace_frames = int(self.trace_frames_input.text())
            except ValueError:
                pass

            # Processing detections and drawing rectangles
            detections = self.gt_coordinates_dict.get(self.current_frame, [])
            tracker_ids = []  # Collect tracker IDs for this frame
            for gt_data in detections:
                if len(gt_data) > 6:
                    if 'OB' in self.active_actions:
                        x, y, w, h, id, ss, sr, fa, g, ob = gt_data
                    else:
                        x, y, w, h, id, ss, sr, fa, g = gt_data
                        ob = 0
                else:
                    x, y, w, h, id = gt_data
                    ss, sr, fa, g, ob = 0, 0, 0, 0, 0

                x_scaled = int(x * target_width / original_width)
                y_scaled = int(y * target_height / original_height)
                w_scaled = int(w * target_width / original_width)
                h_scaled = int(h * target_height / original_height)

                # Check if bounding box touches the circle
                x_min = x_scaled
                x_max = x_scaled + w_scaled
                y_min = y_scaled
                y_max = y_scaled + h_scaled

                center_x = x_scaled + w_scaled // 2
                center_y = y_scaled + h_scaled // 2

                closest_x = max(x_min, min(self.interest_point[0], x_max))
                closest_y = max(y_min, min(self.interest_point[1], y_max))
                distance = np.sqrt((closest_x - self.interest_point[0]) ** 2 + (closest_y - self.interest_point[1]) ** 2)
                touches_circle = distance < self.trigger_radius
                bbox_color = (255, 0, 30) if not touches_circle else (255, 0, 255)
                id_color = (0, 255, 0) if not touches_circle else (255, 0, 255)

                if self.show_only_actions_checkbox.isChecked() and self.has_actions:
                    # Filter bboxes with actions, that is any(ss, sr, fa, g, ob)
                    if not any([ss, sr, fa, g, ob]):
                        continue

                # Get the color state of the tracker ID
                color_state = self.tracker_color_state.get(id, 'green')
                bbox_color = bbox_color if color_state == 'green' else (0, 0, 255)  # Red

                # Draw the bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), bbox_color, 1)  # Thinner bounding box

                # Store trajectory point and draw trace
                if self.trace_checkbox.isChecked():
                    if id not in self.trajectories:
                        self.trajectories[id] = deque(maxlen=self.max_trace_frames)
                    self.trajectories[id].append((center_x, center_y))

                    # Draw trajectory
                    points = list(self.trajectories[id])
                    for i in range(1, len(points)):
                        cv2.line(image, points[i-1], points[i], (0, 255, 255), 1)

                # Draw the point of the center of the bounding box
                cv2.circle(image, (center_x, center_y), 3, (0, 255, 0), -1)

                # ID and action colors
                id_color = id_color if color_state == 'green' else (0, 0, 255)  # Red
                text_y = y_scaled - 10 if y_scaled > 10 else 15
                cv2.putText(image, f"ID: {int(id)}", (x_scaled, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Display actions if the checkbox is checked
                if self.show_actions_checkbox.isChecked() and self.has_actions:
                    actions_display = []
                    if ss: actions_display.append("SS")
                    if sr: actions_display.append("SR")
                    if fa: actions_display.append("FA")
                    if g: actions_display.append(f"G-{int(g)}")
                    if ob: actions_display.append("OB")

                    action_text = ",".join(actions_display)
                    if action_text:
                        text_y = y_max + 15 if y_max + 60 < target_height else target_height - 60
                        cv2.putText(image, f"{action_text}", (x_scaled, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        # action_color = self.action_colors.get(actions_display[0], (255, 255, 255))  # Use the first action's color
                        # cv2.putText(image, f"{action_text}", (x_scaled, y_scaled + h_scaled + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, action_color, 2)

                # Add tracker ID to the list
                tracker_ids.append(id)

            # Update tracker table
            self.tracker_table.setRowCount(len(tracker_ids))
            for row, tracker_id in enumerate(tracker_ids):
                item = QTableWidgetItem(str(int(tracker_id)))  # Convert to int before setting the item text
                color_state = self.tracker_color_state.get(tracker_id, 'green')
                #item.setBackground(Qt.white if color_state == 'green' else Qt.red)  # Set color based on state
                self.tracker_table.setItem(row, 0, item)

            # Convert to Qt format and display
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            qImg = QImage(image.data, target_width, target_height, image.strides[0], QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(qImg))

            # Update UI with frame information
            seconds = self.current_frame / self.frame_rate
            self.table_widget.setItem(0, 0, QTableWidgetItem(self.img_seq_path))
            self.table_widget.setItem(0, 1, QTableWidgetItem(self.gt_file_path))
            self.table_widget.setItem(0, 2, QTableWidgetItem(str(self.frame_rate)))
            self.table_widget.setItem(0, 3, QTableWidgetItem(str(self.current_frame)))
            self.table_widget.setItem(0, 4, QTableWidgetItem(str(len(detections))))
            self.table_widget.setItem(0, 5, QTableWidgetItem(f"{seconds:.2f} seconds"))

        # Handle playback
        if self.playing:
            self.current_frame += 1
            if self.current_frame >= len(self.image_files):
                if self.repeat_checkbox.isChecked():
                    self.current_frame = 0
                else:
                    self.timer.stop()  # Stop at the end of the sequence

        if not self.slider_busy:
            self.video_slider.setValue(self.current_frame)  # Update slider only if not busy

    def slider_changed(self, value):
        """
        Update the frame based on the slider value.
        """
        self.slider_busy = True
        self.current_frame = value
        self.update_frame()
        self.slider_busy = False

    def keyPressEvent(self, event):
        """
        Handle key press events for navigation.
        """
        if event.key() == Qt.Key_Right:
            self.next_frame()
        elif event.key() == Qt.Key_Left:
            self.prev_frame()
        elif event.key() == Qt.Key_Space:
            self.play_pause_video()

    def update_table(self):
        """
        Update the table with the current file paths and frame rate.
        """
        self.table_widget.setItem(0, 0, QTableWidgetItem(self.img_seq_path))
        self.table_widget.setItem(0, 1, QTableWidgetItem(self.gt_file_path))
        self.table_widget.setItem(0, 2, QTableWidgetItem(str(self.frame_rate)))

    def add_annotation(self):
        """
        Add a new annotation to the table.
        """
        row_position = self.annotations_table.rowCount()
        self.annotations_table.insertRow(row_position)
        self.annotations_table.setItem(row_position, 0, QTableWidgetItem(str(self.current_frame)))
        self.annotations_table.setItem(row_position, 1, QTableWidgetItem(str(self.current_frame)))
        self.annotations_table.setItem(row_position, 2, QTableWidgetItem(str(self.current_frame)))
        for col in range(3, self.annotations_table.columnCount()):
            self.annotations_table.setItem(row_position, col, QTableWidgetItem("0"))

    def add_gathering_annotation(self):
        """
        Add a new gathering annotation to the table.
        """
        selected_rows = self.tracker_table.selectionModel().selectedRows()
        if len(selected_rows) < 3:
            QMessageBox.warning(self, "Selection Error", "Please select at least 3 tracker IDs.")
            return

        current_frame = self.current_frame
        for row in selected_rows:
            tracker_id = int(self.tracker_table.item(row.row(), 0).text())
            row_position = self.annotations_table.rowCount()
            self.annotations_table.insertRow(row_position)
            self.annotations_table.setItem(row_position, 0, QTableWidgetItem(str(tracker_id)))
            self.annotations_table.setItem(row_position, 1, QTableWidgetItem(str(current_frame)))
            self.annotations_table.setItem(row_position, 2, QTableWidgetItem(str(current_frame)))
            self.annotations_table.setItem(row_position, 3, QTableWidgetItem("0"))  # SS
            self.annotations_table.setItem(row_position, 4, QTableWidgetItem("0"))  # SR
            self.annotations_table.setItem(row_position, 5, QTableWidgetItem("0"))  # FA
            self.annotations_table.setItem(row_position, 6, QTableWidgetItem("1"))  # Set G column to 1

        self.update_gathering_button_state()

    def export_annotations(self):
        """
        Export the annotations to a CSV file.
        """
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Annotations", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                # Write the headers
                headers = [self.annotations_table.horizontalHeaderItem(i).text() for i in range(self.annotations_table.columnCount())]
                writer.writerow(headers)
                # Write the data
                for row in range(self.annotations_table.rowCount()):
                    row_data = [self.annotations_table.item(row, col).text() if self.annotations_table.item(row, col) else "" for col in range(self.annotations_table.columnCount())]
                    writer.writerow(row_data)

    def import_annotations(self):
        """
        Import annotations from a CSV file.
        """
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Annotations", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, mode='r') as file:
                reader = csv.reader(file)
                headers = next(reader)  # Skip the header row
                self.annotations_table.setRowCount(0)  # Clear existing rows
                for row_data in reader:
                    row_position = self.annotations_table.rowCount()
                    self.annotations_table.insertRow(row_position)
                    for col, data in enumerate(row_data):
                        self.annotations_table.setItem(row_position, col, QTableWidgetItem(data))

    def highlight_tracker_id(self, row, column):
        """
        Highlight the tracker ID row when clicked.
        Args:
            row (int): The row index.
            column (int): The column index.
        """
        # Get the tracker ID item
        item = self.tracker_table.item(row, column)
        if item:
            tracker_id = int(float(item.text()))  # Ensure the tracker ID is an integer
            # Toggle color state
            current_color = self.tracker_color_state.get(tracker_id, 'green')
            new_color = 'red' if current_color == 'green' else 'green'
            self.tracker_color_state[tracker_id] = new_color

            # Update the item background color
            item.setBackground(Qt.red if new_color == 'red' else Qt.white)

            # Update the frame to reflect the new bounding box color
            self.update_frame()
            self.update_gathering_button_state()

    def update_gathering_button_state(self):
        """
        Update the state of the gathering annotation button based on the selected rows.
        """
        selected_rows = self.tracker_table.selectionModel().selectedRows()
        self.btn_add_gathering_annotation.setEnabled(len(selected_rows) >= 3)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    window = VideoAnnotationTool()
    window.show()
    sys.exit(app.exec_())
