from __future__ import annotations

import random

import cv2
from collections import defaultdict, deque
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import numpy as np
from .position import Position
from .utils import get_data_item, is_data_equal, merge_data, calculate_masks_centroids

from .validators import validate_detections_fields


@dataclass
class Detections:
    """
    The `sv.Detections` class in the Supervision library standardizes results from
    various object detection and segmentation models into a consistent format. This
    class simplifies data manipulation and filtering, providing a uniform API for
    integration with Supervision [trackers](/trackers/), [annotators](/detection/annotators/), and [tools](/detection/tools/line_zone/).

    === "Inference"

        Use [`sv.Detections.from_inference`](/detection/core/#supervision.detection.core.Detections.from_inference)
        method, which accepts model results from both detection and segmentation models.

        ```python
        import cv2
        import supervision as sv
        from inference import get_model

        model = get_model(model_id="yolov8n-640")
        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        results = model.infer(image)[0]
        detections = sv.Detections.from_inference(results)
        ```

    === "Ultralytics"

        Use [`sv.Detections.from_ultralytics`](/detection/core/#supervision.detection.core.Detections.from_ultralytics)
        method, which accepts model results from both detection and segmentation models.

        ```python
        import cv2
        import supervision as sv
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        image = cv2.imread(<SOURCE_IMAGE_PATH>)
        results = model(image)[0]
        detections = sv.Detections.from_ultralytics(results)
        ```

    === "Transformers"

        Use [`sv.Detections.from_transformers`](/detection/core/#supervision.detection.core.Detections.from_transformers)
        method, which accepts model results from both detection and segmentation models.

        ```python
        import torch
        import supervision as sv
        from PIL import Image
        from transformers import DetrImageProcessor, DetrForObjectDetection

        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        image = Image.open(<SOURCE_IMAGE_PATH>)
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        width, height = image.size
        target_size = torch.tensor([[height, width]])
        results = processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_size)[0]
        detections = sv.Detections.from_transformers(
            transformers_results=results,
            id2label=model.config.id2label)
        ```

    Attributes:
        xyxy (np.ndarray): An array of shape `(n, 4)` containing
            the bounding boxes coordinates in format `[x1, y1, x2, y2]`
        mask: (Optional[np.ndarray]): An array of shape
            `(n, H, W)` containing the segmentation masks.
        confidence (Optional[np.ndarray]): An array of shape
            `(n,)` containing the confidence scores of the detections.
        class_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the class ids of the detections.
        tracker_id (Optional[np.ndarray]): An array of shape
            `(n,)` containing the tracker ids of the detections.
        data (Dict[str, Union[np.ndarray, List]]): A dictionary containing additional
            data where each key is a string representing the data type, and the value
            is either a NumPy array or a list of corresponding data.
    """  # noqa: E501 // docs

    xyxy: np.ndarray
    mask: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None
    class_id: Optional[np.ndarray] = None
    tracker_id: Optional[np.ndarray] = None
    data: Dict[str, Union[np.ndarray, List]] = field(default_factory=dict)

    def __post_init__(self):
        validate_detections_fields(
            xyxy=self.xyxy,
            mask=self.mask,
            confidence=self.confidence,
            class_id=self.class_id,
            tracker_id=self.tracker_id,
            data=self.data,
        )

    def __len__(self):
        """
        Returns the number of detections in the Detections object.
        """
        return len(self.xyxy)

    def __iter__(
        self,
    ) -> Iterator[
        Tuple[
            np.ndarray,
            Optional[np.ndarray],
            Optional[float],
            Optional[int],
            Optional[int],
            Dict[str, Union[np.ndarray, List]],
        ]
    ]:
        """
        Iterates over the Detections object and yield a tuple of
        `(xyxy, mask, confidence, class_id, tracker_id, data)` for each detection.
        """
        for i in range(len(self.xyxy)):
            yield (
                self.xyxy[i],
                self.mask[i] if self.mask is not None else None,
                self.confidence[i] if self.confidence is not None else None,
                self.class_id[i] if self.class_id is not None else None,
                self.tracker_id[i] if self.tracker_id is not None else None,
                get_data_item(self.data, i),
            )

    def __eq__(self, other: Detections):
        return all(
            [
                np.array_equal(self.xyxy, other.xyxy),
                np.array_equal(self.mask, other.mask),
                np.array_equal(self.class_id, other.class_id),
                np.array_equal(self.confidence, other.confidence),
                np.array_equal(self.tracker_id, other.tracker_id),
                is_data_equal(self.data, other.data),
            ]
        )

    @classmethod
    def from_ultralytics(cls, ultralytics_results) -> Detections:
        CLASS_NAME_DATA_FIELD = "class_name"
        ORIENTED_BOX_COORDINATES = "xyxyxyxy"
        """
        Creates a `sv.Detections` instance from a
        [YOLOv8](https://github.com/ultralytics/ultralytics) inference result.

        !!! Note

            `from_ultralytics` is compatible with
            [detection](https://docs.ultralytics.com/tasks/detect/),
            [segmentation](https://docs.ultralytics.com/tasks/segment/), and
            [OBB](https://docs.ultralytics.com/tasks/obb/) models.

        Args:
            ultralytics_results (ultralytics.yolo.engine.results.Results):
                The output Results instance from Ultralytics

        Returns:
            Detections: A new Detections object.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')
            results = model(image)[0]
            detections = sv.Detections.from_ultralytics(results)
            ```

        !!! tip

            Class names values can be accessed using `detections["class_name"]`.
        """  # noqa: E501 // docs

        class_id = ultralytics_results.boxes.cls.cpu().numpy().astype(int)
        class_names = np.array([ultralytics_results.names[i] for i in class_id])
        return cls(
            xyxy=ultralytics_results.boxes.xyxy.cpu().numpy(),
            confidence=ultralytics_results.boxes.conf.cpu().numpy(),
            class_id=class_id,
            mask=None,
            tracker_id=ultralytics_results.boxes.id.int().cpu().numpy()
            if ultralytics_results.boxes.id is not None
            else None,
            data={CLASS_NAME_DATA_FIELD: class_names},
        )

    @classmethod
    def empty(cls) -> Detections:
        """
        Create an empty Detections object with no bounding boxes,
            confidences, or class IDs.

        Returns:
            (Detections): An empty Detections object.

        Example:
            ```python
            from supervision import Detections

            empty_detections = Detections.empty()
            ```
        """
        return cls(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.array([], dtype=np.float32),
            class_id=np.array([], dtype=int),
        )

    def is_empty(self) -> bool:
        """
        Returns `True` if the `Detections` object is considered empty.
        """
        empty_detections = Detections.empty()
        empty_detections.data = self.data
        return self == empty_detections

    @classmethod
    def merge(cls, detections_list: List[Detections]) -> Detections:
        """
        Merge a list of Detections objects into a single Detections object.

        This method takes a list of Detections objects and combines their
        respective fields (`xyxy`, `mask`, `confidence`, `class_id`, and `tracker_id`)
        into a single Detections object.

        For example, if merging Detections with 3 and 4 detected objects, this method
        will return a Detections with 7 objects (7 entries in `xyxy`, `mask`, etc).

        !!! Note

            When merging, empty `Detections` objects are ignored.

        Args:
            detections_list (List[Detections]): A list of Detections objects to merge.

        Returns:
            (Detections): A single Detections object containing
                the merged data from the input list.

        Example:
            ```python
            import numpy as np
            import supervision as sv

            detections_1 = sv.Detections(
                xyxy=np.array([[15, 15, 100, 100], [200, 200, 300, 300]]),
                class_id=np.array([1, 2]),
                data={'feature_vector': np.array([0.1, 0.2)])}
             )

            detections_2 = sv.Detections(
                xyxy=np.array([[30, 30, 120, 120]]),
                class_id=np.array([1]),
                data={'feature_vector': [np.array([0.3])]}
             )

            merged_detections = Detections.merge([detections_1, detections_2])

            merged_detections.xyxy
            array([[ 15,  15, 100, 100],
                   [200, 200, 300, 300],
                   [ 30,  30, 120, 120]])

            merged_detections.class_id
            array([1, 2, 1])

            merged_detections.data['feature_vector']
            array([0.1, 0.2, 0.3])
            ```
        """
        detections_list = [
            detections for detections in detections_list if not detections.is_empty()
        ]

        if len(detections_list) == 0:
            return Detections.empty()

        for detections in detections_list:
            validate_detections_fields(
                xyxy=detections.xyxy,
                mask=detections.mask,
                confidence=detections.confidence,
                class_id=detections.class_id,
                tracker_id=detections.tracker_id,
                data=detections.data,
            )

        xyxy = np.vstack([d.xyxy for d in detections_list])

        def stack_or_none(name: str):
            if all(d.__getattribute__(name) is None for d in detections_list):
                return None
            if any(d.__getattribute__(name) is None for d in detections_list):
                raise ValueError(f"All or none of the '{name}' fields must be None")
            return (
                np.vstack([d.__getattribute__(name) for d in detections_list])
                if name == "mask"
                else np.hstack([d.__getattribute__(name) for d in detections_list])
            )

        mask = stack_or_none("mask")
        confidence = stack_or_none("confidence")
        class_id = stack_or_none("class_id")
        tracker_id = stack_or_none("tracker_id")

        data = merge_data([d.data for d in detections_list])

        return cls(
            xyxy=xyxy,
            mask=mask,
            confidence=confidence,
            class_id=class_id,
            tracker_id=tracker_id,
            data=data,
        )

    def get_anchors_coordinates(self, anchor: Position) -> np.ndarray:
        """
        Calculates and returns the coordinates of a specific anchor point
        within the bounding boxes defined by the `xyxy` attribute. The anchor
        point can be any of the predefined positions in the `Position` enum,
        such as `CENTER`, `CENTER_LEFT`, `BOTTOM_RIGHT`, etc.

        Args:
            anchor (Position): An enum specifying the position of the anchor point
                within the bounding box. Supported positions are defined in the
                `Position` enum.

        Returns:
            np.ndarray: An array of shape `(n, 2)`, where `n` is the number of bounding
                boxes. Each row contains the `[x, y]` coordinates of the specified
                anchor point for the corresponding bounding box.

        Raises:
            ValueError: If the provided `anchor` is not supported.
        """
        if anchor == Position.CENTER:
            return np.array(
                [
                    (self.xyxy[:, 0] + self.xyxy[:, 2]) / 2,
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_OF_MASS:
            if self.mask is None:
                raise ValueError(
                    "Cannot use `Position.CENTER_OF_MASS` without a detection mask."
                )
            return calculate_masks_centroids(masks=self.mask)
        elif anchor == Position.CENTER_LEFT:
            return np.array(
                [
                    self.xyxy[:, 0],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.CENTER_RIGHT:
            return np.array(
                [
                    self.xyxy[:, 2],
                    (self.xyxy[:, 1] + self.xyxy[:, 3]) / 2,
                ]
            ).transpose()
        elif anchor == Position.BOTTOM_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 3]]
            ).transpose()
        elif anchor == Position.BOTTOM_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.BOTTOM_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 3]]).transpose()
        elif anchor == Position.TOP_CENTER:
            return np.array(
                [(self.xyxy[:, 0] + self.xyxy[:, 2]) / 2, self.xyxy[:, 1]]
            ).transpose()
        elif anchor == Position.TOP_LEFT:
            return np.array([self.xyxy[:, 0], self.xyxy[:, 1]]).transpose()
        elif anchor == Position.TOP_RIGHT:
            return np.array([self.xyxy[:, 2], self.xyxy[:, 1]]).transpose()

        raise ValueError(f"{anchor} is not supported.")

    def __getitem__(
        self, index: Union[int, slice, List[int], np.ndarray, str]
    ) -> Union[Detections, List, np.ndarray, None]:
        """
        Get a subset of the Detections object or access an item from its data field.

        When provided with an integer, slice, list of integers, or a numpy array, this
        method returns a new Detections object that represents a subset of the original
        detections. When provided with a string, it accesses the corresponding item in
        the data dictionary.

        Args:
            index (Union[int, slice, List[int], np.ndarray, str]): The index, indices,
                or key to access a subset of the Detections or an item from the data.

        Returns:
            Union[Detections, Any]: A subset of the Detections object or an item from
                the data field.

        Example:
            ```python
            import supervision as sv

            detections = sv.Detections()

            first_detection = detections[0]
            first_10_detections = detections[0:10]
            some_detections = detections[[0, 2, 4]]
            class_0_detections = detections[detections.class_id == 0]
            high_confidence_detections = detections[detections.confidence > 0.5]

            feature_vector = detections['feature_vector']
            ```
        """
        if isinstance(index, str):
            return self.data.get(index)
        if isinstance(index, int):
            index = [index]
        return Detections(
            xyxy=self.xyxy[index],
            mask=self.mask[index] if self.mask is not None else None,
            confidence=self.confidence[index] if self.confidence is not None else None,
            class_id=self.class_id[index] if self.class_id is not None else None,
            tracker_id=self.tracker_id[index] if self.tracker_id is not None else None,
            data=get_data_item(self.data, index),
        )

    def __setitem__(self, key: str, value: Union[np.ndarray, List]):
        """
        Set a value in the data dictionary of the Detections object.

        Args:
            key (str): The key in the data dictionary to set.
            value (Union[np.ndarray, List]): The value to set for the key.

        Example:
            ```python
            import cv2
            import supervision as sv
            from ultralytics import YOLO

            image = cv2.imread(<SOURCE_IMAGE_PATH>)
            model = YOLO('yolov8s.pt')

            result = model(image)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections['names'] = [
                 model.model.names[class_id]
                 for class_id
                 in detections.class_id
             ]
            ```
        """
        if not isinstance(value, (np.ndarray, list)):
            raise TypeError("Value must be a np.ndarray or a list")

        if isinstance(value, list):
            value = np.array(value)

        self.data[key] = value

    @property
    def area(self) -> np.ndarray:
        """
        Calculate the area of each detection in the set of object detections.
        If masks field is defined property returns are of each mask.
        If only box is given property return area of each box.

        Returns:
          np.ndarray: An array of floats containing the area of each detection
            in the format of `(area_1, area_2, , area_n)`,
            where n is the number of detections.
        """
        if self.mask is not None:
            return np.array([np.sum(mask) for mask in self.mask])
        else:
            return self.box_area

    @property
    def box_area(self) -> np.ndarray:
        """
        Calculate the area of each bounding box in the set of object detections.

        Returns:
            np.ndarray: An array of floats containing the area of each bounding
                box in the format of `(area_1, area_2, , area_n)`,
                where n is the number of detections.
        """
        return (self.xyxy[:, 3] - self.xyxy[:, 1]) * (self.xyxy[:, 2] - self.xyxy[:, 0])


class ColorPalette:
    """
    A class to generate and manage a color palette for different classes.
    """
    def __init__(self, seed=42):
        random.seed(seed)
        self.colors = {}

    def get_color(self, class_name):
        if class_name not in self.colors:
            self.colors[class_name] = self._generate_color()
        return self.colors[class_name]

    def _generate_color(self):
        return tuple(random.randint(0, 255) for _ in range(3))


class AnnotationDrawer:
    def __init__(self):
        self.color_palette = ColorPalette()
        self.trajectory_manager = TrajectoryManager(maxlen=50, color_palette=self.color_palette)

    def draw_annotations(self, frame, detections, class_names, ar_results, frame_number, fps, save_video=False):
        labels = [f"#{tracker_id} {class_names[class_id]} {confidence:.2f}"
                  for tracker_id, class_id, confidence in
                  zip(detections.tracker_id, detections.class_id, detections.confidence)]

        for det, label, tracker_id, class_id in zip(detections.xyxy, labels, detections.tracker_id, detections.class_id):
            x1, y1, x2, y2 = map(int, det)
            color = self.color_palette.get_color(class_names[class_id])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Calculate the center of the bounding box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            self.trajectory_manager.update(tracker_id, (center_x, center_y), class_names[class_id])

        # Draw all trajectories
        frame = self.trajectory_manager.draw(frame)

        # Add action recognition results if any
        if ar_results:
            self._draw_action_results(frame, ar_results)

        if save_video:
            cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame

    def _draw_action_results(self, frame, ar_results):
        if 'individual' in ar_results and ar_results['individual']:
            for track_id, result in ar_results['individual'].items():
                actions = ', '.join(result.get('actions', []))
                bbox = result.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, actions, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        if 'group' in ar_results and ar_results['group']:
            group_gather = ar_results['group'].get('gather')
            if group_gather:
                for group_id, bbox in group_gather.items():
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.putText(frame, 'Group ' + str(group_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)


class TrajectoryManager:
    """
    A class to manage and draw trajectories of detected objects.
    """
    def __init__(self, maxlen=50, color_palette=None):
        self.trajectories = defaultdict(lambda: deque(maxlen=maxlen))
        self.color_palette = color_palette
        self.class_colors = {}

    def update(self, tracker_id, center, class_name):
        self.trajectories[tracker_id].append(center)
        if class_name not in self.class_colors:
            self.class_colors[class_name] = self.color_palette.get_color(class_name)

    def draw(self, frame):
        for tracker_id, points in self.trajectories.items():
            points_list = list(points)
            class_name = self._get_class_name(tracker_id)
            color = self.class_colors[class_name]
            for i in range(1, len(points_list)):
                cv2.line(frame, points_list[i - 1], points_list[i], color, 2)
                cv2.circle(frame, points_list[i], 3, color, -1)
        return frame

    def _get_class_name(self, tracker_id):
        return list(self.class_colors.keys())[tracker_id % len(self.class_colors)]
