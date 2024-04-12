#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# https://github.com/NVIDIA/TensorRT/blob/release/10.0/samples/python/efficientdet/build_engine.py

import os
import sys
import tensorrt as trt
from ultralytics.utils import LOGGER

sys.path.insert(1, os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

from ultralytics.engine.tensorrt_int8.image_batcher import ImageBatcher

log = LOGGER


# class EngineCalibrator(trt.IInt8EntropyCalibrator2):
class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    # class EngineCalibrator(trt.IInt8EntropyCalibrator):
    # class EngineCalibrator(trt.IInt8LegacyCalibrator):
    """Implements the INT8 Entropy Calibrator 2 or IInt8MinMaxCalibrator."""

    def __init__(self, cache_file, device="cuda:0"):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        self.cache_file = cache_file
        self.image_batcher = None
        self.batch_allocation = None
        self.batch_generator = None
        self.device = device

    def set_image_batcher(self, image_batcher: ImageBatcher):
        """
        Define the image batcher to use, if any.

        If using only the cache file, an image batcher doesn't need to be defined.
        :param image_batcher: The ImageBatcher object
        """
        self.image_batcher = image_batcher
        self.batch_generator = self.image_batcher.get_batch()

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.image_batcher:
            return self.image_batcher.batch_size
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.image_batcher:
            return None
        try:
            batch = next(self.batch_generator)
            LOGGER.info(
                "Calibrating image {} / {}".format(self.image_batcher.image_index, self.image_batcher.num_images)
            )
            # common.memcpy_host_to_device(self.batch_allocation, np.ascontiguousarray(batch))
            # return [int(self.batch_allocation)]
            return [int(batch.data_ptr())]
            # return [batch.data_ptr()]
        except StopIteration:
            LOGGER.info("Finished calibration batches")
            return None

    def read_calibration_cache(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Read the calibration cache file stored on disk, if it exists.
        :return: The contents of the cache file, if any.
        """
        if self.cache_file is not None and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                LOGGER.info("Using calibration cache file: {}".format(self.cache_file))
                return f.read()

    def write_calibration_cache(self, cache):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Store the calibration cache to a file on disk.
        :param cache: The contents of the calibration cache to store.
        """
        if self.cache_file is None:
            return
        with open(self.cache_file, "wb") as f:
            LOGGER.info("Writing calibration cache data to: {}".format(self.cache_file))
            f.write(cache)
