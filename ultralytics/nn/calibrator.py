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

import tensorrt as trt

from ultralytics.utils import LOGGER


# https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Int8/MinMaxCalibrator.html#tensorrt.IInt8MinMaxCalibrator
# class EngineCalibrator(trt.IInt8EntropyCalibrator2):
class EngineCalibrator(trt.IInt8MinMaxCalibrator):
    # class EngineCalibrator(trt.IInt8EntropyCalibrator):
    # class EngineCalibrator(trt.IInt8LegacyCalibrator):
    """Implements the INT8 Entropy Calibrator 2 or IInt8MinMaxCalibrator."""

    def __init__(self, cache_file, args, yolo_model):
        """
        :param cache_file: The location of the cache file.
        """
        super().__init__()
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.batch_generator = None
        from ultralytics import YOLO

        if cache_file is None or not os.path.exists(cache_file):
            self.BasePredictor = YOLO()._smart_load(key="predictor")(overrides=args, _callbacks=None)
            self.BasePredictor.setup_model(yolo_model)
            self.BasePredictor.args.batch = self.BasePredictor.batch = args.calib_batch
            self.BasePredictor.setup_source(args.source)
            self.BasePredictor.fp16 = self.BasePredictor.model.fp16 = False
            self.batch_generator = iter(self.BasePredictor.dataset)

    def get_batch_size(self):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Get the batch size to use for calibration.
        :return: Batch size.
        """
        if self.batch_generator:
            return self.batch_generator.bs
        return 1

    def get_batch(self, names):
        """
        Overrides from trt.IInt8EntropyCalibrator2.

        Get the next batch to use for calibration, as a list of device memory pointers.
        :param names: The names of the inputs, if useful to define the order of inputs.
        :return: A list of int-casted memory pointers.
        """
        if not self.batch_generator:
            return None
        try:
            batchimg = next(self.batch_generator)
            batchimg = self.BasePredictor.preprocess(batchimg[1])
            LOGGER.info("Calibrating image {} / {}".format(self.batch_generator.count, self.batch_generator.ni))
            return [int(batchimg.data_ptr())]
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