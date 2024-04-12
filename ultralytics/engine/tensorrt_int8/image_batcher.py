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
# https://github.com/NVIDIA/TensorRT/blob/release/10.0/samples/python/efficientnet/build_engine.py
import torch
import cv2
import os
import sys
import numpy as np
from PIL import Image
from ultralytics.data.augment import LetterBox


class ImageBatcher:
    """Creates batches of pre-processed images."""

    def __init__(
        self,
        input,
        shape,
        dtype,
        max_num_images=None,
        exact_batches=False,
        config_file=None,
        shuffle_files=True,
        device="cuda:0",
    ):
        """
        :param input: The input directory to read images from.
        :param shape: The tensor shape of the batch to prepare, either in NCHW or NHWC format.
        :param dtype: The (numpy) datatype to cast the batched data to.
        :param max_num_images: The maximum number of images to read from the directory.
        :param exact_batches: This defines how to handle a number of images that is not an exact multiple of the batch
        size. If false, it will pad the final batch with zeros to reach the batch size. If true, it will *remove* the
        last few images in excess of a batch size multiple, to guarantee batches are exact (useful for calibration).
        :param config_file: The path pointing to the Detectron 2 yaml file which describes the model.
        """
        # Find images in the given input path.
        input = os.path.realpath(input)
        self.images = []

        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".ppm"]

        def is_image(path):
            return os.path.isfile(path) and os.path.splitext(path)[1].lower() in extensions

        if os.path.isdir(input):
            self.images = [os.path.join(input, f) for f in os.listdir(input) if is_image(os.path.join(input, f))]
            self.images.sort()
            if shuffle_files:
                np.random.seed(999999999)
                np.random.shuffle(self.images)
        elif os.path.isfile(input):
            if is_image(input):
                self.images.append(input)
        self.num_images = len(self.images)
        if self.num_images < 1:
            print("No valid {} images found in {}".format("/".join(extensions), input))
            sys.exit(1)

        # Handle Tensor Shape.
        if dtype == np.float32:
            self.dtype = torch.float32
        elif dtype == np.float16:
            self.dtype = torch.float16
        elif dtype == np.int8:
            self.dtype = torch.int8
        self.shape = shape
        assert len(self.shape) == 4
        self.batch_size = shape[0]
        assert self.batch_size > 0
        self.format = None
        self.width = -1
        self.height = -1
        if self.shape[1] == 3:
            self.format = "NCHW"
            self.height = self.shape[2]
            self.width = self.shape[3]
        elif self.shape[3] == 3:
            self.format = "NHWC"
            self.height = self.shape[1]
            self.width = self.shape[2]
        assert all([self.format, self.width > 0, self.height > 0])

        # Adapt the number of images as needed.
        if max_num_images and 0 < max_num_images < len(self.images):
            self.num_images = max_num_images
        if exact_batches:
            self.num_images = self.batch_size * (self.num_images // self.batch_size)
        if self.num_images < 1:
            print("Not enough images to create batches")
            sys.exit(1)
        self.images = self.images[0 : self.num_images]

        # Subdivide the list of images into batches.
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = []
        for i in range(self.num_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_images)
            self.batches.append(self.images[start:end])

        # Indices.
        self.image_index = 0
        self.batch_index = 0
        self.newshape = [self.height, self.width]
        self.device = device
        self.LetterBox = LetterBox(self.newshape, scaleup=False)

    def preprocess_image(self, image_path):
        """
        The image preprocessor loads an image from disk and prepares it as needed for batching. This includes padding,
        resizing, normalization, data type casting, and transposing.

        This Image Batcher implements one algorithm for now:
        * Resizes and pads the image to fit the input size.
        :param image_path: The path to the image on disk to load.
        :return: Two values: A numpy array holding the image sample, ready to be contacatenated into the rest of the
        batch, and the resize scale used, if any.
        """
        image = Image.open(image_path)
        image = image.convert(mode="RGB")
        # Pad with mean values of COCO dataset, since padding is applied before actual model's
        # preprocessor steps (Sub, Div ops), we need to pad with mean values in order to reverse
        # the effects of Sub and Div, so that padding after model's preprocessor will be with actual 0s.
        image = np.asarray(image, dtype=np.float32).copy()
        image = self.LetterBox(labels=None, image=image)
        # cv2.imwrite(r'E:\work\codeRepo\deploy\jz\image_batcher\%d.jpg'%np.random.randint(999999999), image)
        # Change HWC -> CHW.
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = torch.from_numpy(image)
        image = torch.tensor(image, dtype=self.dtype)
        return image

    def get_batch(self):
        """
        Retrieve the batches.

        This is a generator object, so you can use it within a loop as:
        for batch, images in batcher.get_batch():
           ...
        Or outside of a batch with the next() function.
        :return: A generator yielding three items per iteration: a numpy array holding a batch of images, the list of
        paths to the images loaded within this batch, and the list of resize scales for each image in the batch.
        """
        for i, batch_images in enumerate(self.batches):
            batch_data = torch.zeros(tuple(self.shape), dtype=self.dtype, device=self.device)
            for j, image in enumerate(self.batches[self.batch_index]):
                self.image_index += 1
                batch_data[j] = self.preprocess_image(image)
            self.batch_index += 1
            yield batch_data
