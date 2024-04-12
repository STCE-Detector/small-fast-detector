# convert a model to mix precision or int8 tensorrt engine file

int8 conversion reference [https://github.com/NVIDIA/TensorRT/blob/main/samples/python/detectron2](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/detectron2)

mix precision conversion reference (much better)<br> Tips: the firstly layer and last layer should be FP16, others can be int8. just try it. [https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientdet](https://github.com/NVIDIA/TensorRT/blob/main/samples/python/efficientdet)
