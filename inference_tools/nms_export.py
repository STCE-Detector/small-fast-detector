import onnx
from onnx import TensorProto

from ultralytics import YOLO

# Load PyTorch model
model = YOLO('./models/8sp2_150.pt', task='detect')

# Export to ONNX
model.export(
    format='onnx',
    imgsz=[640,640],
    batch=1,
    simplify=True,
    dynamic=False,
    opset=12,
    half=False,  # If true, uses FP16 ONNX
)

# LOAD ONNX MODEL AND MANIPULATE
onnx_model = onnx.load('./models/8sp2_150.onnx')
graph = onnx_model.graph
onnx_fpath = './models/8sp2_150_nms.onnx'

# Transpose bbox before NMS
transpose_bboxes_node = onnx.helper.make_node("Transpose", inputs=["/model.28/Mul_2_output_0"], outputs=["bboxes"], perm=(0,2,1))
graph.node.append(transpose_bboxes_node)

# Create constant tensors for NMS
score_threshold = onnx.helper.make_tensor("score_threshold", TensorProto.FLOAT, [1], [0.001])
iou_threshold = onnx.helper.make_tensor("iou_threshold", TensorProto.FLOAT, [1], [0.7])
max_output_boxes_per_class = onnx.helper.make_tensor("max_output_boxes_per_class", TensorProto.INT64, [1], [200])

# NMS node
# create the NMS node
inputs = ['bboxes', '/model.28/Sigmoid_output_0', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold',]
outputs = ["selected_indices"]
nms_node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs,
    ["selected_indices"],
    # center_point_box=1 is very important, PyTorch model's output is
    #  [x_center, y_center, width, height], but default NMS expect
    #  [x_min, y_min, x_max, y_max]
    center_point_box=1,
)
graph.node.append(nms_node)

# Append to the output (now the outputs would be scores, bboxes, selected_indices)
output_value_info = onnx.helper.make_tensor_value_info("selected_indices", TensorProto.INT64, shape=["num_results", 3])
graph.output.append(output_value_info)

# Add to initializers - without this, onnx will not know where these came from, and complain that
# they're neither outputs of other nodes, nor inputs. As initializers, however, they are treated
# as constants needed for the NMS op
graph.initializer.append(score_threshold)
graph.initializer.append(iou_threshold)
graph.initializer.append(max_output_boxes_per_class)

# Remove the unused concat node
last_concat_node = [node for node in onnx_model.graph.node if node.name == "/model.28/Concat_6"][0]
onnx_model.graph.node.remove(last_concat_node)

# Remove the original output0
output0 = [o for o in onnx_model.graph.output if o.name == "output0"][0]
onnx_model.graph.output.remove(output0)

# Output bboxes and scores for downstream tasks
graph.output.append([v for v in onnx_model.graph.value_info if v.name == "/model.28/Mul_2_output_0"][0])
graph.output.append([v for v in onnx_model.graph.value_info if v.name == "/model.28/Sigmoid_output_0"][0])

# TODO: static NMS output
# Create constant tensors for slicing
start = onnx.helper.make_tensor("start", TensorProto.INT64, [1], [0])
end = onnx.helper.make_tensor("end", TensorProto.INT64, [1], [300])     # Max number of detections

slice_node = onnx.helper.make_node("Slice", inputs=["selected_indices", "start", "end", "axes"], outputs=["slice_output"])

# Check that it works and re-save
onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, onnx_fpath)




