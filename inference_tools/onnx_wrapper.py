import numpy as np
import cv2
import onnxruntime as ort


class YOLOv8OnnxRT:

    def __init__(self, onnx_model_path, input_image_path, confidence_threshold, iou_threshold):
        """
        Initialize the YOLOv8OnnxRuntime class

        Args:
            onnx_model_path (str): Path to the onnx model
            input_image_path (str): Path to the input image
            confidence_threshold (float): Confidence threshold
            iou_threshold (float): IoU threshold
        """
        self.onnx_model_path = onnx_model_path
        self.input_image_path = input_image_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        # Class names are hardcoded for now
        # self.class_names = yaml_load(check_yaml('coco128.yaml'))['names']
        self.class_names = {
            0: 'person',
            1: 'car',
            2: 'truck',
            3: 'uav',
            4: 'airplane',
            5: 'boat',
        }

        # Generate the colors for the bounding boxes of every class
        self.color_palette = np.random.uniform(0, 255, size=(len(self.class_names), 3))

    def preprocess(self):
        """
        Preprocess the input image

        Returns:
            image_data (np.array): Preprocessed image data ready for inference
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image_path)

        # Get the image shape
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing every pixel by 255
        image_data = np.array(img) / 255.0

        # Transpose image data to have the channels in the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))

        # Expand the image dimensions to have the batch size in the first dimension
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the image data
        return image_data

    def draw_detections(self, img, box, score, class_id):
        # Get the bounding box coordinates
        x1, y1, w, h = box

        # Retrieve the color for the current class
        color = self.color_palette[class_id]

        # Draw the bounding box rectangle
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f'{self.class_names[class_id]}: {score:.2f}'

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                      cv2.FILLED)

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def postprocess(self, input_image, output):
        """
        Postprocess the outputs of the model to obtain the output image with drawn detections

        Args:
            input_image (np.array): Input image
            output (np.array): Output of the model

        Returns:
            input_image (np.array): Output image with drawn detections
        """
        # Transpose and squeeze the outputs to remove batch size and match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Create empty lists to store the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding boxes
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over every row in the outputs array
        for i in range(rows):
            # Extract the class scores for the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # Check if the maximum score is greater than the confidence threshold
            if max_score > self.confidence_threshold:
                # Get the index of the class with the maximum score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates for the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate scaled bounding box coordinates
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class id, confidence and bounding box coordinates to their respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Perform non-maximum suppression to remove overlapping bounding boxes with lower confidences
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.iou_threshold)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score and class id for the current index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]

            # Draw the bounding box on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the input image with drawn detections
        return input_image

    def main(self):
        """
        Main function of the YOLOv8OnnxRuntime class, performs inference using the onnx model and returns the output
        image with drawn detections

        Returns:
            output_image (np.array): Output image with drawn detections
        """

        # Create an inference session using the onnx model and specify execution providers
        session = ort.InferenceSession(self.onnx_model_path, providers=['CPUExecutionProvider'])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the input image
        image_data = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = session.run(None, {model_inputs[0].name: image_data})

        # Perform postprocessing on the outputs to obtain output image
        return self.postprocess(self.img, outputs)


if __name__ == '__main__':
    model = YOLOv8OnnxRT(
        onnx_model_path='/Users/inaki-eab/Desktop/small-fast-detector/inference_tools/models/yolov8n.onnx',
        input_image_path='/Users/inaki-eab/Desktop/datasets/coco128/images/train2017',
        confidence_threshold=0.25,
        iou_threshold=0.7
    )