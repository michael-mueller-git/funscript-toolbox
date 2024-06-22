import cv2
import os
import numpy as np
import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C


class YOLOv10:
    """YOLOv10 object detection model class for handling inference and visualization."""

    def __init__(self, confidence_thres = 0.4):
        """
        Initializes an instance of the YOLOv10 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
        """
        self.confidence_thres = confidence_thres
        self.session = ort.InferenceSession(
            os.path.join(os.path.dirname(__file__), "best.onnx"), 
            providers=C.get_available_providers()
        )
        
        self.model_inputs = self.session.get_inputs()

        input_shape = self.model_inputs[0].shape

        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        x1, y1, w, h = box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), (255,0,0), 2)

    def preprocess(self, image_path):
        """
        Preprocesses the input image before performing inference.

        Args:
            image_path (str, np.ndarray): path to image or image data

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        if isinstance(image_path, str):
            self.img = cv2.imread(image_path)
        elif isinstance(image_path, np.ndarray):
            self.img = image_path
        else:
            raise ValueError('please make sure the image_path is str or np.ndarray')
        
        self.img_height, self.img_width = self.img.shape[:2]
        
        obj_shape = max(self.img_height, self.img_width)
        self.real_shape = obj_shape
        top_pad = (obj_shape - self.img_height) // 2
        bottom_pad = obj_shape - self.img_height - top_pad
        left_pad = (obj_shape - self.img_width) // 2
        right_pad = obj_shape - self.img_width - left_pad
        
        img = cv2.copyMakeBorder(self.img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[127,127,127])
  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        image_data = (np.array(img)) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        return image_data
        
    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).
            ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                calculated based on the size difference between the two images.
            padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
                rescaling.
            xywh (bool): The box format is xywh or not, default=False.

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)

    def clip_boxes(self, boxes, shape):
        """
        Takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the shape.

        Args:
            boxes (list): the bounding boxes to clip
            shape (tuple): the shape of the image

        Returns:
            (torch.Tensor | numpy.ndarray): Clipped boxes
        """
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        outputs = output[0][0]
        rows = outputs.shape[0]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            max_score = outputs[i,4]
            if max_score >= self.confidence_thres:
                class_id = int(outputs[i,5])
                new_bbox = self.scale_boxes([640,640],(outputs[i,:4]), (self.img_height, self.img_width),xywh=False)
                
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([int(new_bbox[0]),int(new_bbox[1]),int(new_bbox[2]-new_bbox[0]),int(new_bbox[3]-new_bbox[1])])

        detections = []
        for i in range(len(class_ids)):
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            self.draw_detections(input_image, box, score, class_id)
            detections.append({"class": class_id, "score": float(score), "box": box})

        return input_image, detections


    def detect(self, input_image):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
        img_data = self.preprocess(input_image)
        outputs = self.session.run(None, {self.model_inputs[0].name: img_data})

        return self.postprocess(self.img, outputs)


if __name__ == "__main__":
    detector = YOLOv10()
    annotaded_img, detections = detector.detect("test.jpg")
    print(detections)
    cv2.imwrite("result.png", annotaded_img)
