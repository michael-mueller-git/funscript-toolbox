import onnxruntime
import os
import cv2

import numpy as np

from onnxruntime.capi import _pybind_state as C


class YoloFace:

    def __init__(self, path=None, detect_threshold=0.5):
        self.session = onnxruntime.InferenceSession(
            os.path.join(os.path.dirname(__file__), "yoloface_8n.onnx") if path is None else path, 
            providers=C.get_available_providers()
        )

        self.model_inputs = self.session.get_inputs()

        input_shape = self.model_inputs[0].shape

        self.input_width = input_shape[2]
        self.input_height = input_shape[3]
        self.detect_threshold = detect_threshold


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
    
    def draw_detections(self, img, boxes):

        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            boxes: Detected bounding boxes.

        Returns:
            None
        """
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 2)

    def postprocess(self, input_image, detections):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        bounding_box_list = []
        face_landmark_5_list = []
        score_list = []

        ratio_height = self.img_height / self.input_width
        ratio_width = self.img_width / self.input_height

        detections = np.squeeze(detections).T
        bounding_box_raw, score_raw, face_landmark_5_raw = np.split(detections, [ 4, 5 ], axis = 1)
        keep_indices = np.where(score_raw > self.detect_threshold)[0]
        if keep_indices.any():
            bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
            for bounding_box in bounding_box_raw:
                bounding_box_list.append(np.array(
                [
                    round((bounding_box[0] - bounding_box[2] / 2) * ratio_width),
                    round((bounding_box[1] - bounding_box[3] / 2) * ratio_height),
                    round((bounding_box[0] + bounding_box[2] / 2) * ratio_width),
                    round((bounding_box[1] + bounding_box[3] / 2) * ratio_height)
                ]))
            face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
            face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
            for face_landmark_5 in face_landmark_5_raw:
                face_landmark_5_list.append(np.array(face_landmark_5.reshape(-1, 3)[:, :2]))
            score_list = score_raw.ravel().tolist()

        indices = cv2.dnn.NMSBoxes(bounding_box_list, score_list, 0.25, 0.45)

        detections = [bounding_box_list[i] for i in indices]

        self.draw_detections(input_image, detections)

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
    face_detector = YoloFace()
    image, detections = face_detector.detect("./mpv-shot0002.jpg")
    print(detections)
    cv2.imwrite("./result.png", image)
