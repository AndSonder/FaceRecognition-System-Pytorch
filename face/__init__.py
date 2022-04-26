from .mtcnn import Detector
from .mtcnn import draw_bboxes
from .mtcnn import get_max_boxes
from .utils import *
from .facenet import FaceExtractor
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from PIL import Image


class FaceSystem:
    def __init__(self):
        self.face_detector = Detector()
        self.face_extractor = FaceExtractor()

    def face_detect(self, image):
        """
        predict the locations of faces in the image
        """
        boxes, landmarks = self.face_detector.detect_faces(image)
        return boxes

    def save_faces(self, image, boxes, save_path='images'):
        image = np.array(image)
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            face = image[y1: y2, x1: x2, :]
            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            path = os.path.join(save_path, "face_" + str(i) + ".jpg")
            print('[save] ', path)
            cv2.imwrite(path, face)

    def show_face_boxes(self, image, boxes):
        """
        draw face boxes on the image
        """
        result = draw_bboxes(image, boxes)
        show_image(result)

    def video_face_reg(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)
        while True:
            ret, image = cap.read()
            image = Image.fromarray(image, mode='RGB')
            faces = self.face_detect(image)
            image = draw_bboxes(image, faces)
            image = np.array(image)
            image = image.astype(np.uint8)
            cv2.imshow("face", image)
            cv2.waitKey(1)

    def get_face_feature(self, face):
        feature = self.face_extractor.extractor(face)
        return feature

    def feature_compare(self, feature1, feature2):
        dist = np.sqrt(np.sum(np.square(np.abs(feature1 - feature2))))
        return dist

