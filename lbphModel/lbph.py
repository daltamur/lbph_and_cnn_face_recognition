from pathlib import Path

import cv2
import os


class lbphModel:
    def __init__(self):
        self.model = cv2.face_LBPHFaceRecognizer.create()
        self.name_dict = dict()

    def train(self, faces, faceLabels, name_dict):
        self.name_dict = name_dict
        self.model.train(faces, faceLabels)
        print("model trained")

    def predictLabel(self, face):
        return self.model.predict(face)
