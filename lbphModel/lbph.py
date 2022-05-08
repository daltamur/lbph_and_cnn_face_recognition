from pathlib import Path

import cv2
import os
from tqdm import tqdm


class lbphModel:
    def __init__(self):
        self.model = cv2.face_LBPHFaceRecognizer.create()
        self.name_dict = dict()

    def train(self, faces, faceLabels, name_dict):
        self.name_dict = name_dict
        self.model.train(faces, faceLabels)
        print("model trained")

    #   now run a test to see how accurate it is.
        correct_amount = 0
        # get the model
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        name_dict = dict()
        cur_face_path = 'Dom_LBPH_Test'
        imageNames = os.listdir(cur_face_path)
        print("Getting Images from " + cur_face_path)
        for imageName in tqdm(imageNames):
            imagePath = 'Dom_LBPH_Test/'+imageName
            image = cv2.imread(imagePath, 0)
            # get the faces that the model detects
            faces = face_cascade.detectMultiScale(image, 1.1, 4)

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                # resize image for lbp analyzer
                scaledFace = cv2.resize(image[y:y + h, x:x + h], (150, 150), interpolation=cv2.INTER_AREA)
                predicted = self.model.predict(scaledFace)
                if(self.name_dict[predicted[0]] == "Dom"):
                    correct_amount += 1

        correct_amount = correct_amount/1000
        print("Accuracy: "+ str(correct_amount))
    def predictLabel(self, face):
        return self.model.predict(face)
