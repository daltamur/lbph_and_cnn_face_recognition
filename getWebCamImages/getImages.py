import os
import cv2
import numpy as np
from tqdm import tqdm


def getImages():
    allFaces = []
    faceLabels = []
    knownFaceDirs = os.listdir('trainImages/')

    # get the model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    name_dict = dict()

    # go through the pictures of faces abd aggregate them
    i = 0
    for face in knownFaceDirs:
        cur_face_path = 'trainImages/' + face
        imageNames = os.listdir(cur_face_path)
        print("Getting Images from " + cur_face_path)
        for imageName in tqdm(imageNames):
            imagePath = 'trainImages/' + face + '/' + imageName
            image = cv2.imread(imagePath, 0)
            # get the faces that the model detects

            faces = face_cascade.detectMultiScale(image, 1.1, 4)

            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                # resize image for lbp analyzer
                scaledFace = cv2.resize(image[y:y + h, x:x + h], (120, 120), interpolation=cv2.INTER_AREA)
                allFaces.append(scaledFace)
                if i not in name_dict:
                    name_dict[i] = face
                faceLabels.append(i)
        i += 1

    print(knownFaceDirs)
    return allFaces, faceLabels, name_dict

def get_ohe_labels(face_labels, name_dict):
    # make a one hot encoding of all the possible labels and append them to a list that is 1-1 with the images
    faceLabels_ohe = list()
    # tqdm will just tell us our progress
    for faceLabel in tqdm(face_labels):
        cur_ohe = list()
        for i in range(len(name_dict.keys())):
            if i == faceLabel:
                cur_ohe.append(1)
            else:
                cur_ohe.append(0)
        faceLabels_ohe.append(np.array(cur_ohe))

    return faceLabels_ohe
