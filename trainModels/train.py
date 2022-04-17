from os.path import exists
from random import shuffle

import numpy as np
from lbphModel import lbph
from getWebCamImages import getImages
import matplotlib.pyplot as plt


def train_lbph(lbph):
    faces, face_labels, name_dict = getImages.getImages()
    lbph.train(faces, np.array(face_labels), name_dict)


def train_cnn(cnn):
    # gather all the images
    faces, face_labels, name_dict = getImages.getImages()

    # set the cnn output layer before we forget
    cnn.set_output_layer(name_dict)

    retrain = 'y'
    if exists('savedModels/cnn.tfl.meta'):
        retrain = input("We found a previously saved model. Proceed with retraining? (type y/n)")

    if retrain == 'y':
        # turn image labels to ohe vector
        ohe_labels = getImages.get_ohe_labels(face_labels, name_dict)

        # match ohe vector with its respective image
        total_face_data = []
        for i in range(len(faces)):
            total_face_data.append([np.array(faces[i]), ohe_labels[i]])
        shuffle(total_face_data)
        train_length = int(.75*len(total_face_data))
        train = total_face_data[:train_length]
        test = total_face_data[train_length:]

        # prepare training and testing data
        x_train = np.array(([i[0] for i in train])).reshape(-1, 120, 120, 1)
        print(x_train.shape)
        y_train = [i[1] for i in train]
        x_test = np.array(([i[0] for i in test])). reshape(-1, 120, 120, 1)
        print(x_test.shape)
        y_test = [i[1] for i in test]
        cnn.train(x_train, y_train, x_test, y_test)
    else:
        cnn.model.load('savedModels/cnn.tfl')