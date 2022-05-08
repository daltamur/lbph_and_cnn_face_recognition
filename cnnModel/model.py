from os.path import exists

import numpy as np
import tensorflow as tf
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class cnnModel:
    def __init__(self):
        self.model = None
        cnn = input_data(shape=[150, 150, 3])

        #32 filters and stride is 5, the filter will move 5 pixels at a time
        cnn = conv_2d(cnn, 32, filter_size=5, activation='relu')
        cnn = max_pool_2d(cnn, 5)
        cnn = conv_2d(cnn, 64, 5, activation='relu')
        cnn = max_pool_2d(cnn, 5)
        cnn = conv_2d(cnn, 128, 5, activation='relu')
        cnn = max_pool_2d(cnn, 5)
        cnn = conv_2d(cnn, 256, 5, activation='relu')

        #input layer
        cnn = fully_connected(cnn, 1024, activation='relu')
        self.cnn = dropout(cnn, 0.5)
        # input layer
        cnn = fully_connected(cnn, 256, activation='relu')
        self.cnn = dropout(cnn, 0.5)

    def set_output_layer(self, labels):
        # output layer
        self.labels = labels
        self.cnn = fully_connected(self.cnn, len(labels.keys()), activation='softmax')
        self.cnn = regression(self.cnn, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

        # final model initializer
        self.model = tflearn.DNN(self.cnn, tensorboard_verbose=1)

    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, n_epoch=5, validation_set=(x_test, y_test), show_metric=True)
        self.model.save('savedModels/cnn.tfl')

    def predict(self, current_face):
        return self.model.predict(np.array(current_face).reshape(-1, 150, 150, 3))