from cnnModel import model
from getWebCamImages import take_pictures
from lbphModel import lbph
from trainModels import train
from runDetector import detectFace

lbph_model = lbph.lbphModel()
cnn_model = model.cnnModel()


def run():
    cur_command = input("Enter Command (type 'help' to see list of commands): ")

    while cur_command != 'exit':
        if cur_command == 'take pictures':
            take_pictures.takeImages()
            print("Pictures added to database!")

        elif cur_command == 'train lbph':
            train.train_lbph(lbph_model)
            print("LBPH model trained successfully!")

        elif cur_command == 'train cnn':
            train.train_cnn(cnn_model)
            print("CNN model trained successfully!")

        elif cur_command == 'run predictor lbph':
            detectFace.runDetector(lbph_model.name_dict, lbph_model, 'lbph')
            print()
            print("Predictor closed successfully")

        elif cur_command == 'run predictor cnn':
            detectFace.runDetector(cnn_model.labels, cnn_model, 'cnn')
            print()
            print("Predictor closed successfully")

        elif cur_command == 'help':
            print()
            print("Commands:")
            print("take pictures: take pictures used for training models")
            print('train lbph: train the local binary pattern histogram model')
            print('train cnn: train convolutional neural network model (Will take a lot longer, if a model has already been trained previously, you '
                  'will be given the option to load that model)')
            print('run predictor: run the real-time prediction algorithm')
            print()

        else:
            print("No such command!")

        cur_command = input("Enter Command (type 'help' to see list of commands): ")


run()
