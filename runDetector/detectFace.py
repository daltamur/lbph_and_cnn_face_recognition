import cv2
import numpy as np
from lbphModel import lbph

# get the model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict(img, face_dict, recognizer, recognizer_type):
    img = cv2.normalize(img, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)

    # face detection only works on grayscaled images, grayscale the image
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # get the faces that the model detects
    faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)
    if len(faces) != 0:
        for face in faces:
            (x, y, w, h) = face

            if recognizer_type == 'lbph':
                # resize image for lbp analyzer
                scaledFace = cv2.resize(grayscale[y:y + h, x:x + h], (150, 150), interpolation=cv2.INTER_AREA)
                face_data = recognizer.predictLabel(scaledFace)
                face_label = face_dict[face_data[0]]
                confidence = face_data[1]
                confidence = round(confidence, 2)
                if confidence > 100:
                    face_label = face_label+"?"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, face_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.putText(img, 'Confidence: ' + str(confidence), (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                            (0, 255, 0), 2)
            else:
                # resize image for lbp analyzer
                scaledFace = cv2.resize(rgb[y:y + h, x:x + h], (150, 150), interpolation=cv2.INTER_AREA)
                face_data = recognizer.predict(scaledFace)[0].tolist()
                predicted_face_val = max(face_data)
                face_label = face_dict[face_data.index(predicted_face_val)]
                predicted_face_val = round(predicted_face_val, 2)
                if predicted_face_val < .75:
                    face_label = face_label+"?"
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(img, face_label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                cv2.putText(img, str(predicted_face_val), (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('img', img)



def runDetector(face_dict, recognizer, recognizer_type):
    # create an object to capture video
    capture = cv2.VideoCapture(0)

    while True:
        # get an image from the video capture
        _, img = capture.read()
        predict(img, face_dict, recognizer, recognizer_type)
        # breaks loop if esc key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break

    # release the capture object after program is done
    capture.release()

    #destroy window
    cv2.destroyAllWindows()



