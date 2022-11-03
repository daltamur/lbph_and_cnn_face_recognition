import os
import cv2

# get the model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imageNames = os.listdir('jpg')

for image in imageNames:
    img = cv2.imread('jpg/'+image)
    faces = face_cascade.detectMultiScale(img, 1.1, 4)

    if len(faces) == 1:
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cur_path = 'trainImages/Unknown/'+image
        cv2.imwrite(cur_path, grayscale)
