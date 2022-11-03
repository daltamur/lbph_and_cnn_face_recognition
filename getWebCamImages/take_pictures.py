import os
import cv2

# get the model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def takeImages():
    # get Face Name
    name = input('Enter name: ')

    if not os.path.isdir('trainImagesCNN/' + name):
        print(False)
        os.mkdir('trainImagesCNN/' + name)

    if not os.path.isdir('trainImagesLBPH/' + name):
        print(False)
        os.mkdir('trainImagesLBPH/' + name)

    cap = cv2.VideoCapture(0)

    i = 0
    while not cap.isOpened():
        print("waiting on capture")
    while True:
        cur_path = 'trainImagesCNN/' + name + '/' + str(i) + '.jpg'
        cur_path_lbph = 'trainImagesLBPH/' + name + '/' + str(i) + '.jpg'
        result, image = cap.read()
        image = cv2.normalize(image, None, alpha=0, beta=200, norm_type=cv2.NORM_MINMAX)
        if result:
            # face detection only works on grayscaled images, grayscale the image
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # get the faces that the model detects
            faces = face_cascade.detectMultiScale(grayscale, 1.1, 4)

            if len(faces) == 1:
                cv2.imwrite(cur_path, image)
                cv2.imwrite(cur_path_lbph, grayscale)
                i += 1
                (x, y, w, h) = faces[0]
                cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Training Image', image)
            if i == 1000:
                break
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        else:
            print("messed up")

    cv2.destroyAllWindows()
