import cv2


# cap = cv2.VideoCapture(0)
import numpy as np


def set_source(source):
    global cap
    cap = []

    for sr in source:
        cap.append(cv2.VideoCapture(sr))


def get_source():
    return cap


def getframes(source):
    while True:
        # success, img = cap.read()
        emptyFrame = cv2.imread("no_image.jpg")
        img = get_source()
        success1, img1 = img[0].read()
        success2, img2 = img[1].read()
        # success, img3 = img[2].read()
        if source == 1:
            if success1:
                return img1
            else:
                print("fail")
                img[0].release()
                return emptyFrame
        elif source == 2:
            if success2:

                return img2
            else:
                print("fail")
                return emptyFrame
        # else: return img3
        # print(success)
        # resize and change colour of the live stream
        # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        # return img
