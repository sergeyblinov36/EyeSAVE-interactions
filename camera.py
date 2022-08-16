import cv2


def set_source(source):
    global cap
    cap = []

    for sr in source:
        temp = cv2.VideoCapture(sr)
        cap.append(temp)


def get_source():
    return cap


def getframes(source):
    while True:
        img = get_source()
        success1, img1 = img[0].read()
        success2, img2 = img[1].read()
        if source == 1:
            if success1:
                return img1
            else:
                img[0].release()
                return None
        elif source == 2:
            if success2:

                return img2
            else:
                img[1].release()
                return None

