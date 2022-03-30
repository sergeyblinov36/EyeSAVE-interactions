import cv2


# cap = cv2.VideoCapture(0)

def set_source(source):
    global cap
    cap = []
    iterator = 0
    for sr in source:
        cap.append(cv2.VideoCapture(sr))


def get_source():
    return cap


def getframes(source):
    while True:
        # success, img = cap.read()
        img = get_source()
        success, img1 = img[0].read()
        success, img2 = img[1].read()
        success, img3 = img[2].read()
        if source == 1:
            return img1
        elif source == 2:
            return img2
        else: return img3
        # print(success)
        # resize and change colour of the live stream
        # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        # return img
