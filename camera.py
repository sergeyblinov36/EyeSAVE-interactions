import cv2


# cap = cv2.VideoCapture(0)

def set_source(source):
    global cap
    cap = cv2.VideoCapture(source)
    return cv2.VideoCapture(source)

def get_source():
    return cap


def getframes():
    while True:
        # success, img = cap.read()
        success, img = get_source().read()
        # print(success)
        # resize and change colour of the live stream
        # imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        # imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        return img