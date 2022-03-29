import cv2
from Facial_expressions.model import FacialExpressionModel
import numpy as np
import camera

facec = cv2.CascadeClassifier('Facial_expressions/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("Facial_expressions/model.json", "Facial_expressions/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX


def get_frame():
    # camera.set_source("test2.mp4")
    fr = camera.getframes()
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    # if fr.any():
    #     print("check")
    #     print(fr.any())
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    pred = "empty"
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y + h, x:x + w]
        empty = "empty"
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        print(type(pred))
        print(pred)
        cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
        cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return fr,pred
