import cv2
from Facial_expressions.model import FacialExpressionModel
import numpy as np
import camera

facec = cv2.CascadeClassifier('Facial_expressions/haarcascade_frontalface_default.xml')
model = FacialExpressionModel("Facial_expressions/model.json", "Facial_expressions/model_weights.h5")
font = cv2.FONT_HERSHEY_SIMPLEX
global sad_counter
sad_counter = 0


def get_frame(source):
    fr = camera.getframes(source)
    gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_fr, 1.3, 5)
    pred = "empty"
    global sad_counter
    for (x, y, w, h) in faces:
        fc = gray_fr[y:y + h, x:x + w]
        roi = cv2.resize(fc, (48, 48))
        pred = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
        if pred == "Disgust" or pred == "Surprise":
            pred = "Neutral"
        if pred == "Sad":
            if sad_counter == 3:
                pred = "Neutral"
                sad_counter = 0
            else:
                sad_counter += 1
        # for visuals only
        # cv2.putText(fr, pred, (x, y), font, 5, (255, 255, 0),10)
        # cv2.rectangle(fr, (x, y), (x + w, y + h), (255, 0, 0), 5)

    return fr, pred
