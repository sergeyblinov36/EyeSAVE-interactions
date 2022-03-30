import json

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from Face_Detection.findEncodings import findEncodings
import imutils
import camera

# paths
pathChild = 'Face_Detection/childImgs'

# תשני לשמות של הקולקשן


# arrays of images, names and roles(child or staff member)
c_list = []
images = []
id_list = []
role = []
childrenList = os.listdir(pathChild)
# print(childrenList)
for cl in childrenList:
    # read image from repository
    Img = cv2.imread(f'{pathChild}/{cl}')
    # add image to the array
    images.append(Img)
    # add the name to the array without .jpg
    id_list.append(os.path.splitext(cl)[0])
    role.append("child")
# print(id_list)

# encode the images
encodeListKnown = findEncodings(images)


# print('Encoding Complete')

# cap = cv2.VideoCapture("test2.mp4")
# cap = cv2.VideoCapture(0)
def face_detection(source):
# while True:
    # success, img = cap.read()
    print("1")
    img = camera.getframes(source)
    # resize and change colour of the live stream
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    print("2")
    # locate and encode faces from the frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
    print("3")
    # compare the faces from the frame to the faces from the repositories
    id = 0
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        # if the faces match change the name to upper letters and draw the rectangle around the face
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        if matches[matchIndex]:
            id = id_list[matchIndex].upper()
            cv2.putText(img, id, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            print("4")
            # return id
        else:
            cv2.putText(img, "unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    print(id)
    img = imutils.resize(img, width=600)
    cv2.imshow('debcam', img)
    cv2.waitKey(1)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     # break

    # cap.release()
    # cv2.destroyAllWindows()
    return id
