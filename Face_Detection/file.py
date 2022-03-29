import cv2
import os
import face_recognition
import pickle
from cv2.cv2 import CAP_DSHOW

known_faces_dir = "known_faces"
video.open(address)
accuracy = 0.6
frame_thikness = 3
font_size = 2
MODEL = "cnn"

print("loading  known faces")
known_faces = []
known_names = []
unknown_faces = []
for name in os.listdir(known_faces_dir):
    for filename in os.listdir(f"{known_faces_dir}/{name}"):
        image = face_recognition.load_image_file(f"{known_faces_dir}/{name}/{filename}")
        encodings = face_recognition.face_encodings(image)[0]
        # encodings = pickle.load(open(f"{name}/{filename}","rb"))
        known_faces.append(encodings)
        known_names.append(name)

print("treating unknow faces")
while True :

    # print(filename)
    image = face_recognition.load_image_file(f"{unknown_faces_dir}/{filename}")
    ret, image = video.read()
    print(video.get(cv2.CAP_PROP_FPS))

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_location, face_encodings in zip(locations, encodings):
        results = face_recognition.compare_faces(known_faces, face_encodings, tolerance=0.54)
        if True in results:
            match = known_names[results.index(True)]
            print("Face Found" f"{match}")

            top_l = (face_location[3], face_location[0])
            bottom_r = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, top_l, bottom_r, color, frame_thikness)

            top_l = (face_location[3], face_location[2])
            bottom_r = (face_location[1], face_location[2] + 22)
            cv2.rectangle(image, top_l, bottom_r, color, cv2.FILLED)
            cv2.putText(image, str(match), (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    cv2.imshow("", image)
    if cv2.waitKey(1)&0xFF == ord("e"):
        break

    # cv2.waitKey(10200)
video.release()
cv2.destroyWindow(filename)