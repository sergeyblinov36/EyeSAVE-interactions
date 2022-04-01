import cv2
import datetime
import imutils
import numpy as np
from Facial_expressions.main import get_expression
import camera
from Distance.centroidtracker import CentroidTracker
from itertools import combinations
import math
from Face_Detection.Face_detection import face_detection
# from camera import getframes
from interaction.Interaction import save_interaction
from interaction.Interaction import determine_interaction
import time

# protopath = "../EyeSAVE-master/interactionsCopy/Distance/MobileNetSSD_deploy.prototxt"
# modelpath = "../EyeSAVE-master/interactionsCopy/Distance/MobileNetSSD_deploy.caffemodel"
protopath = "Distance/MobileNetSSD_deploy.prototxt"
modelpath = "Distance/MobileNetSSD_deploy.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath)
# detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
# detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

global children
children = []
global object_id_list
object_id_list = []
global dtime
dtime = dict()
global dwell_time
dwell_time = dict()
global timer
timer = dict()


def non_max_suppression_fast(boxes, overlapThresh):
    try:
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    except Exception as e:
        print("Exception occurred in non_max_suppression : {}".format(e))


def get_distance():
    # cap = camera.source('test2.mp4')
    global children, timer, slept, finishTime
    finishTime = datetime.datetime.now().replace(hour=21, minute=45, second=0, microsecond=0)
    count = 0
    print("dis")
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0
    interaction = 0
    status = 0
    startTime = ""
    slept = 0

    while True:

        frame = camera.getframes(1)
        frame = imutils.resize(frame, width=600)
        total_frames = total_frames + 1

        (H, W) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)

        detector.setInput(blob)
        person_detections = detector.forward()
        rects = []
        for i in np.arange(0, person_detections.shape[2]):
            confidence = person_detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(person_detections[0, 0, i, 1])

                if CLASSES[idx] != "person":
                    continue

                person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = person_box.astype("int")
                rects.append(person_box)

        boundingboxes = np.array(rects)
        boundingboxes = boundingboxes.astype(int)
        rects = non_max_suppression_fast(boundingboxes, 0.3)
        centroid_dict = dict()
        objects = tracker.update(rects)
        for (objectId, bbox) in objects.items():
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            if objectId not in object_id_list:
                object_id_list.append(objectId)
                now = datetime.datetime.now()
                dtime[objectId] = now
                dwell_time[objectId] = 0
                startTime = now.strftime("%H:%M:%S")
            else:
                curr_time = datetime.datetime.now()
                old_time = dtime[objectId]
                time_diff = curr_time - old_time
                dtime[objectId] = datetime.datetime.now()
                sec = time_diff.total_seconds()
                dwell_time[objectId] += sec

            centroid_dict[objectId] = (cX, cY, x1, y1, x2, y2)

            # text = "ID: {}".format(objectId)
            # cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        red_zone_list = []
        for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
            dx, dy = p1[0] - p2[0], p1[1] - p2[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance < 400.0:
                # temp = []
                status = 1
                print("id1 = {id1}".format(id1=id1))
                print("id2 = {id2}".format(id2=id2))
                print("childid1 = {childid1}".format(childid1=children))

                while count < 2:
                    temp1 = face_detection(1, 0)
                    print("8888888888888888888")
                    temp2 = face_detection(2, 1)
                    if not children or temp1 != children[0]:
                        if temp1 != 0:
                            children.append(temp1)
                            print(children[0])
                            print("---------")
                            print(temp1)
                            count = count + 1
                    elif not children or temp2 != children[0]:
                        if temp2 != 0:
                            children.append(temp2)
                            print(children[0])
                            print("---------")
                            print(temp2)
                            count = count + 1
                prediction = []
                # prediction1 = get_expression(2)
                # prediction2 = get_expression(3)
                prediction1 = get_expression(1, 0)
                prediction2 = get_expression(2, 1)
                prediction.append(prediction1)
                prediction.append(prediction2)
                interaction_value = determine_interaction(prediction)
                interaction += interaction_value
                # if prediction1 == "empty" or prediction1 == "Neutral" or prediction2 == "empty" or prediction2 == "Neutral":
                #     interaction = interaction
                # elif prediction1 == "Happy" or prediction1 == "Happy":
                #     interaction = interaction + 1
                # else:
                #     interaction = interaction - 1
                print(interaction)
                print("+++++++++")
                print(dwell_time)
                # print(distance)
                if id1 not in red_zone_list:
                    red_zone_list.append(id1)
                if id2 not in red_zone_list:
                    red_zone_list.append(id2)

        for id, box in centroid_dict.items():
            if id in red_zone_list:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (box[2], box[3]), (box[4], box[5]), (0, 255, 0), 2)
        if len(centroid_dict) < 2:
            if status == 1:
                if slept == 0:
                    print("going to sleep")
                    time.sleep(5)
                    slept = 1
                else:
                    now = datetime.datetime.now()
                    time_diff = now - dtime[objectId]
                    duration = time_diff.total_seconds()
                    int(duration)
                    save_interaction(children, startTime, interaction, duration)
                    children.clear()
                    interaction = 0
                    count = 0
                    status = 0
                    slept = 0
                # if objectId not in timer:
                #     print("no one in the frame")
                #     timer[objectId] = 0
                # else:
                #     curr_time = datetime.datetime.now()
                #     old_time = dtime[objectId]
                #     time_diff = curr_time - old_time
                #     dtime[objectId] = datetime.datetime.now()
                #     sec = time_diff.total_seconds()
                #     timer[objectId] += sec
                #     print(timer[objectId])
                #     if timer[objectId] > 5:
                #         status = 0
                #         count = 0
                #         print(children[0])
                #         print(children[1])
                #         print(dwell_time[objectId])
                #         print(interaction)
                #         print("end of interaction")

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        cv2.imshow("distance", frame)
        key = cv2.waitKey(1)
        if key == ord('q') or finishTime < datetime.datetime.now():
            break

    cv2.destroyAllWindows()
