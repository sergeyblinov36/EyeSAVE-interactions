import cv2
import imutils
import Facial_expressions.image_manipulation


def get_expression(source, flag):
    frame, pred = Facial_expressions.image_manipulation.get_frame(source)
    frame = imutils.resize(frame, width=900)
    # for visuals only
    # if flag == 0:
    #     cv2.imshow('emo1', frame)
    # else:
    #     cv2.imshow('emo2', frame)
    cv2.waitKey(1)

    return pred

