from flask import Flask, render_template, Response
import Facial_expressions.image_manipulation
import camera
import cv2
import imutils


# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# def gen(camera):
#     while True:
#         frame = camera.get_frame()
#
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
def get_expression():
    # camera.source("test2.mp4")
    # print("face")
    # while True:
    frame, pred = Facial_expressions.image_manipulation.get_frame()
    frame = imutils.resize(frame, width=600)
    cv2.imshow('Webcam', frame)
    cv2.waitKey(1)

    return pred
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
