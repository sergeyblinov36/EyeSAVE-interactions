from Facial_expressions.main import get_expression
from Distance.distance import get_distance
from Facial_expressions import image_manipulation
import camera
# import datetime
import time
import schedule
import requests


# rtsp://tapocamnum1:Ss321352387@30.30.23.34:554/stream1
def interaction():
    source = []
    # source.append("inter2.mp4")
    # source.append("test1.mp4")
    # source.append("child1.mp4")
    source.append("rtsp://tapocamnum1:Ss321352387@192.168.0.3:554/stream1")
    source.append("rtsp://tapocamnum2:Ss321352387@192.168.0.8:554/stream1")
    # source.append(0)
    camera.set_source(source)
    # camera.set_source(0)
    get_distance()
    # get_expression()
    print("main")


def main():
    # schedule.every().day.at("21:44").do(interaction)
    # while True:
    #     schedule.run_pending()
    #     print("sleeping")
    #     time.sleep(1)
    interaction()


main()
