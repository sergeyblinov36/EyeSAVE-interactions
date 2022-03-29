from Facial_expressions.main import get_expression
from Distance.distance import get_distance
from Facial_expressions import image_manipulation
import camera



def main():
    camera.set_source("inter2.mp4")
    # camera.set_source(0)
    get_distance()
    # get_expression()
    print("main")

main()





