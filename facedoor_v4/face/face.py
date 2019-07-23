import cv2
import numpy as np
import random
import dlib


class Face_detection(object):
    def __init__(self):
        # Define window names
        self.win_delaunay = "Delaunay Triangulation"
        self.win_voronoi = "Voronoi Diagram"

        # Turn on animation while drawing triangles
        self.animate = True

        # Define colors for drawing.
        self.delaunay_color = (255, 255, 255)
        self.points_color = (0, 0, 255)
        size = img.shape
        a, b, c, d = 0, 0, size[1], size[0]
        self.rect = (a, b, c, d)
        print(self.rect)

    def detecter(self, frame):
        

    def show_1(self):

    def show_2(self):



if __name__ == "__main__":
    f = Face_detection()
    cap = cv2.VideoCapture(0)
    while True:
        hasFrame, frame = cap.read()
        f.detecter(frame)
        f.show_1()
        key = cv2.waitKey(10)
        if key == ord('d'):
            f.show_2()
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
