import cv2
import numpy as np
import random
import dlib
from emotions_master.emotion import Emotion_detection

# Check if a point is inside a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True


# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)


# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color):

    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(
                r, pt3):

            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


# Draw voronoi diagram
def draw_voronoi(img, subdiv):

    (facets, centers) = subdiv.getVoronoiFacetList([])

    for i in range(0, len(facets)):
        ifacet_arr = []
        for f in facets[i]:
            ifacet_arr.append(f)

        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255))

        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0)
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0),
                   cv2.FILLED, cv2.LINE_AA, 0)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    '''
    To draw some fancy box around founded faces in frame
        https://stackoverflow.com/questions/46036477/drawing-fancy-rectangle-around-face
    '''

    x1, y1 = pt1
    x2, y2 = pt2

    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)


if __name__ == '__main__':

    e = Emotion_detection(path='emotions_master/')

    # Turn on animation while drawing triangles
    animate = True

    # Define colors for drawing.
    delaunay_color = (255, 255, 255)
    points_color = (0, 0, 255)

    cap = cv2.VideoCapture(0)

    ret, img = cap.read()

    # Rectangle to be used with Subdiv2D
    size = img.shape
    a, b, c, d = 0, 0, size[1], size[0]
    rect = (a, b, c, d)
    print(rect)

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Create an array of points.
    points = []

    #人脸分类器
    detector = dlib.get_frontal_face_detector()
    # 获取人脸检测器
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    count = 10 

    while ret:
        ret, img = cap.read()
        ret, img = cap.read()
        ret, img = cap.read()
        img_orig = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dets = detector(gray, 1)

        if len(dets) == 0:
            cv2.imshow("img", img)
            key = cv2.waitKey(10)
            if key == ord('q'):
                break
            continue

        points = []
        for face in [dets[0]]:
            shape = predictor(img, face)  # 寻找人脸的68个标定点
            a, b, c, d = face.left(), face.top(), face.right(), face.bottom()

            draw_border(img, (a, b), (c, d), (205, 92, 92), 2, 10, 20)
            # Create an instance of Subdiv2D
            # subdiv = cv2.Subdiv2D(rect);
            # 遍历所有点，打印出其坐标，并圈出来
            for pt in shape.parts():
                pt_pos = (pt.x, pt.y)
                points.append(pt_pos)
                cv2.circle(img, pt_pos, 2, (0, 255, 0), 1)

        e.detecter(img)
        # cv2.rectangle(img, (e.x, e.y),
        #               (e.x + e.w, e.y + e.h), e.color, 2)
        cv2.putText(img, e.emotion_mode,
                    (a + 30, b - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    e.color, 1, cv2.LINE_AA)

        cv2.imshow("img", img)
        key = cv2.waitKey(10)

        print(e.emotion_mode)

        if (count < 10):
            count += 1

        for p in points:
             if p[0] < 0 or p[1] < 0 or p[0] > 640  or p[1] > 480:
                    count = 9 
                    print('out face')

        if count == 10 and e.emotion_mode == 'happy':
            count = 0
            print(a, b, c, d)
            subdiv = cv2.Subdiv2D(rect)
            # Insert points into subdiv
            for p in points:
                subdiv.insert(p)
                # Show animation
                if animate:
                    img_copy = img_orig.copy()
                    # Draw delaunay triangles
                    draw_delaunay(img_copy, subdiv, (255, 255, 255))
                    cv2.imshow("img", img_copy)
                    cv2.waitKey(20)

            # Draw delaunay triangles
            draw_delaunay(img, subdiv, (255, 255, 255))

            # Draw points
            for p in points:
                draw_point(img, p, (0, 0, 255))

            # Allocate space for voronoi Diagram
            img_voronoi = np.zeros(img.shape, dtype=img.dtype)

            # Draw voronoi diagram
            draw_voronoi(img_voronoi, subdiv)

            # Show results
            cv2.imshow("img", img)
            cv2.waitKey(100)
            cv2.imshow("img", img_voronoi)
            # cv2.imshow(win_voronoi,img_voronoi)
            cv2.waitKey(100)

        if key == ord('q'):
            break
