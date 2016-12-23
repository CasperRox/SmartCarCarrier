import numpy as np
import cv2
import sys

def sameSideLine(p1, p2, l1, l2):
    p1Line = (p1[0]-l1[0])*(l1[1]-l2[1]) - (p1[1]-l1[1])*(l1[0]-l2[0])
    p2Line = (p2[0]-l1[0])*(l1[1]-l2[1]) - (p2[1]-l1[1])*(l1[0]-l2[0])
    if p1Line*p2Line < 0:
        return False
    else:
        return True

def pointInTriangle(p1, t1, t2, t3):
    if sameSideLine(p1, t1, t2, t3) and sameSideLine(p1, t2, t1, 
        t3) and sameSideLine(p1, t3, t1, t2):
        return True
    else:
        return False

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("/home/rangathara/FYP/RoadDetection/CutHighwayVideo.mp4")

while (True):

    s, img = cam.read()
    # img = cv2.imread("/home/rangathara/FYP/images/testingImage4.bmp")
    height, width, colorDepth = img.shape
    # print (height, width, colorDepth)
    heightFilter1 = (int)(height * 65 / 100)
    heightFilter2 = (int)(height * 80 / 100)
    heightFilterMiddle = (int)(height * 60 / 100)
    widthFilter1 = (int)(width * 40 / 100)
    widthFilter2 = (int)(width * 60 / 100)
    widthFilterMiddle = (int)(width * 50 / 100)

    leftTriangleT1 = [0, heightFilter1]
    leftTriangleT2 = [widthFilter1, heightFilter1]
    leftTriangleT3 = [0, heightFilter2]
    rightTriangleT1 = [widthFilter2, heightFilter1]
    rightTriangleT2 = [width, heightFilter1]
    rightTriangleT3 = [width, heightFilter2]

    xLeftDown = 0
    xRightDown = sys.maxsize
    xLeftUp = width / 2
    xRightUp = width / 2

    winName = "Movement Indicator"
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(img, 100, 200)
    edges = cv2.Canny(img, 50, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 4, 2, None, 10, 1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 25, 10)

    if lines is not None:
        for line in lines:
            for obj in line:
                [x1, y1, x2, y2] = obj
                if y1<heightFilter1 or y2<heightFilter1:    # To ignore lines above the road
                    continue
                dx, dy = x2 - x1, y2 - y1
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if angle <= 20 and angle >= -20:    # To ignore horizontal lines
                    continue
                if dy<50 and dy>-50:    # To ignore short lines
                    continue
                if pointInTriangle([x1,y1], leftTriangleT1, leftTriangleT2, 
                    leftTriangleT3) or pointInTriangle([x2,y2], leftTriangleT1, 
                    leftTriangleT2, leftTriangleT3):  # To ignore lines beyond the road in left
                    continue
                if  pointInTriangle([x1,y1], rightTriangleT1, rightTriangleT2, 
                    rightTriangleT3) or pointInTriangle([x2,y2], rightTriangleT1, 
                    rightTriangleT2, rightTriangleT3):  # To ignore lines beyond the road in right
                    continue
                if 0 != dy:     # if dy=0 following equations will not work
                    x1New = (int)((((height-y1)/dy*dx)+x1))         # Increase the line length
                    x2New = (int)((((heightFilter1-y1)/dy*dx)+x1))  # Increase the line length
                    if x1New <= widthFilterMiddle and x1New > xLeftDown:   # Left lane marking
                        xLeftDown = x1New
                        xLeftUp = x2New
                        cv2.line(img, (xLeftDown,height), (xLeftUp,heightFilter1), (0, 255, 0), 2)
                    elif x1New > widthFilterMiddle and x1New < xRightDown:  # Right lane marking
                        xRightDown = x1New
                        xRightUp = x2New
                        cv2.line(img, (xRightDown,height), (xRightUp,heightFilter1), (0, 255, 0), 2)
                    # cv2.line(img, (x1New,height), (x2New,heightFilter1), (0, 255, 0), 2)
                # cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
        # pts = np.array([[xLeftDown,height],[xLeftUp,heightFilter1],[xRightUp,heightFilter1],
            # [xRightDown,height]], np.int32)
        pts = np.array([[xLeftDown,height],[xLeftUp,heightFilter1],[xLeftUp,heightFilterMiddle],
            [xLeftDown,heightFilterMiddle]], np.int32)
        pts2 = np.array([[xRightDown,height],[xRightUp,heightFilter1],[xRightUp,heightFilterMiddle],
            [xRightDown,heightFilterMiddle]], np.int32)

        # pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img,[pts],(0,255,255,0.1))
        cv2.fillPoly(img,[pts2],(0,255,255,0.1))

    cv2.imshow('edges', edges)
    cv2.imshow('original', img)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
