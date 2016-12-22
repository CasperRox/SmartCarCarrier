import numpy as np
import cv2

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
cam = cv2.VideoCapture('CutHighwayVideo.mp4')

while (True):

    s, img = cam.read()
    # img = cv2.imread("/home/rangathara/FYP/images/testingImage4.bmp")
    height, width, colorDepth = img.shape
    # print (height, width, colorDepth)
    heightFilter1 = (int)(height * 60 / 100)
    heightFilter2 = (int)(height * 80 / 100)
    widthFilter1 = (int)(width * 40 / 100)
    widthFilter2 = (int)(width * 60 / 100)

    leftTriangleT1 = [0, heightFilter1]
    leftTriangleT2 = [widthFilter1, heightFilter1]
    leftTriangleT3 = [0, heightFilter2]
    rightTriangleT1 = [widthFilter2, heightFilter1]
    rightTriangleT2 = [width, heightFilter1]
    rightTriangleT3 = [width, heightFilter2]

    winName = "Movement Indicator"
    cv2.namedWindow(winName, cv2.WINDOW_AUTOSIZE)

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = cv2.Canny(img, 100, 200)
    edges = cv2.Canny(img, 50, 150)
    # lines = cv2.HoughLinesP(edges, 1, np.pi / 4, 2, None, 10, 1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 25, 10)

    if lines is not None:
        for line in lines:
            # pt1 = (line[0], line[1])
            # pt2 = (line[2], line[3])
            for obj in line:
                [x1, y1, x2, y2] = obj
                if y1<heightFilter1 or y2<heightFilter1:
                    continue
                dx, dy = x2 - x1, y2 - y1
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if angle <= 20 and angle >= -20:
                    continue
                if dy<50 and dy>-50:
                    continue
                if pointInTriangle([x1,y1], leftTriangleT1, leftTriangleT2, 
                    leftTriangleT3) or pointInTriangle([x1,y1], rightTriangleT1, 
                    rightTriangleT2, rightTriangleT3):
                    continue
                if 0 != dy:
                    x1New = (int)((((height-y1)/dy*dx)+x1))
                    x2New = (int)((((heightFilter1-y1)/dy*dx)+x1))
                    # print (x1New, height, x2New, heightFilter)
                # print (obj)
            # cv2.line(img, pt1, pt2, (0, 0, 255), 3)
                # cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    cv2.line(img, (x1New,height), (x2New,heightFilter1), (0, 255, 0), 2)

    cv2.imshow('edges', edges)
    cv2.imshow('original', img)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
