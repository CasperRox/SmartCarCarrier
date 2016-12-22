import numpy as np
import cv2

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('CutHighwayVideo.mp4')

while (True):

    s, img = cam.read()
    # img = cv2.imread("/home/rangathara/FYP/images/testingImage4.bmp")
    ht, wd, dp = img.shape
    htFilter = ht * 60 / 100

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
                # print (obj)
                [x1, y1, x2, y2] = obj
                if y1<htFilter or y2<htFilter:
                    continue
                dx, dy = x2 - x1, y2 - y1
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if angle <= 20 and angle >= -20:
                    continue

            # cv2.line(img, pt1, pt2, (0, 0, 255), 3)
                cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)

    cv2.imshow('edges', edges)
    cv2.imshow('original', img)

    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
