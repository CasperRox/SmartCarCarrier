import numpy as np
import cv2
import sys

# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("/home/rangathara/FYP/RoadDetection/CutHighwayVideo.mp4")

while (True):
	s, img = cam.read()
	# img = cv2.imread("/home/rangathara/FYP/images/ADS_1086.JPG")
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgGray,35,255,cv2.THRESH_BINARY)
	edges = cv2.Canny(imgGray, 30, 150)
	# edges = cv2.Canny(thresh, 30, 150)

	kernel = np.ones((5,5),np.uint8)
	closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	ret,thresh1 = cv2.threshold(opening,35,255,cv2.THRESH_BINARY_INV)
	cv2.imshow("morphology1", thresh1)

	# im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	im2,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	# im2,contours,hierarchy = cv2.findContours(thresh, 1, 2)
	# im2,contours,hierarchy = cv2.findContours(edges, 1, 2)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

	for cnt in contours[1:4]:
	# cnt = contours[0]
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

	cv2.imshow("morphology", thresh1)
	cv2.imshow("threshold", thresh)
	cv2.imshow("edge", edges)
	cv2.imshow("result", img)

	if cv2.waitKey(10) & 0xff == ord('q'):
		break

# cv2.waitKey(0)
cam.release()
cv2.destroyAllWindow()
