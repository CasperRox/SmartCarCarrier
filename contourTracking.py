import numpy as np
import cv2
import imutils
import time

def detectSameObject(imgSrc, imgTarget):
	imgOrig = imgSrc
	sizeCounter = 1.0
	zeroCountMax = 0
	# scaling = 0
	# xyCordinates = (0,0)
	while True:
		if (imgSrc.shape[0]<imgTarget.shape[0]) or (imgSrc.shape[1]<imgTarget.shape[1]):
			break
		# imgDiff = np.full((imgTarget.shape[0],imgTarget.shape[1]), 255, dtype=int)
		for y in (0, imgSrc.shape[0]-imgTarget.shape[0], 10):
			for x in (0, imgSrc.shape[1]-imgTarget.shape[1], 10):
				# print (y)
				imgTemp = imgSrc[y:y+imgTarget.shape[0], x:x+imgTarget.shape[1]]
				imgDiff = imgTemp - imgTarget
				# imgDiff = cv2.subtract(imgTemp, imgTarget)
				zeroCount = (imgDiff==0).sum()	# returns the number of zeros in the array
				if zeroCount > zeroCountMax:	# get the most similar comparison
					zeroCountMax = zeroCount
					xyCordinates = (y,x)
					scaling = sizeCounter
				# result = np.any(imgDiff)	#if imgDiff is all zeros this will return False

		# compute the new dimensions of the image and resize it
		w = int(imgSrc.shape[1]/1.5)
		imgSrc = imutils.resize(imgSrc, width=w)
		sizeCounter *= 1.5
	xy = [int(x*scaling) for x in xyCordinates]
	cv2.imshow("Detected", imgOrig[xy[0]:xy[0]+imgTarget.shape[0], xy[1]:xy[1]+imgTarget.shape[1]])
	return scaling


# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture("/home/rangathara/FYP/RoadDetection/CutHighwayVideo.mp4")

while (True):
	s, img = cam.read()
	# img = cv2.imread("/home/rangathara/FYP/images/colombo---kandy-road-warakapola.jpg")
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	ret,thresh = cv2.threshold(imgGray,35,255,cv2.THRESH_BINARY)
	edges = cv2.Canny(imgGray, 30, 150)

	kernel = np.ones((5,5),np.uint8)
	closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
	opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
	ret,thresh1 = cv2.threshold(opening,35,255,cv2.THRESH_BINARY_INV)
	# cv2.imshow("morphology1", thresh1)

	# im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	im2,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

	# orb = cv2.ORB_create()
	# kpOrig, desOrig = orb.detectAndCompute(imgGray,None)
	# bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

	for cnt in contours[1:3]:	# the biggest rectangle is around the whole image
		x,y,w,h = cv2.boundingRect(cnt)
		imgCrop = imgGray[y:y+h, x:x+w]
		# kpObj, desObj = orb.detectAndCompute(imgCrop,None)
		# if len(kpObj) == 0:
			# continue
		# matches = bf.match(desOrig,desObj)
		# matches = sorted(matches, key = lambda x:x.distance)
		# img3 = cv2.drawMatches(imgGray,kpOrig,imgCrop,kpObj,matches[:10],None, flags=2)
		# cv2.imshow("cropped", img3)
		cv2.imshow("cropped", imgCrop)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		scaling = detectSameObject(imgGray, imgCrop)
		# print (scaling)
		# time.sleep(1)

	# cv2.imshow("threshold", thresh)
	# cv2.imshow("edge", edges)
	cv2.imshow("Original", img)

	if cv2.waitKey(10) & 0xff == ord('q'):
		break

	# time.sleep(0.25)

# cv2.waitKey(0)
cam.release()
cv2.destroyAllWindow()
