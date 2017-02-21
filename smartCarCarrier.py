import numpy as np
import cv2
import imutils
import math
import time

def calculateVehicleDistance():
	global timePrevious
	timeCurrent = round(time.time()*1000)	# current time in milliseconds
	timeDiff = (timeCurrent - timePrevious) / 1000	# time difference (s)
	# print ("t", timeDiff)
	# acceleration = 2	# this value is from the accelerometer reading (m/s2)
	# velocity = acceleration * timeDiff	# (m/s)
	velocity = 10	# this value have to be measured (m/s)
	distanceDiff = velocity * timeDiff	# (m)
	timePrevious = timeCurrent
	# print ("l", distanceDiff)
	return distanceDiff

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

def laneDetection():
	global img
	height, width, colorDepth = img.shape
	# print (height, width, colorDepth)
	heightFilter65 = (int)(height * 65 / 100)
	heightFilter80 = (int)(height * 80 / 100)
	heightFilter35 = (int)(height * 35 / 100)
	widthFilter40 = (int)(width * 40 / 100)
	widthFilter60 = (int)(width * 60 / 100)
	widthFilterMiddle = (int)(width * 50 / 100)

	leftTriangleT1 = [0, heightFilter65]
	leftTriangleT2 = [widthFilter40, heightFilter65]
	leftTriangleT3 = [0, heightFilter80]
	rightTriangleT1 = [widthFilter60, heightFilter65]
	rightTriangleT2 = [width, heightFilter65]
	rightTriangleT3 = [width, heightFilter80]

	xLeftDown = 0
	xRightDown = width
	xLeftUp = (int)(width / 2)
	xRightUp = (int)(width / 2)

	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# edges = cv2.Canny(img, 100, 200)
	edges = cv2.Canny(img, 50, 150)
	# lines = cv2.HoughLinesP(edges, 1, np.pi / 4, 2, None, 10, 1)
	lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, None, 25, 10)

	if lines is not None:
		for line in lines:
			for obj in line:
				[x1, y1, x2, y2] = obj
				if y1<heightFilter65 or y2<heightFilter65:	# To ignore lines above the road
					continue
				dx, dy = x2 - x1, y2 - y1
				angle = np.arctan2(dy, dx) * 180 / np.pi
				if angle <= 20 and angle >= -20:	# To ignore horizontal lines
					continue
				if dy<50 and dy>-50:	# To ignore short lines
					continue
				if pointInTriangle([x1,y1], leftTriangleT1, leftTriangleT2, 
					leftTriangleT3) or pointInTriangle([x2,y2], leftTriangleT1, 
					leftTriangleT2, leftTriangleT3):  # To ignore lines beyond the road in left
					continue
				if  pointInTriangle([x1,y1], rightTriangleT1, rightTriangleT2, 
					rightTriangleT3) or pointInTriangle([x2,y2], rightTriangleT1, 
					rightTriangleT2, rightTriangleT3):  # To ignore lines beyond the road in right
					continue
				if 0 != dy:	 # if dy=0 following equations will not work
					x1New = (int)((((height-y1)/dy*dx)+x1))	 # Increase the line length
					x2New = (int)((((heightFilter65-y1)/dy*dx)+x1))  # Increase the line length
					if x1New <= widthFilterMiddle and x1New > xLeftDown:   # Left lane marking
						xLeftDown = x1New
						xLeftUp = x2New
					elif x1New > widthFilterMiddle and x1New < xRightDown:  # Right lane marking
						xRightDown = x1New
						xRightUp = x2New
					cv2.line(img, (xLeftDown,height), (xLeftUp,heightFilter65), (0, 255, 0), 2)
					cv2.line(img, (xRightDown,height), (xRightUp,heightFilter65), (0, 255, 0), 2)
					# cv2.line(img, (x1New,height), (x2New,heightFilter65), (0, 255, 0), 2)
				# cv2.line(img, (x1,y1), (x2,y2), (0, 0, 255), 2)
		# pts = np.array([[xLeftDown,height],[xLeftUp,heightFilter65],[xRightUp,heightFilter65],
			# [xRightDown,height]], np.int32)

		# Assuming that the camera is mounted at middle of the height of the vehicle
		pts1 = np.array([[0,height],[xLeftDown,height],[xLeftUp,heightFilter65],[xLeftUp,heightFilter35],
			[xLeftDown,0],[0,0]], np.int32)
		pts2 = np.array([[width,height],[xRightDown,height],[xRightUp,heightFilter65],[xRightUp,heightFilter35],
			[xRightDown,0],[width,0]], np.int32)

		# pts = pts.reshape((-1,1,2))
		cv2.fillPoly(img,[pts1],(0,255,255,0.1))
		cv2.fillPoly(img,[pts2],(0,255,255,0.1))

	# cv2.imshow('edges', edges)

def detectSameObject(imgSrc, imgTarget):
	imgOrig = imgSrc
	sizeCounter = 1.0
	zeroCountMax = 0
	scaling = 0
	xyCordinates = (0,0)
	while True:
		if imgSrc.shape[0]<imgTarget.shape[0] or imgSrc.shape[1]<imgTarget.shape[1]:
			break
		# imgDiff = np.full((imgTarget.shape[0],imgTarget.shape[1]), 255, dtype=int)
		for y in (0, imgSrc.shape[0]-imgTarget.shape[0], 1):
			for x in (0, imgSrc.shape[1]-imgTarget.shape[1], 1):
				# print (y)
				imgTemp = imgSrc[y:y+imgTarget.shape[0], x:x+imgTarget.shape[1]]
				if not np.array_equal(imgTarget.shape,imgTemp.shape):
					continue
				# imgDiff = imgTemp - imgTarget
				imgDiff = cv2.absdiff(imgTemp, imgTarget)	# per-element absolute difference
				zeroCount = (imgDiff==0).sum()	# returns the number of zeros in the array
				if zeroCount > zeroCountMax:	# get the most similar comparison
					zeroCountMax = zeroCount
					xyCordinates = (y,x)
					scaling = sizeCounter
				# result = np.any(imgDiff)	#if imgDiff is all zeros this will return False

		# compute the new dimensions of the image and resize it
		w = int(imgSrc.shape[1]*0.9)
		imgSrc = imutils.resize(imgSrc, width=w)
		sizeCounter /= 0.9
	print (zeroCountMax)
	xy = [int(x*scaling) for x in xyCordinates]
	cv2.rectangle(img,(xy[1],xy[0]),(xy[1]+imgTarget.shape[1],xy[0]+imgTarget.shape[0]),(0,0,255),2)
	cv2.imshow("Detected", imgOrig[xy[0]:xy[0]+imgTarget.shape[0], xy[1]:xy[1]+imgTarget.shape[1]])
	return scaling

def objectDetection():
	global img
	global imgPrevious
	imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	imgPreviousGray = cv2.cvtColor(imgPrevious,cv2.COLOR_BGR2GRAY)
	# ret,thresh = cv2.threshold(imgGray,35,255,cv2.THRESH_BINARY)
	edges = cv2.Canny(imgPreviousGray, 30, 150)

	kernel = np.ones((5,5),np.uint8)
	closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)	# dilation then erosion
	# cv2.imshow("morphology3", closing)
	# opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)	# erosion then dilation
	# cv2.imshow("morphology2", opening)
	ret,thresh1 = cv2.threshold(closing,127,255,cv2.THRESH_BINARY_INV)
	# cv2.imshow("morphology1", thresh1)

	# im2,contours,hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	im2,contours,hierarchy = cv2.findContours(thresh1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

	orb = cv2.ORB_create(nfeatures=100)	# default no of features is 500
	kpOrig, desOrig = orb.detectAndCompute(imgGray,None)
	bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

	for cnt in contours[1:4]:	# the biggest rectangle is around the whole image
		x,y,w,h = cv2.boundingRect(cnt)
		imgCrop = imgPreviousGray[y:y+h, x:x+w]
		kpObj, desObj = orb.detectAndCompute(imgCrop,None)
		if len(kpObj) == 0:	# no detectable features available
			continue
		matches = bf.match(desOrig,desObj)
		matches = sorted(matches, key = lambda x:x.distance)
		if len(matches) < 2:	# need minimum two feature points to get distance
			continue
		# get keypoint coordinates of first two lowest distance matches
		listkpOrig = [kpOrig[mat.queryIdx].pt for mat in matches[:2]]
		listkpObj = [kpObj[mat.trainIdx].pt for mat in matches[:2]]
		# calculate distance between first two matches
		distancekpOrig = math.sqrt(math.pow((listkpOrig[0][0]-listkpOrig[1][0]),2)+
			math.pow((listkpOrig[0][1]-listkpOrig[1][1]),2))
		distancekpObj = math.sqrt(math.pow((listkpObj[0][0]-listkpObj[1][0]),2)+
			math.pow((listkpObj[0][1]-listkpObj[1][1]),2))
		if distancekpOrig <= distancekpObj:	# detected object is too far
			continue
		# calculate distance to the object from the vehicle (m)
		# print ("a", distancekpObj)
		# print ("b", distancekpOrig)
		distanceToObject = (calculateVehicleDistance() * distancekpObj)/(distancekpOrig - distancekpObj)
		print ("distance", "%.2f" %distanceToObject, "m")
		# calibration
		# assuming 50m long lane is shown in lower 35% of the image
		if distanceToObject < 50:
			yCoordinate = (int)(img.shape[0]/100*(100-(distanceToObject / 50 * 35)))
			heightPixel = yCoordinate - listkpOrig[0][1]
			# assume the focal length of the camera is 50mm
			focalLength = 50 / 1000
			# calibration
			# height of the image = 185mm = 720px
			heightActual = (heightPixel/720*0.185) * distanceToObject / focalLength
			print ("heightLimit =", "%.2f" %heightActual, "m")
		img3 = cv2.drawMatches(imgGray,kpOrig,imgCrop,kpObj,matches[:10],None, flags=2)
		cv2.imshow("cropped", img3)
		# cv2.imshow("cropped", imgCrop)
		cv2.rectangle(imgPrevious,(x,y),(x+w,y+h),(0,0,255),2)
		cv2.putText(imgPrevious, '%.2f' %distanceToObject, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2, cv2.LINE_AA)
		# scaling = detectSameObject(imgGray, imgCrop)
		# time.sleep(0.5)

	# cv2.imshow("threshold", thresh)
	# cv2.imshow("edge", edges)



# **********Main**********

cam = cv2.VideoCapture(1)
# cam = cv2.VideoCapture("/home/rangathara/FYP/RoadDetection/Videos/WIN_20170218_171342.MP4")

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

timePrevious = round(time.time()*1000)	# current time in milliseconds
s, imgPrevious = cam.read()	# keep track of previous frame to calculate distance

while (True):
	s, img = cam.read()
	# img = cv2.imread("/home/rangathara/FYP/images/colombo---kandy-road-warakapola.jpg")

	laneDetection()
	objectDetection()

	cv2.imshow("Original", imgPrevious)

	out.write(imgPrevious)

	imgPrevious = img

	if cv2.waitKey(10) & 0xff == ord('q'):
		break

	time.sleep(0.25)

cam.release()
cv2.destroyAllWindow()
