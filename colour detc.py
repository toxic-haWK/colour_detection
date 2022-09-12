
import numpy as np
import cv2

webcam = cv2.VideoCapture(0)
#imageFrame=cv2.imread("__________")

while(1):
	_, imageFrame = webcam.read()

	hsv = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)

    #==red===
	red_lower = np.array([120, 60, 100])
	red_upper = np.array([180, 255, 255])
	red_mask = cv2.inRange(hsv, red_lower, red_upper)

	# ==green====
	green_lower = np.array([35, 70, 70])
	green_upper = np.array([90, 255, 255])
	green_mask = cv2.inRange(hsv, green_lower, green_upper)

	# ==blue==
	blue_lower = np.array([110, 50, 50])
	blue_upper = np.array([140, 255, 255])
	blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
	

	#=====yellow====
	yellow_lower=np.array([10,110,110])
	yellow_upper=np.array([35,255,255])
	yellow_mask=cv2.inRange(hsv,yellow_lower,yellow_upper)

#===============================================================================
	kernal = np.ones((5, 5), "uint8")
	
	# For red color
	mask = cv2.dilate(red_mask, kernal)
	res_red = cv2.bitwise_and(imageFrame, imageFrame,mask = mask)
	cv2.imshow("red ", res_red)
    
	# For yellow color
	yellow_mask = cv2.dilate(yellow_mask, kernal)
	res_yellow = cv2.bitwise_and(imageFrame, imageFrame,mask = yellow_mask)
	cv2.imshow("yellow ", res_yellow)
	
	# For green color
	green_mask = cv2.dilate(green_mask, kernal)
	res_green = cv2.bitwise_and(imageFrame, imageFrame,mask = green_mask)
	cv2.imshow("green ", res_green)

	# For blue color
	blue_mask = cv2.dilate(blue_mask, kernal)
	res_blue = cv2.bitwise_and(imageFrame, imageFrame,mask = blue_mask)
	cv2.imshow("blue", res_blue)
#=================================================================================
	# Creating contour to track red color
	contours, hierarchy = cv2.findContours(red_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 0, 255), 2)
			
			cv2.putText(imageFrame, "Red Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0, 0, 255))

#======================================================================================
        # Creating contour to track yellow color
	contours, hierarchy = cv2.findContours(yellow_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0,255,255), 2)
			
			cv2.putText(imageFrame, "yellow Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,255))

#=================================================================================
	# Creating contour to track green color
	contours, hierarchy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(0, 255, 0), 2)
			
			cv2.putText(imageFrame, "Green Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0))
#=================================================================================
	# Creating contour to track blue color
	contours, hierarchy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 300):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(imageFrame, (x, y),(x + w, y + h),(255, 0, 0), 2)
			
			cv2.putText(imageFrame, "Blue Colour", (x, y),cv2.FONT_HERSHEY_SIMPLEX,1.0, (255, 0, 0))
			
#=========MAIN====================================================================
	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
	if cv2.waitKey(10) & 0xFF == ord('q'):
		cap.release()
		cv2.destroyAllWindows()
		break
