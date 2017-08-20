# -*- coding: utf-8 -*-

import cv2
import numpy as np


# getting the video information
video = cv2.VideoCapture("example3.mp4")
firstFrame = None
#get one frame from the video
[grabbed, frame] = video.read()
cX = 0
midX =0
#if video information success getting.
while grabbed:
#setting the video setting
    frame = cv2.resize(frame, (720, 480))
#converting color video to grayscale and blur it.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray,(20,20))

    [nrow, ncol] = gray.shape

    mask = np.zeros((nrow, ncol), dtype=np.uint8)
#check and store the firstframe into a variable
    if firstFrame is None:
        firstFrame = gray
        continue
#create a window 3x3 size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#Background subtractor
    firstFrame_f = firstFrame.astype(np.float64)
    gray_f = gray.astype(np.float64)
    diff = firstFrame_f - gray_f
    diff = np.abs(diff)
#Dilate the frame and mask it using thresholding
    diff = cv2.dilate(diff,kernel,iterations=2) 
    mask[diff>35] = 255
#Find the contour (edges)
    [cnts, _] = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    
    counter = 0

    for c in cnts:
        
        area = cv2.contourArea(c)
        
        if area<2500:
            continue
        else:
            #getting the previous Centroid(mid point)
            if cX > 0:
                midX = cX
            #Find the initial coordinates and size of the rectangle
            [x, y, w, h] = cv2.boundingRect(c)
            #accessing the moments dictionary  to find the Centroid(midpoint)
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            count = count +1
            #create a rectangle with the initial coordinates
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            #create a dot to represent mid point
            cv2.circle(frame, (cX, cY), 7, (0, 255, 0), -1)
            #check whether the frame contains how many object
            if len(cnts) < 2:
                if cX < midX - 2:
                    #object going left
                    cv2.putText(frame,"Object heading: left",(cX,cY+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                elif cX > midX + 2:
                    #object going right
                    cv2.putText(frame,"Object heading: right",(cX,cY+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
                else:
                    #object is stationary
                    cv2.putText(frame,"Object heading: stationary",(cX,cY+100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
            #show the direction        
            cv2.putText(frame, "Count " + str(count) , (cX,cY ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            
    #show the total object
    cv2.putText(frame, "Number of Objects : " + str(count) , (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #show the Processed video
    cv2.imshow("Video", frame)
    #show the thresholding mask
    cv2.imshow("Threshhold", mask)
    #Close application
    key = cv2.waitKey(1) & 0xFF
        
    if key == ord("q"):
        break
        
    [grabbed, frame] = video.read()
    
video.release()
cv2.destroyAllWindows()