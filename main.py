import cv2 
import numpy as np
from collections import deque
from turtle import width

buffer_size = 2
pts = deque(maxlen=buffer_size)

blueLower = (110,50,50)
blueUpper = (130,255,255) 
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success , imgOriginal = cap.read()
    if success:
        blur = cv2.GaussianBlur(imgOriginal, (11,11), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        cv2.imshow("hsw" , hsv)

        mask = cv2.inRange(hsv, blueLower, blueUpper)

        cv2.imshow("masked", mask)

        mask = cv2.erode(mask, None, iterations=2 )
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("mask+erozyon ve genisleme", mask)

        (contours, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        if len(contours) > 0:
            c = max(contours, key = cv2.contourArea)
            rect = cv2.minAreaRect(c)
            ((x,y),(width, height),rotation) = rect
            s = "x: {}, y: {}, width: {}, height: {}, rotation: {}".format(np.round(x), np.round(y), np.round(width), np.round(height),np.round(rotation))
            print(s)
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            M = cv2.moments(c)
            center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
            cv2.drawContours(imgOriginal, [box],0,(0,0,255),2)
            cv2.circle(imgOriginal, center , 5,(255,0,255),-1)
            cv2.putText(imgOriginal,s,(25,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,3/5,(0,0,0),1 )

            pts.appendleft(center)

            for i in range(1,len(pts)):
                if pts[i-1] is None or pts[i] is None:continue
                cv2.line(imgOriginal, pts[i-1],pts[i],(0,255,0),3)
            cv2.imshow("merkezi nokta tspt", imgOriginal)

        if cv2.waitKey(1) & 0xFF == ord("q"): break
