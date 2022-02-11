import cv2
import numpy as np
import time
cap = cv2.VideoCapture(1)

colorLower = (200)
colorUpper = (250)

while(True):
 ret, frame = cap.read()
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #print (frame.shape) 480:640
 step=5 
 step_x=(frame.shape[0])
 step_y=(frame.shape[1])

 y_resolution = [128,256,384,512]
 x_resolution = [96,196,288,384]
 
 circle = cv2.inRange(frame, colorLower, colorUpper)
 #cv2.imshow("Frame", frame)
 
 cnts = cv2.findContours(circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

 for c in cnts:   
  ((x, y), radius) = cv2.minEnclosingCircle(c)
  if radius > 0:
   cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)   
 for y_axe in y_resolution:
  cv2.line(frame,(y_axe,0),(y_axe,640),(255,0,0),2)
 for x_axe in x_resolution:
  cv2.line(frame,(0,x_axe),(640,x_axe),(255,0,0),2)

 cv2.circle(frame,(320,240), 20, (0,0,255), -1)
 
 cv2.imshow("Frame", frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
  break

cap.release()
cv2.destroyAllWindows()
