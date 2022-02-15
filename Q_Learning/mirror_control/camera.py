import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

colorLower = (200)
colorUpper = (250)

while(True):
 ret, frame = cap.read()
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #print (frame.shape) # 480:640  height:lenght
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
  cv2.line(frame,(0,x_axe),(640,x_axe),(155,69,0),2)

 y_laser = 320
 x_laser = 459

 test_y = y_laser
 test_x = x_laser
 
 cv2.circle(frame,(y_laser, x_laser), 20, (0,0,255), -1)

 for a in range (0,5,1): 
  pos_x0 = 0 + 96*a 
  pos_x1 = pos_x0 + 96
  
  pos_y0 = 0 + 128*a 
  pos_y1 = pos_y0 + 128
  
  if ((test_x > pos_x0) & (test_x < pos_x1)): 
   print ("y",a)
  if ((test_y > pos_y0) & (test_y < pos_y1)): 
   print ("x", a)
   s=2         

 cv2.imshow("Frame", frame)
 if cv2.waitKey(1) & 0xFF == ord('q'):
  break

cap.release()
cv2.destroyAllWindows()
