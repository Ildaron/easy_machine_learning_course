import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

colorLower = (200)
colorUpper = (250)

def camera (x_receive,y_receive): # x_receive,y_receive
 x_task = 600
 y_task = 460

 if (x_receive - x_task>0):
  reward_x = 1
 else:
  reward_x = 0         
  
 if (y_receive - y_task>0):
  reward_y = 1
 else:
  reward_y = 0 
 
 while(True):
  ret, frame = cap.read()
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 #print (frame.shape) # 480:640  height:lenght # print (frame.shape[0])
  
  step_y=int(frame.shape[1]/5) # step = 5 
  y_resolution = [step_y,step_y*2,step_y*3,step_y*4]
  step_x=int(frame.shape[0]/5)
  x_resolution = [step_x,step_x*2,step_x*3,step_x*4]
  
  circle = cv2.inRange(frame, colorLower, colorUpper)
 #cv2.imshow("Frame", frame)
 
  cnts = cv2.findContours(circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

  for c in cnts:   
   ((x, y), radius) = cv2.minEnclosingCircle(c)
   if radius > 0:
    cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)   
  for y_axe in y_resolution:
   cv2.line(frame,(y_axe,0),(y_axe,int (frame.shape[1])),(255,0,0),2)
  for x_axe in x_resolution:
   cv2.line(frame,(0,x_axe),(int (frame.shape[1]),x_axe),(155,69,0),2)

  y_laser = 20
  x_laser = 459

  test_y = y_laser
  test_x = x_laser
 
  cv2.circle(frame,(y_laser, x_laser), 20, (0,0,255), -1)

  for a in range (0,5,1): 
   pos_x0 = 0 + step_x*a 
   pos_x1 = pos_x0 + step_x
  
   pos_y0 = 0 + step_y*a 
   pos_y1 = pos_y0 + step_y
  
   if ((test_x > pos_x0) & (test_x < pos_x1)): 
    print ("y",a)
   if ((test_y > pos_y0) & (test_y < pos_y1)): 
    print ("x", a)      

  cv2.imshow("Frame", frame)
  if cv2.waitKey(1) & 0xFF == ord('q'):
   break

  return (reward_x, reward_y)
 cap.release()
 cv2.destroyAllWindows()

while 1:
 print (camera(1,2))

 
