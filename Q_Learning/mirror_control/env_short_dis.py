import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

colorLower = (200)
colorUpper = (250)

#def camera (): # x_receive,y_receive

x_task = 300
y_task = 300

one_time = 0

def camera (x_offset, y_offset): # +-x_pos, +-y_pos
 global one_time
 global x_before
 global y_before
 
 ret, frame = cap.read()
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 if (one_time == 0):
  print ("one time was")
  x_before=0
  y_before=0
  one_time=1
   
 # before movinf  
 # circle = cv2.inRange(frame, colorLower, colorUpper)
 # cnts = cv2.findContours(circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
 # for c in cnts:   
 #  ((x_before, y_before), radius) = cv2.minEnclosingCircle(c)
 #  if radius > 0:
 #   cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)
 #  else:
 
 # moving action
 # simulation + 30 or - 30 in x and y
    
 # after moving

 circle = cv2.inRange(frame, colorLower, colorUpper)
 cnts = cv2.findContours(circle.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
 for c in cnts:   
  ((x, y), radius) = cv2.minEnclosingCircle(c)
  if radius > 0:
   cv2.circle(frame, (int(x), int(y)), int(radius),(0, 255, 255), 2)   
  
 y_laser = y_before + y_offset
 x_laser = x_before + x_offset
  
 if ((abs(x_task - x_laser) - abs(x_task - x_before))>0):
  reward_x = 0
 else:
  reward_x = 1        

 if ((abs(y_task-y_laser) - abs(y_task - y_before))>0):
  reward_y = 0
 else:
  reward_y = 1
 cv2.circle(frame,(320, 230), 5, (250,0,255), -1)
 cv2.circle(frame,(y_laser, x_laser), 5, (0,0,255), -1)
 cv2.imshow("Frame", frame)

 x_before = x_laser
 y_before = y_laser


 if cv2.waitKey(1) & 0xFF == ord('q'):
  print ("break")
 # break

 return (reward_x,reward_y)

 cap.release()
 cv2.destroyAllWindows()

#while 1:
for _ in range (0,50,1):
 time.sleep(1)         
 y_offset = 25
 x_offset = 25
 print (camera(x_offset,y_offset))

