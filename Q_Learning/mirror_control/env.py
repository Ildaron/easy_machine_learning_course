import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)
one_time = 0

def camera (steps, state): # x_offset, y_offset передать себя текущею позицию

 x_task = 300
 y_task = 300
 #print (state) 
 test_x=state[0]
 test_x=test_x[0]
 
 test_y=state[0]
 test_y=test_y[1]
 global one_time
 ret, frame = cap.read()
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 #print ("steps", steps, "test_y", test_y, "test_x", test_x)
 if (steps == 0):
  #print ("ok0")   
  y_laser_after = test_y + 10
  x_laser_after = test_x 
 if (steps == 1 ): #& test_y>0
  #print ("ok1")   
  y_laser_after = test_y - 10 #y_laser = y_before - 50
  x_laser_after = test_x 
 #else:
 # y_laser_after = test_y
 # x_laser_after = test_x
 if (steps == 2):
  #print ("ok2")   
  y_laser_after = test_y 
  x_laser_after = test_x +10
  
 if (steps == 3 ):#& test_x>0
  #print ("ok3")   
  y_laser_after = test_y 
  x_laser_after = test_x - 10 #x_laser = x_before -50 
 # else:
 # y_laser_after = test_y
 # x_laser_after = test_x
 # print ("y_laser_after",y_laser_after,"x_laser_after", x_laser_after)
  
 if ((abs(x_task - x_laser_after) < abs(x_task - test_x))):
  reward_x = 1
 else:
  reward_x = 0
  
 if ((abs(y_task-y_laser_after) < abs(y_task - test_y))):
  reward_y = 1
 else:
  reward_y = 0

 #print ("reward_x",reward_x,"reward_y",reward_y)  
 reward = reward_x+reward_y 

 cv2.circle(frame,(x_task, y_task), 5, (250,0,255), -1)
 cv2.circle(frame,(x_laser_after, y_laser_after), 5, (0,0,255), -1)
 cv2.imshow("Frame", frame)

 #x_before = x_laser
 #y_before = y_laser

 if cv2.waitKey(1) & 0xFF == ord('q'):
  print ("break")
 # break
 if ((y_laser_after>480) or (x_laser_after>640) or (y_laser_after<0) or (x_laser_after<0)):
  condition=0
  print ("stop game")
 else:
  condition=1
  
 
# if (reward==2 | reward==1):
#  reward=1
# else:
#  reward=0   

 #if (reward==1):
 # reward=1
 #print ("condition",condition)
 #print ()
 #print ("reward", reward)
 return (reward, condition, x_laser_after, y_laser_after)
 #return (reward_x,reward_y, condition, y_laser, x_laser )

 cap.release()
 cv2.destroyAllWindows()

#while 1:
#for _ in range (0,50,1):
# time.sleep(1)         
# y_offset = 25
# x_offset = 25
# print (camera(x_offset,y_offset))
