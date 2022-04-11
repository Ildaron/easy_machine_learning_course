import cv2
import numpy as np
import time
#cap = cv2.VideoCapture(0)
import math
one_time = 0

def camera (steps, state): # steps - комманда на действие, state - текущее состояние 

 x_task = 100
 y_task = 100
 #print (state) 
 test_x=state[0]
 test_x=test_x[0]
 
 test_y=state[0]
 test_y=test_y[1]
 global one_time
 frame=cv2.imread("img.bmp") # для примера
 frame=cv2.resize(frame,(200,200))

 if (steps == 0):   
  y_laser_after = test_y + 10
  x_laser_after = test_x 
 if (steps == 1 ): #& test_y>0
  y_laser_after = test_y - 10 #y_laser = y_before - 50
  x_laser_after = test_x 
 if (steps == 2):
  y_laser_after = test_y 
  x_laser_after = test_x +10
 if (steps == 3 ):#& test_x>0
  y_laser_after = test_y 
  x_laser_after = test_x - 10 #x_laser = x_before -50 

  
 if ((abs(x_task - x_laser_after) < abs(x_task - test_x))):
  reward_x = 1
 else:
  reward_x = 0
 if ((abs(y_task-y_laser_after) < abs(y_task - test_y))):
  reward_y = 1
 else:
  reward_y = 0

 if (y_laser_after==100 & x_laser_after==100): # когда дошли до нужной точки
  reward=2
  condition=0
 #else:           
 # reward=0   
 #calculate distance  - думал ввести коээфициент для того чтобы увеличить награду если становимся ближе к цели
 distance_after = (x_task -  x_laser_after)**2+(y_task-y_laser_after)**2 
 distance_after = math.sqrt(distance_after)
 #print ("distance", distance)

 distance_before = (x_task -  test_x)**2+(y_task-test_y)**2 
 distance_before = math.sqrt(distance_before)

 distance = abs(distance_before-distance_after)
 reward = reward_x+reward_y

 if distance_after<distance_before:
  reward=1
 else:
  reward=0

 if x_laser_after==x_task & y_laser_after==y_task:
  reward=2   
 
# try:
#  reward= 100/distance
# except ZeroDivisionError:
#  reward = 100
  
 cv2.circle(frame,(x_task, y_task), 5, (250,0,255), -1)
 cv2.circle(frame,(x_laser_after, y_laser_after), 5, (0,0,255), -1)
 cv2.imshow("Frame", frame)

 if cv2.waitKey(1) & 0xFF == ord('q'):
  print ("break")
 # break

 if ((y_laser_after>200) or (x_laser_after>200) or (y_laser_after<0) or (x_laser_after<0)):
  condition=0
  print ("stop game")
  y_laser_after = 0
  x_laser_after = 0
  reward=0
 else:
  condition=1 
 return (reward, condition, x_laser_after, y_laser_after)
