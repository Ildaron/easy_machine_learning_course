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
 print ("steprs", steps)
 if (steps == 0):   
  y_laser_after = test_y + 1
  x_laser_after = test_x 
 if (steps == 1 ): #& test_y>0
  y_laser_after = test_y - 1 #y_laser = y_before - 50
  x_laser_after = test_x 
 if (steps == 2):
  y_laser_after = test_y 
  x_laser_after = test_x +1
 if (steps == 3 ):#& test_x>0
  y_laser_after = test_y 
  x_laser_after = test_x - 1 #x_laser = x_before -50 

  

 #else:           
 # reward=0   
 #calculate distance  - думал ввести коээфициент для того чтобы увеличить награду если становимся ближе к цели
 distance_after = (x_task -  x_laser_after)**2+(y_task-y_laser_after)**2 
 distance_after = math.sqrt(distance_after)
 #print ("distance", distance)

 distance_before = (x_task -  test_x)**2+(y_task-test_y)**2 
 distance_before = math.sqrt(distance_before)

 #distance = abs(distance_before-distance_after)

 #reward = reward_x+reward_y

 print ("distance_after",distance_after)
 #print ("distance_before",distance_before)
 #print ("reward", reward)
 if abs(distance_after)<abs(distance_before):
  #try:
  if distance_after >= 5:
   reward= 141.42/distance_after # 141.42*(1/141.42)
   #reward = reward - 4
  #except ZeroDivisionError:
  else:
   reward = 200 # 141.2
   condition=0
 else:
   reward= -distance_after/10 # 141.42*(1/141.42)
 rewar=int(reward)
 print (reward)
 
 #if x_laser_after==x_task & y_laser_after==y_task:
 # reward=1  
 # condition=0
 # try:


  
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
  reward=-100
 else:
  condition=1
 if reward==200:
  condition=0   
 return (reward, condition, x_laser_after, y_laser_after)
