from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import math
import  cv2
import time

#frame = np.zeros((200,200,1), dtype = np.uint8)

#cv2.imshow("Frame", frame)
class ShowerEnv(Env):
    def __init__(self):
        # Actions we can take, down, stay, up
        self.action_space = Discrete(4)
        # Temperature array
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        # Set start temp
        self.state_x = 38
        self.state_y = 38
        # Set shower length
        self.shower_length = 1000
        
    def step(self, action):
        #print ("action", action)
        x_task=100
        y_task=100
        
        self.shower_length -= 1 
        distance_before = (x_task -  self.state_x)**2+(y_task-self.state_y)**2 
        distance_before = math.sqrt(distance_before)
        if (action == 0):
            self.state_x += 1 
            # y_laser_after = test_y + 1
            # x_laser_after = test_x 
        if (action == 1 ): #& test_y>0
            self.state_x -= 1 
            #y_laser_after = test_y - 1 #y_laser = y_before - 50
            #x_laser_after = test_x 
        if (action == 2):
            self.state_y  += 1
            #y_laser_after = test_y 
            #x_laser_after = test_x +1
        if (action == 3 ):#& test_x>0
            self.state_y  -= 1
            #y_laser_after = test_y 
            #x_laser_after = test_x - 1 #x_laser = x_before -50 

        distance_after = (x_task -  self.state_x)**2+(y_task-self.state_y)**2 
        distance_after = math.sqrt(distance_after)
        frame=cv2.imread("C:/Users/ir2007/Desktop/HW/1.Reinforcment_learning/shower/works/img.bmp")
        cv2.circle(frame,(x_task, y_task), 5, (250,0,255), -1)
        cv2.circle(frame,(self.state_x, self.state_y), 5, (0,0,255), -1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          print ("break")     

 
        if abs(distance_after)<abs(distance_before):
            reward = 1 
        else: 
            reward = -1 
        
        # Check if shower is done
        if (self.shower_length <= 0 or(self.state_x >200) or (self.state_y>200) or (self.state_x <0) or (self.state_y<0)): 
            done = True
        else:
            done = False
        
        # Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        # Return step information
        self.state=distance_after
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass
    
    def reset(self):
        # Reset shower temperature
        self.state = 38 + random.randint(-3,3)
        self.state_x = 38
        self.state_y = 38
        # Reset shower time
        self.shower_length = 1000
        return self.state
    
