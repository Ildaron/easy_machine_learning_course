from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import math
import  cv2
import time

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state_x = 38
        self.state_y = 38
        self.shower_length = 10000
        
    def step(self, action):
        x_task=100
        y_task=100
        
        self.shower_length -= 1 
        distance_before = (x_task -  self.state_x)**2+(y_task-self.state_y)**2 
        distance_before = math.sqrt(distance_before)
        if (action == 0):
            self.state_x += 1 
        if (action == 1 ): 
            self.state_x -= 1 
        if (action == 2):
            self.state_y  += 1
        if (action == 3 ):
            self.state_y  -= 1

        distance_after = (x_task -  self.state_x)**2+(y_task-self.state_y)**2 
        distance_after = math.sqrt(distance_after)
        frame=a = np.zeros((200,200,1), dtype = np.uint8)
        cv2.circle(frame,(x_task, y_task), 5, (250,0,255), -1)
        cv2.circle(frame,(self.state_x, self.state_y), 5, (250,0,255), -1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          print ("break")     

 
        if abs(distance_after)<abs(distance_before):
            reward = 1 
        else: 
            reward = -1 
        if (self.shower_length <= 0 or(self.state_x >200) or (self.state_y>200) or (self.state_x <0) or (self.state_y<0)): 
            done = False
        else:
            done = True        
        #self.state += random.randint(-1,1)
        info = {}
        
        self.state=distance_after
        return int(distance_before), int(self.state), reward, done, info

    def render(self):
        pass
    
    def reset(self):
        self.state = 38 + random.randint(-3,3)
        self.state_x = 38
        self.state_y = 38
        self.shower_length = 10000
        return self.state
