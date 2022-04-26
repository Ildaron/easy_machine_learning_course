from gym import Env
from gym.spaces import Discrete, Box
import numpy as np

import  cv2

class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(4)
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([200, 200]))
        self.task = np.array([100, 100])
        self.state = None
        self.total_steps = 200
        self.n_steps = self.total_steps

        self.actions = {
            0: np.array([1, 0]),
            1: np.array([-1, 0]),
            2: np.array([0, 1]),
            3: np.array([0, -1]),
        }
        
    def step(self, action):
        self.n_steps -= 1

        distance_before = np.linalg.norm(self.state - self.task)
        self.state += self.actions[int(action)]
        distance_after = np.linalg.norm(self.state - self.task)
        succeeded = self.is_success()

        frame=a = np.zeros((200,200,1), dtype = np.uint8)
        cv2.circle(frame,(self.task), 5, (250,0,255), -1)
        cv2.circle(frame,(self.state), 5, (250,0,255), -1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          print ("break")     

        if succeeded:
            # motivate agent to solve task precisely
            reward = 100
        elif distance_after < distance_before:
            reward = 1
        else: 
            reward = -1

        return self.state, reward, self.is_terminal(), {'success': succeeded, 'distance': distance_after}

    def render(self, mode='human'):
        assert False
    
    def reset(self):
        self.state = np.random.randint(0, 200, (2,))
        self.n_steps = self.total_steps
        return self.state

    def is_terminal(self):
        return self.n_steps == 0 or \
               self.state[0] > 200 or \
               self.state[1] > 200 or \
               self.state[0] < 0 or \
               self.state[1] < 0 or \
               self.is_success()

    def is_success(self):
        return self.state[0] == self.task[0] and self.state[1] == self.task[1]
