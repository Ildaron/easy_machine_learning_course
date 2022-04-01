#https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
import numpy
import env
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from tensorflow.keras.optimizers import Adam
import time

one_time=0
class Agent():
    def __init__(self, state_size, action_size):
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95    # гамма — используется для расчета будущего вознаграждения со скидкой.
        self.exploration_rate   = 1.0     # когда агент становится более опытным, мы позволяем ему решать, какое действие предпринять.
        self.exploration_min    = 0.01    #
        self.exploration_decay  = 0.995   # мы хотим уменьшить количество исследований, так как играть в игры становится все лучше и лучше.
        self.brain              = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu'))
        model.add(Dense(6, activation='relu')) #24
        model.add(Dense(4, activation='softmax'))        
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        return model

    def act(self, state):
        self.coordnate = self.brain.predict(state) # передаем на прогноз состояние среды
        return np.argmax(self.coordnate)

    def remember(self, state, action, reward, next_state, done):        
        self.memory.append((state, action, reward, next_state, done))
        
    def replay(self, sample_batch_size):  # Эта функция уменьшает разницу между нашим прогнозом и целью на скорость обучения.
        if len(self.memory) <32: # sample_batch_size: # пока 32 раща не сделает, собака,знак лти не наоборот
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            
            target = reward
            # print ("reward", target)                           # target 1
            if (done == 1):
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
              
            target_f = self.brain.predict(state)                # target_f [[0.0000000e+00 0.0000000e+00 1.0000000e+00 7.3512885e-10]]
            # target_f =  np.argmax(target_f)

            if type(state)==list:
             state= np.array([[50, 50]])
            print ("target",target) 
            target_f[0][action] = target # target_f[0][action]  = 1.95 # target = 1.95
            
            #time.sleep(2)                

            #print ("action",action)      # action = 2
            #print ("target_f",target_f)  # target_f [[4.3031235e-11 6.6206676e-18 1.9500000e+00 3.1728639e-26]]
            self.brain.fit(state, target_f, epochs=1, verbose=0)# model.fit(x_train, y_train)   Керас вычитает target из вывода нейронной сети и возводит ее в квадрат

            
            
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32   # Размер партии образца
        self.episodes          = 600 # Это указывает, сколько игр мы хотим, чтобы агент сыграл, чтобы обучить себя.
        self.state_size        = 2   # self.env.observation_space.shape[0]
        self.action_size       = 2   # self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)

    def run(self):
        
            for index_episode in range(self.episodes):
                global one_time
                if (one_time == 0):
                    one_time=1
                    state = [[150,150]]
                index = 0

                #print ("state", state)
                action = self.agent.act(state)           # y_laser, x_laser #передаем на прогноз состояние среду
                x_y_coord = env.camera(action, state)    # x_offset, y_offset
                
                reward = x_y_coord[0]
                next_state = [x_y_coord[2], x_y_coord[3]]

                #print ("next_state", next_state)        # ildar [ 0.04968709 -0.17844233  0.04515736  0.33838844]                 

                next_state = np.reshape(next_state, [1, self.state_size])


                #print ("next_state_reshape", next_state)# ildar1 [[ 0.04968709 -0.17844233  0.04515736  0.33838844]]

                done =  x_y_coord[1]
                #state = next_state 

                self.agent.remember(state, action, reward, next_state, done) # 

                state = next_state 

                index += 1
                #print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size) # собака sample_batch_size = 32


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
