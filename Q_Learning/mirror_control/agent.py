import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import env

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000) #2000
        self.gamma = 0.95                #discount rate
        self.epsilon = 1.0               #exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #model.add(Dense(4, activation='relu'))
        #model.add(Dense(self.action_size, activation='linear'))
        #model.compile(loss='mse',optimizer=Adam(lr=self.learning_rate))
        model.add(Dense(4, activation='softmax'))        
        model.compile(optimizer='adam',loss='categorical_crossentropy')
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon: 
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            print ("reward", reward)
            target = reward
            if not done:
                
             target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))
            # начение Q для определенной пары состояние-действие должно быть наградой,
            # полученной при переходе в новое состояние (путем выполнения этого действия), добавленной к значению наилучшего действия в следующем состоянии.
            # обесценивание будующих наград
            target_f = self.model.predict(state)
            target_f[0][action] = target       
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    #env = gym.make('CartPole-v1')
    state_size = 2
    action_size = 4
    agent = DQNAgent(state_size, action_size) 
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 1 # 32 #128

    for e in range(EPISODES):
        state=[[25,25]] # start point
        state = np.reshape(state, [1, 2])
        for time in range(5000): 
            action = agent.act(state) 
            data_from_env = env.camera(action, state)
            reward = data_from_env[0]
            next_state = [data_from_env[2], data_from_env[3]]  # x and y coordinate
            next_state = np.reshape(next_state, [1, 2])
            done =  data_from_env[1]            
            #reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, 2])
            agent.memorize(state, action, reward, next_state, done)

            state = next_state
            if done==0:
                #print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > 100: #batch_size
                #print ("problem")
                agent.replay(batch_size)
        # if e % 10 == 0:
        #     agent.save("./save/cartpole-dqn.h5")
