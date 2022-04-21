import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import env_shower
env = env_shower.ShowerEnv()
EPISODES = 500
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1001) 
        self.gamma = 0.95                
        self.epsilon = 0.9   #1.0           
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(4, activation='softmax'))        
        model.compile(optimizer='adam',loss='mse') 
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
            target = reward
            if not done:
             target = (reward + self.gamma *np.amax(self.model.predict(next_state)[0]))
            state = np.reshape(state, [1, 1])
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
    state_size = 1
    action_size = 4
    agent = DQNAgent(state_size, action_size) 
    done = False
    batch_size = 128 #2048 # 32 #128
    reward=0
    state = env.reset()
    for e in range(EPISODES):
        state=[[25,25]] # start point
        state = np.reshape(state, [1, 2])
        for time in range(500):             
            action = env.action_space.sample()
            state, next_state, score, done, info = env.step(action)
            reward+=score
            print ("reward", reward)
            print ("next_state",next_state)
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done==False:
                #print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > 1000: #batch_size
                agent.replay(batch_size)
