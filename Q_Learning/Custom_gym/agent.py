
import env_shower
env = env_shower.ShowerEnv()
print (env.observation_space.sample())

episodes = 250
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0 
    
    while not done:
        #env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score+=reward
        print ("score", score)
    print('Episode:{} Score:{}'.format(episode, score))


import numpy as np
from tensorflow.keras.models import Sequential
#from keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

states = env.observation_space.shape
actions = env.action_space.n
#print ("actions", actions)
def build_model(states, actions):
    model = Sequential()    
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


model = build_model(states, actions)
del model

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


model = build_model(states, actions)

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

print ("ok")
dqn = build_agent(model, actions)
print ("ok1")
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
print ("ok2")
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
print ("ok3")



