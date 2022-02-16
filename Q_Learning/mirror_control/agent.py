#https://medium.com/@gtnjuvin/my-journey-into-deep-q-learning-with-keras-and-gym-3e779cc12762
import gym
import random
import os
import numpy as np
from collections      import deque
from keras.models     import Sequential
from keras.layers     import Dense
from tensorflow.keras.optimizers import Adam

class Agent():
    def __init__(self, state_size, action_size):
        self.state_size         = state_size
        self.action_size        = action_size
        self.memory             = deque(maxlen=2000)
        self.learning_rate      = 0.001
        self.gamma              = 0.95  # гамма — используется для расчета будущего вознаграждения со скидкой.
        self.exploration_rate   = 1.0   # когда агент становится более опытным, мы позволяем ему решать, какое действие предпринять.
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995 # мы хотим уменьшить количество исследований, так как играть в игры становится все лучше и лучше.
        self.brain              = self._build_model()

    def _build_model(self):
        print ("action_size", self.action_size) # action_size 2; 0 or 1 left or right 
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            print ("random",random.randrange(self.action_size) )# random 0
            return random.randrange(self.action_size) 
        act_values = self.brain.predict(state) # передаем на прогноз состояние среды
        print ("act_values",act_values)        # 
        return np.argmax(act_values[0])        # возвращает влево или в право


    def remember(self, state, action, reward, next_state, done):
        print ("state",state) #state [[ 0.0312402  -0.02082476  0.00057091  0.02942555]]
        print ("action",action) #action 1 or 0
        print ("reward",reward) #reward 1.0
        print ("next_state",next_state) #next_state [[ 0.03082371  0.174289    0.00115942 -0.2630772 ]]
        self.memory.append((state, action, reward, next_state, done))
        #print ("memory", self.memory)
        # Поэтому нам нужен список предыдущего опыта и наблюдений, чтобы переобучить модель с этим предыдущим опытом.
        # В алгоритме Deep Q Network нейронная сеть используется для выполнения наилучшего действия в зависимости от среды (обычно называемой «состоянием»).
                      #action У нас есть функция под названием Q Function, которая используется для оценки потенциального вознаграждения на основе состояния.
                      #Мы называем это Q (состояние, действие), где Q — функция, которая вычисляет ожидаемое будущее значение на основе состояния и действия.
                      #reward  from self.env.step(action)
                      #next_state from self.env.step(action)
                      #done from self.env.step(action)
        
    def replay(self, sample_batch_size):  # Эта функция уменьшает разницу между нашим прогнозом и целью на скорость обучения.
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        print ("sample_batch_start", len (sample_batch)) # запоминает 32 эпизода обучения 
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0]) #loss
            # Target - уменьшить потери, то есть разрыв между прогнозом и целью.
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)# Керас вычитает target из вывода нейронной сети и возводит ее в квадрат
                                                                # Эта функция уменьшает разницу между нашим прогнозом и целью на скорость обучения.
                                                                # И по мере того, как мы повторяем процесс обновления,
                                                                # аппроксимация значения Q сходится к истинному значению Q:
                                                                # потери уменьшаются, а оценка становится выше
                                                                # веса корректируются для модели
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes          = 290 # Это указывает, сколько игр мы хотим, чтобы агент сыграл, чтобы обучить себя.
        self.env               = gym.make('CartPole-v1')

        self.state_size        = self.env.observation_space.shape[0]
        print ("self.state_size", self.state_size)
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size)
        print ("self.agent", Agent(self.state_size, self.action_size))

    def run(self):
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])
                
                done = False
                index = 0
                #while not done: # Done логическое значение, указывающее, закончилась игра или нет.
                #self.env.render()
                action = self.agent.act(state) # передаем на прогноз состояние среду
                next_state, reward, done, _ = self.env.step(action) # передаем прогноз в среду - и тут фишка, что сама уже среда выдает - next_state, reward, done
                next_state = np.reshape(next_state, [1, self.state_size])

                self.agent.remember(state, action, reward, next_state, done) # 

                state = next_state
                index += 1
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                self.agent.replay(self.sample_batch_size)


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
