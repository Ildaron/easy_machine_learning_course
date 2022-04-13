#https://wingedsheep.com/lunar-lander-dqn/

import tensorflow as tf
import gym
import os
import random

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2

import numpy as np
import scipy
import uuid
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K

import env

#env = gym.make("LunarLander-v2")

def masked_huber_loss(mask_value, clip_delta):
  def f(y_true, y_pred):
    error = y_true - y_pred
    cond  = K.abs(error) < clip_delta
    mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
    masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
    linear_loss  = mask_true * (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
    huber_loss = tf.where(cond, masked_squared_error, linear_loss)
    return K.sum(huber_loss) / K.sum(mask_true)
  f.__name__ = 'masked_huber_loss'
  return f

input_shape = input_shape = (3,)#2 # 9 #8 variables in the environment + the fraction finished we add ourselves
outputs = 4

def create_model(learning_rate, regularization_factor):
  model = Sequential([
    Dense(64, input_shape=input_shape, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(64, activation="relu", kernel_regularizer=l2(regularization_factor)),
    Dense(outputs, activation='linear', kernel_regularizer=l2(regularization_factor))
  ])

  optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=masked_huber_loss(0.0, 1.0))
  
  return model



def get_q_values(model, state):
  input = state[np.newaxis, ...]
  print ("ok1")
  return model.predict(input)[0]

def get_multiple_q_values(model, states):
  print("ok2")  
  return model.predict(states)



def select_action_epsilon_greedy(q_values, epsilon):
  random_value = random.uniform(0, 1)
  if random_value < epsilon: 
    return random.randint(0, len(q_values) - 1)
  else:
    return np.argmax(q_values)

def select_best_action(q_values):
  return np.argmax(q_values)




class StateTransition():

  def __init__(self, old_state, action, reward, new_state, done):
    self.old_state = old_state
    self.action = action
    self.reward = reward
    self.new_state = new_state
    self.done = done

class ReplayBuffer():
  current_index = 0

  def __init__(self, size = 10000):
    self.size = size
    self.transitions = []

  def add(self, transition):
    if len(self.transitions) < self.size: 
      self.transitions.append(transition)
    else:
      self.transitions[self.current_index] = transition
      self.__increment_current_index()

  def length(self):
    return len(self.transitions)

  def get_batch(self, batch_size):
    return random.sample(self.transitions, batch_size)

  def __increment_current_index(self):
    self.current_index += 1
    if self.current_index >= self.size - 1: 
      self.current_index = 0

def calculate_target_values(model, target_model, state_transitions, discount_factor):
  states = []
  new_states = []
  for transition in state_transitions:
    states.append(transition.old_state)
    new_states.append(transition.new_state)

  new_states = np.array(new_states)

  q_values_new_state = get_multiple_q_values(model, new_states)
  q_values_new_state_target_model = get_multiple_q_values(target_model, new_states)
  
  targets = []
  for index, state_transition in enumerate(state_transitions):
    best_action = select_best_action(q_values_new_state[index])
    best_action_next_state_q_value = q_values_new_state_target_model[index][best_action]
    
    if state_transition.done:
      target_value = state_transition.reward
    else:
      target_value = state_transition.reward + discount_factor * best_action_next_state_q_value

    target_vector = [0] * outputs
    target_vector[state_transition.action] = target_value
    targets.append(target_vector)

  return np.array(targets)

def train_model(model, states, targets):
  print ("ok3")  
  model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)
  print("ok4")

def copy_model(model):
  backup_file = 'backup_'+str(uuid.uuid4())
  model.save(backup_file)
  new_model = load_model(backup_file, custom_objects={ 'masked_huber_loss': masked_huber_loss(0.0, 1.0) })
  shutil.rmtree(backup_file)
  return new_model


class AverageRewardTracker():
  current_index = 0

  def __init__(self, num_rewards_for_average=100):
    self.num_rewards_for_average = num_rewards_for_average
    self.last_x_rewards = []

  def add(self, reward):
    if len(self.last_x_rewards) < self.num_rewards_for_average: 
      self.last_x_rewards.append(reward)
    else:
      self.last_x_rewards[self.current_index] = reward
      self.__increment_current_index()

  def __increment_current_index(self):
    self.current_index += 1
    if self.current_index >= self.num_rewards_for_average: 
      self.current_index = 0

  def get_average(self):
    return np.average(self.last_x_rewards)


class FileLogger():

  def __init__(self, file_name='progress.log'):
    self.file_name = file_name
    self.clean_progress_file()

  def log(self, episode, steps, reward, average_reward):
    f = open(self.file_name, 'a+')
    f.write(f"{episode};{steps};{reward};{average_reward}\n")
    f.close()

  def clean_progress_file(self):
    if os.path.exists(self.file_name):
      os.remove(self.file_name)
    f = open(self.file_name, 'a+')
    f.write("episode;steps;reward;average\n")
    f.close()


replay_buffer_size = 200000
learning_rate = 0.001
regularization_factor = 0.001
training_batch_size = 128
training_start = 256
max_episodes = 10000
max_steps = 1000
target_network_replace_frequency_steps = 1000
model_backup_frequency_episodes = 100
starting_epsilon = 1.0
minimum_epsilon = 0.01
epsilon_decay_factor_per_episode = 0.995
discount_factor = 0.99
train_every_x_steps = 4

#Finally it all comes together in the main loop.

replay_buffer = ReplayBuffer(replay_buffer_size)
model = create_model(learning_rate, regularization_factor)
target_model = copy_model(model)
epsilon = starting_epsilon
step_count = 0
average_reward_tracker = AverageRewardTracker(100)
file_logger = FileLogger()


for episode in range(max_episodes):
  print(f"Starting episode {episode} with epsilon {epsilon}")

  episode_reward = 0
  #state = env.reset()
  #state=[[25,25]] # start point

  state=[[25,25]] 
  state = np.reshape(state, [1, 2])
  
  
  fraction_finished = 0.0
  state = np.append(state, fraction_finished)

  first_q_values = get_q_values(model, state)
  
  print(f"Q values: {first_q_values}")
  print(f"Max Q: {max(first_q_values)}")

  for step in range(1, max_steps + 1):
    step_count += 1
    q_values = get_q_values(model, state)
    action = select_action_epsilon_greedy(q_values, epsilon)
    #print ("action" action)
    #print ("state", state)
    data_from_env = env.camera(action, state)

    reward = data_from_env[0]
    new_state = [data_from_env[2], data_from_env[3],fraction_finished]  # x and y coordinate
    new_state = np.reshape(new_state, [1, 3])
    done =  data_from_env[1]            
    
    fraction_finished = (step + 1) / max_steps
    #new_state = np.append(new_state, fraction_finished)
    
    episode_reward += reward

    if step == max_steps:
      print(f"Episode reached the maximum number of steps. {max_steps}")
      done = True

    state_transition = StateTransition(state, action, reward, new_state, done)
    replay_buffer.add(state_transition)

    state = new_state

    if step_count % target_network_replace_frequency_steps == 0:
      print("Updating target model")
      target_model = copy_model(model)

    if replay_buffer.length() >= training_start and step_count % train_every_x_steps == 0:
      batch = replay_buffer.get_batch(batch_size=training_batch_size)
      print ("error")
      targets = calculate_target_values(model, target_model, batch, discount_factor) # error
      print ("error2")  
      states = np.array([state_transition.old_state for state_transition in batch])
      print ("ok5")
      train_model(model, states, targets)
      pritn ("ok6")  
    if done:
      break

  average_reward_tracker.add(episode_reward)
  average = average_reward_tracker.get_average()

  print(
    f"episode {episode} finished in {step} steps with reward {episode_reward}. "
    f"Average reward over last 100: {average}")

  if episode != 0 and episode % model_backup_frequency_episodes == 0:
    backup_file = f"model_{episode}.h5"
    print(f"Backing up model to {backup_file}")
    model.save(backup_file)

  epsilon *= epsilon_decay_factor_per_episode
  epsilon = max(minimum_epsilon, epsilon)

  data = pd.read_csv(file_logger.file_name, sep=';')

plt.figure(figsize=(20,10))
plt.plot(data['average'])
plt.plot(data['reward'])
plt.title('Reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.legend(['Average reward', 'Reward'], loc='upper right')
plt.show()




      
