from env_shower import ShowerEnv
from stable_baselines3 import DQN
import numpy as np
import os

if __name__ == '__main__':
    env = ShowerEnv()
    if not os.path.isfile('model.pt'):
        model = DQN(
            'MlpPolicy',
            env,
            verbose=1
        )
        model.learn(total_timesteps=500000)
        model.save('model.pt')
        del model  # remove to demonstrate saving and loading

    model = DQN.load('model.pt')
    rewards = []
    successes = []
    distances = []
    for episode in range(100):
        obs, tot_reward, done, info = env.reset(), 0, False, {}
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            tot_reward += reward
        rewards.append(tot_reward)
        successes.append(info['success'])
        distances.append(info['distance'])
    print(f'reward: {np.mean(rewards)} +- {np.std(rewards)}')
    print(f'final distance: {np.mean(distances)} +- {np.std(distances)}')
    print(f'success: {np.mean(successes)} +- {np.std(successes)}')
