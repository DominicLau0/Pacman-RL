import gymnasium as gym
import ale_py
import numpy as np
import numpy 
import matplotlib.pyplot as plt
import random
import torch
import time
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# numpy.set_printoptions(threshold=sys.maxsize)

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Normalise observation by 10000
        return obs / 10000

# Algorithm Lecture 9, Slide 6
class PacManSARSA():

    def train(self, episodes=100, render=False):
        # hyperparameters
        epsilon = 0.1
        alpha = 0.1
        gamma = 0.9
        d = 210*160*3 # state features --> 100,800
        # d = 32*32*3
        k = 9 # actions
        w = np.random.rand(d, k)

        env = gym.make('ALE/MsPacman-v5', render_mode=None)
        # print(f"Initial observation space: {env.observation_space.shape}")
        wrapped_env = ObservationWrapper(env)
        # wrapped_env = gym.wrappers.ResizeObservation(wrapped_env, 32)
        # print(f"New observation space: {env.observation_space.shape}")

        rewards = np.zeros(episodes)

        for i in range(episodes):
            print(f"Episode {i}")
            # initialize S and choose A
            obs, info = wrapped_env.reset()
            state = obs.flatten()
            episode_over = False

            q = w.T @ state
            randVal = random.random()

            # take random action
            if randVal < epsilon:
                action = env.action_space.sample()
            # take the greedy action
            else:
                action = np.argmax(q)

            while not episode_over:
                # take action A, observe R, S'
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                statePrime = obs.flatten()

                # choose A'
                qPrime = w.T @ statePrime

                randVal = random.random()
                # take random action
                if randVal < epsilon:
                    actionPrime = env.action_space.sample()
                # take the greedy action
                else:
                    actionPrime = np.argmax(qPrime)

                qSA = w[:, action] @ state
                qSAPrime = w[:, actionPrime] @ statePrime

                # print(f"state-action value: {qSA}")
                # print(f"Next state-action value: {qSAPrime}")

                rewards[i] += reward
                episode_over = terminated or truncated

                # update W
                if terminated:
                    change = alpha * (reward - qSA) * state
                else:
                    change = alpha * (reward + gamma * (qSAPrime) - qSA) * state

                w[:, action] += change

                action = actionPrime
                state = statePrime

        env.close()

        graphRewards(rewards, "SARSA Rewards")
        return w

def graphRewards(data, title):
    # Create new graph 
    plt.figure(1)
    plt.suptitle(title)
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.plot(data)
    
    fileName = title + ".png"
    plt.savefig(fileName)

if __name__ == '__main__':
    pacMan = PacManSARSA()
    start = time.time()
    pacMan.train(500)
    end = time.time()
    print(f"Took {end-start:.3f} seconds")