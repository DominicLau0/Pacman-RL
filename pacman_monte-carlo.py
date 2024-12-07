# TODO: COMMENTS

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

def mapStateToIndex(dictionary, state):
    key = state.tobytes()
    if key not in dictionary:
        dictionary[key] = len(dictionary)
    return dictionary[key]

def appendToReturns(returns, state_action_pair, G):
        # Each state/action pair key stores an array of values
        # No duplicates are allowed
        if state_action_pair not in returns:
            returns[state_action_pair] = [G]
        else:
            returns[state_action_pair].append(G)
        return returns

class PacManMonteCarlo:
    
    def train(self, epsilon, step_size):
        # Gym environment
        env = gym.make('ALE/MsPacman-v5', render_mode=None, obs_type="grayscale")
        # State/action sizes
        #state_size = 210*160
        state_size = 500000 # TODO
        action_size = env.action_space.n

        # TODO: variables
        policy = np.full((state_size, action_size), epsilon / action_size)
        for s in range(state_size):
             random_init_action = np.random.randint(action_size)
             policy[s][random_init_action] = 1 - epsilon + epsilon / action_size

        state_dict = {}
        returns    = {}
        Q          = np.zeros((state_size, action_size))
        total_rewards = []

        # TODO
        episodes = 300
        for i in range(episodes):
            states  = []
            actions = []
            rewards = []

            # TODO: Generate episodes
            obs, _ = env.reset()
            episode_over = False
            prev_state = mapStateToIndex(state_dict, obs.flatten())
            while (not episode_over):
                action = np.random.choice(np.arange(action_size), p=policy[prev_state])
                obs, reward, terminated, truncated, _ =  env.step(action)
                state = mapStateToIndex(state_dict, obs.flatten())

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                prev_state = state
                episode_over = terminated or truncated
            
            G = 0
            total_rewards.append(np.sum(rewards))

            # TODO: loop for each step of the episode, T-1 -> 0
            for i in reversed(range(len(states)-1)):
                S = states[i]
                A = actions[i]
                R = rewards[i+1]
                state_action_pair = (S,A)

                G = R + step_size * G
                appendToReturns(returns, state_action_pair, G)
                Q[S][A] = np.mean(returns[state_action_pair])

                a_star = np.argmax(Q[S])
                for a in range(action_size):
                    if (a_star == a):
                        policy[S][a] = 1 - epsilon + epsilon / action_size
                    else:
                        policy[S][a] = epsilon / action_size

        return total_rewards

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def graphRewards(data, title):
    # Create new graph 
    #data = moving_average(data, 5)
    plt.figure(1)
    plt.suptitle(title)
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.plot(data)
    plt.show()


def main():
    # TODO: refactor
    pacMan = PacManMonteCarlo()
    start = time.time()
    a = pacMan.train(epsilon=0.05, step_size=0.1)
    b = pacMan.train(epsilon=0.05, step_size=0.1)
    c = pacMan.train(epsilon=0.05, step_size=0.1)
    d = pacMan.train(epsilon=0.05, step_size=0.1)
    e = pacMan.train(epsilon=0.05, step_size=0.1)
    a = moving_average(a, 10)
    b = moving_average(b, 10)
    c = moving_average(c, 10)
    d = moving_average(d, 10)
    e = moving_average(e, 10)
    data = np.mean( np.array([ a, b, c, d, e ]), axis=0 )
    graphRewards(data, "TODO")
    end = time.time()
    print(f"Took {end-start:.3f} seconds")

if __name__ == '__main__':
    main()