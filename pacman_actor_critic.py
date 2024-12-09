import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import sys
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        return obs / 255

class PacManActorCritic:
    def train(self, episodes=100, render=False):
        # Hyperparameters
        gamma = 0.99
        alpha_actor = 0.0005
        alpha_critic = 0.001
        epsilon = 0.1
        d = 210 * 160 * 3
        k = 9  # Number of actions

        # Initialize weights
        actor_weights = np.random.rand(d, k)
        critic_weights = np.random.rand(d)

        env = gym.make('ALE/MsPacman-v5', render_mode=None)
        wrapped_env = ObservationWrapper(env)

        rewards = np.zeros(episodes)

        for i in range(episodes):
            print(f"Episode {i+1}")
            obs, info = wrapped_env.reset()
            state = obs.flatten()
            episode_over = False
            total_reward = 0

            # Choose initial action
            q = actor_weights.T @ state
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q)

            while not episode_over:
                # Calculate value for current state
                value = critic_weights @ state

                # Take action and observe new state, reward
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                state_prime = obs.flatten()

                # Calculate policy probabilities (softmax)
                q = actor_weights.T @ state
                policy = np.exp(q) / np.sum(np.exp(q))

                # Choose next action
                q_prime = actor_weights.T @ state_prime
                if random.random() < epsilon:
                    action_prime = env.action_space.sample()
                else:
                    action_prime = np.argmax(q_prime)

                # Calculate TD error
                next_value = critic_weights @ state_prime
                td_error = reward + gamma * next_value - value

                # Update critic weights
                critic_weights += alpha_critic * td_error * state

                # Update actor weights
                for a in range(k):
                    if a == action:
                        actor_weights[:, a] += alpha_actor * td_error * (1 - policy[a]) * state
                    else:
                        actor_weights[:, a] -= alpha_actor * td_error * policy[a] * state

                state = state_prime
                action = action_prime
                total_reward += reward
                episode_over = terminated or truncated

            rewards[i] = total_reward

        env.close()
        graph_rewards(rewards, "10000 Actor-Critic Rewards")
        average_graphRewards(rewards)

        return actor_weights, critic_weights

def graph_rewards(data, title):
    # Create new graph
    plt.figure(1)
    plt.suptitle(title)
    plt.xlabel("Episodes")
    plt.ylabel("Total Rewards")
    plt.plot(data)

    file_name = title + ".png"
    plt.savefig(file_name)


def average_graphRewards(rewards):
    rewards = np.mean(np.array(rewards).reshape(-1, 100), axis=1)

    plt.figure()
    plt.suptitle("Average Rewards for Actor-Critic")
    plt.xlabel('Episodes (averaged over every 100 games)')
    plt.ylabel('Average Rewards')
    plt.plot(rewards, label="Rewards")
    
    fileName = "Actor-Critic_Averaged" + ".png"
    plt.savefig(fileName)
    return