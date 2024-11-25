import numpy as np

from collections import deque
import random
import itertools

import torch
from torch import nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayMemory():
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, observation_size, action_size):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(observation_size, 256)
        self.layer2 = nn.Linear(256, action_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        return self.layer2(x)

def pacmanDQL(env, gamma=0.9, alpha=0.1, epsilon=1, epsilon_decay = 0.9995, epsilon_min=0.05, max_episode=1000, is_training=True):

    def choose_action():
        if is_training and random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                action = policy(state.unsqueeze(dim=0)).squeeze().argmax()
        return action
                
    # Get state and action size
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    policy = DQN(state_size, action_size).to(device)

    rewards_per_episode = []
    epsilon_history = []

    if is_training:
        replay_memory = ReplayMemory(100000)

    for episode in range(max_episode):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)

        terminated = False
        episode_reward = 0.0

        while not terminated:
            # Select action using epsilon greedy policy
            action = choose_action()

            next_state, reward, terminated, _, info = env.step(action.item())

            next_state = torch.tensor(next_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            episode_reward += reward

            if is_training:
                replay_memory.push((state, action, next_state, reward, terminated))

            state = next_state

        rewards_per_episode.append(episode_reward)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        epsilon_history.append(epsilon)

        print(episode_reward)
    
    return rewards_per_episode, epsilon_history