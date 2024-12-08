import numpy as np
import matplotlib.pyplot as plt

from collections import deque
import random
import os
import cv2

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# Learned Deep Q Learning from following links:
# Official pytorch Deep Q Network Tutorial:
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Deep Q Learning in Flappy Bird:
#   https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi

# Honestly great paper that taught me Deep Q Learning/Network:
# Playing Atari with Deep Reinforcement Learning
#   https://arxiv.org/pdf/1312.5602

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def plot(rewards):
    plt.figure()
    plt.suptitle("Rewards for DQL")
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.plot(rewards, label="Rewards")
    
    fileName = "DQL" + ".png"
    plt.savefig(fileName)
    return

def average_plot(rewards):
    plt.figure()
    plt.suptitle("Average Rewards for DQL")
    plt.xlabel('Episodes (averaged over every 5 games)')
    plt.ylabel('Average Rewards')
    plt.plot(rewards, label="Rewards")
    
    fileName = "DQL_Averaged" + ".png"
    plt.savefig(fileName)
    return

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
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1)

        self.fc1 = nn.Linear(5632, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x    

def pacmanDQL(env):

    def choose_action():
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_network(state.unsqueeze(dim=0)).squeeze().argmax().item()
        action = torch.tensor(action, dtype=torch.int64, device=device)

        return action
    
    def normalize(image):
        image = cv2.resize(image, (110, 84), interpolation=cv2.INTER_AREA)
        image = image / 255.0
        
        return torch.tensor(image, dtype=torch.float, device=device).unsqueeze(dim=0)
    
    def optimize_model():
        if len(replay_memory) < 32:
            return
        
        transitions = replay_memory.sample(32)

        # Seperate batch into states, actions, etc.
        states, actions, new_states, rewards, terminations = zip(*transitions)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = rewards + (1-terminations) * gamma * target_network(new_states).max(dim=1)[0]

        current_q = policy_network(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Optimizer and loss function
        loss = nn.MSELoss()
        optimizer = optim.AdamW(policy_network.parameters(), lr=alpha, amsgrad=True)
        loss = loss(current_q, target_q)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return

    # Initialize some stuff
    action_size = env.action_space.n

    episodes = 200
    gamma = 0.99
    alpha = 0.0005

    epsilon = 1
    epsilon_decay = 0.99995
    epsilon_min=0.05

    replay_memory = ReplayMemory(100000)

    stored_rewards = []

    step_count = 0

    # Create policy network
    policy_network = DQN(action_size).to(device)

    # Creates the target network
    target_network = DQN(action_size).to(device)
    target_network.load_state_dict(policy_network.state_dict())

    for episode in range(episodes):
        state, info = env.reset()
        state = normalize(state)

        terminated = False
        episode_reward = 0.0

        while not terminated:
            # Select action using epsilon greedy policy
            action = choose_action()

            # Take action and get next state and reward
            next_state, reward, terminated, _, info = env.step(action)

            next_state = normalize(next_state)            
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            episode_reward += reward

            epsilon = max(epsilon * epsilon_decay, epsilon_min) #Epsilon decay

            replay_memory.push((state, action, next_state, reward, terminated))
            step_count += 1

            # Sync the target network to the policy network after 32 steps
            optimize_model()

            if step_count > 100:
                target_network.load_state_dict(policy_network.state_dict())
                step_count = 0

            state = next_state

        # Keep track of the stored rewards
        stored_rewards.append(episode_reward.item())
        print(episode_reward)
    
    return np.array(stored_rewards)