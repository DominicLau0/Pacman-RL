import numpy as np

from collections import deque
import random
import os

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Learned Deep Q Learning from following links
# Official pytorch Deep Q Network Tutorial:
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# Deep Q Learning in Flappy Bird:
#   https://www.youtube.com/playlist?list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi

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
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.fc1 = nn.Linear(33280, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x    

def pacmanDQL(env, gamma=0.99, alpha=0.001, epsilon=1, epsilon_decay = 0.9995, epsilon_min=0.05, max_episode=1000, is_training=True):

    def choose_action():
        if is_training and random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_network(state.unsqueeze(dim=0)).squeeze().argmax().item()
        action = torch.tensor(action, dtype=torch.int64, device=device)

        return action
    
    def normalize(image):
        image = image / 255.0
        
        return torch.tensor(image, dtype=torch.float, device=device).unsqueeze(dim=0)
    
    def optimize_model(mini_batch, policy_network, target_network):
        # Seperate batch into states, actions, etc.
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        
        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            target_q = reward + (1-terminations) * gamma * target_network(new_states).max(dim=1)[0]

        current_q = policy_network(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute the loss for the whole minibatch
        loss = loss_fn(current_q, target_q)

        # Optimiize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

    # Get action size
    action_size = env.action_space.n

    # Create policy network
    policy_network = DQN(action_size).to(device)

    # Optimizer and loss function
    loss_fn = nn.MSELoss()

    # Save model in folder
    model_save_path = os.path.join("DQL_saved_model", "best_model.pth")

    if is_training:
        replay_memory = ReplayMemory(100000)

        # Creates the target network
        target_network = DQN(action_size).to(device)
        target_network.load_state_dict(policy_network.state_dict())

        step_count = 0
        optimizer = optim.AdamW(policy_network.parameters(), lr=alpha, amsgrad=True)

        best_reward = -float('inf')
    
    else:
        policy_network.load_state_dict(torch.load(model_save_path))
        policy_network.eval()

    for episode in range(max_episode):
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

            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            if is_training:
                replay_memory.push((state, action, next_state, reward, terminated))
                step_count += 1

            # Sync the target network to the policy network after 32 steps
            if len(replay_memory) > 32:
                mini_batch = replay_memory.sample(32)

                optimize_model(mini_batch, policy_network, target_network)

                if step_count > 100:
                    target_network.load_state_dict(policy_network.state_dict())
                    step_count = 0

            state = next_state

        print(episode_reward)

        if is_training and episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_network.state_dict(), model_save_path)
            print(f"Saved model, Reward: {best_reward}")
    
    return

def test():
    return