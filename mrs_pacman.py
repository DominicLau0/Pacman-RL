import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

def run():
    gym.register_envs(ale_py)

    env = gym.make('ALE/MsPacman-v5', render_mode="human")

    # if GPU is to be used
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )

    obs, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

        env.close()

def train(nTrials=1000):
    env = gym.make('ALE/MsPacman-v5')
    # num_states = env.observation_space.n
    num_actions = env.action_space.n

    # print(num_states)
    print(num_actions)
    print(env.action_space)

if __name__ == '__main__':
    run()
    train()