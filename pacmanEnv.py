import gymnasium as gym
import pygame
import numpy as np

class PacMan(gym.Env):
    # TODO: initialize environment, self.observation_space, self.action_space etc.
    def __init__(self):
        return
    
    # TODO: reset to a new game of pacman
    def reset(self):
        return
    
    # TODO: take the action and update the state
    def step(self, action):
        state = None
        reward = 0
        terminated = False
        return state, reward, terminated
    
    # TODO: Use pygame to display the current state
    def render(self):
        return 

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()