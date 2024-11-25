import gymnasium as gym
import ale_py

from pacmanEnv import PacMan
from pacman_dql import pacmanDQL

def createEnv():
    # env = PacMan()
    env = gym.make('ALE/MsPacman-v5', render_mode=None)
    executePacmanDQL(env)
    env.close()

def executePacmanDQL(env):
    # Hyperparameters

    pacmanDQL(env)

if __name__ == "__main__":
    createEnv()