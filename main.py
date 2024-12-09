import gymnasium as gym
import ale_py
import time
import numpy as np

from pacman_dql import pacmanDQL, plot, average_plot
from pacman_SARSA_QLearning import FunctionApproximation, ObservationWrapper
from pacman_actor_critic import PacManActorCritic

def SARSA():
    pacMan = FunctionApproximation()
    
    env = gym.make('ALE/MsPacman-v5', obs_type="ram", render_mode=None)
    env = ObservationWrapper(env)

    start = time.time()
    w = pacMan.SARSA(env, 50000)    
    end = time.time()

    print(f"Took {end-start:.3f} seconds")
    env.close()

    # Uncomment if testing
    # input("Press the Enter key to continue: ") 
    # pacMan.test(w)

def QLearning():
    pacMan = FunctionApproximation()
    
    env = gym.make('ALE/MsPacman-v5', obs_type="ram", render_mode=None)
    env = ObservationWrapper(env)

    start = time.time()
    w = pacMan.QLearning(env, 50000)
    end = time.time()

    print(f"Took {end-start:.3f} seconds")
    env.close()

    # Uncomment if testing
    # input("Press the Enter key to continue: ") 
    # pacMan.test(w)

def Actor_Critic():
    pacman_ac = PacManActorCritic()

    start = time.time()
    pacman_ac.train(episodes=10000)
    end = time.time()

    print(f"Took {end - start:.3f} seconds")

def DQL():
    env = gym.make('ALE/MsPacman-v5', obs_type="grayscale", render_mode=None)

    reward = pacmanDQL(env)

    plot(reward)
    average_plot(np.mean(np.array(reward).reshape(-1, 5), axis=1))

    env.close()

if __name__ == "__main__": 
    # Execute SARSA
    SARSA()

    # Execute Q-Learning
    QLearning()

    # Execute Actor-Critic
    Actor_Critic()

    # Execute Deep-Q Learning
    DQL()