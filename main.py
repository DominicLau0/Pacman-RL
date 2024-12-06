import gymnasium as gym
import ale_py
import time

# from pacmanEnv import PacMan
from pacman_dql import pacmanDQL
from pacman_SARSA import FunctionApproximation, ObservationWrapper

def createDQLEnv():
    # env = PacMan()
    env = gym.make('ALE/MsPacman-v5', render_mode=None, obs_type="grayscale")
    pacmanDQL(env)
    env.close()

def runDQL():
    createDQLEnv()

def runFunctionApproximation():
    pacMan = FunctionApproximation()
    
    env = gym.make('ALE/MsPacman-v5', obs_type="ram", render_mode=None)
    env = ObservationWrapper(env)

    start = time.time()

    # SARSA
    w = pacMan.SARSA(env, 50000)

    # Q-Learning
    # w = pacMan.QLearning(env, 50000)
    
    end = time.time()
    print(f"Took {end-start:.3f} seconds")
    env.close()

    # uncomment if testing
    # input("Press the Enter key to continue: ") 
    # pacMan.test(w)


if __name__ == "__main__":
    ### DQL (Dominic) ###
    # runDQL()
    
    ### Function Approximation (Sara) ###
    runFunctionApproximation()