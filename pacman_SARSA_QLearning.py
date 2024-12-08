import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# random.seed(10)
# np.random.seed(seed=10)

class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):
        # Normalize observation by 255 to prevent overflow errors
        return obs / 255

# Algorithm Lecture 9, Slide 6
class FunctionApproximation():
    # Hyperparameters
    epsilon = 0.1
    gamma = 0.99
    step_size = 0.001

    def epsilonGreedy(self, env, x, W):
        randVal = random.random()

        if randVal < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(W.T @ x)
    
        return action

    def SARSA(self, env, episodes=50000):
        d = 128
        k = env.action_space.n
        # initialize W randomly
        W = np.random.rand(d, k)
        
        rewards = np.zeros(episodes)

        for i in range(episodes):
            # Initialize S and choose A
            s, info = env.reset()
            a = self.epsilonGreedy(env, s, W)
        
            episode_over = False
            while not episode_over:
                
                # Take action A, Observe R, S'
                sPrime, reward, terminated, truncated, info = env.step(a)
                
                # Choose A'
                aPrime = self.epsilonGreedy(env, sPrime, W)
                
                # update W
                if terminated:
                    error = reward - (W[:, a] @ s)
                else:
                    error = reward + (self.gamma * (W[:, aPrime] @ sPrime)) - (W[:, a] @ s)
                
                W[:, a] = W[:, a] + (self.step_size * error * s)
                
                rewards[i] += reward
                
                # Proceed to next state of env
                s = sPrime
                a = aPrime
                episode_over = terminated or truncated

        if episodes > 1000 and episodes % 1000 == 0:
            # get runtime average of rewards (avg rewards of every 1000 games)
            mean = np.mean(rewards.reshape(-1, 1000), axis=1)
        else:
            mean = rewards
        
        graphRewards(mean, "FA SARSA rewards")

        return W
    
    def QLearning(self, env, episodes=50000):
        d = 128
        k = env.action_space.n

        # initialize parameters W
        W = np.random.rand(d, k)

        rewards = np.zeros(episodes)

        for i in range(episodes):
            # initialize S
            s, info = env.reset()

            episode_over = False
            while not episode_over:

                a = self.epsilonGreedy(env, s, W)
                
                # Take action A, observe R, S'
                sPrime, reward, terminated, truncated, info = env.step(a)

                # update W
                if terminated:
                    error = reward - (W[:, a] @ s)
                else:
                    error = reward + (self.gamma * np.max(W.T @ sPrime)) - (W[:, a] @ s)
                
                W[:, a] = W[:, a] + (self.step_size * error * s)
                
                rewards[i] += reward
                
                # Proceed to next state of env
                s = sPrime
                episode_over = terminated or truncated

        if episodes > 1000 and episodes % 1000 == 0:
            # get runtime average of rewards (avg rewards of every 1000 games)
            mean = np.mean(rewards.reshape(-1, 1000), axis=1)
        else:
            mean = rewards
        
        graphRewards(mean, "FA QL rewards")

        return W

    def test(self, w, episodes=5):

        env = gym.make('ALE/MsPacman-v5', obs_type="ram", render_mode="human")
        wrapped_env = ObservationWrapper(env)

        d = 128 # state features
        k = env.action_space.n # actions
        w = np.random.rand(d, k)

        rewards = np.zeros(episodes)

        for i in range(episodes):
            print(f"Game {i+1}")

            # initialize S and choose A
            state, info = wrapped_env.reset()
            
            episode_over = False

            while not episode_over:

                # always greedy
                action = np.argmax(w.T @ state)

                # take action A, observe R, S'
                statePrime, reward, terminated, truncated, info = wrapped_env.step(action)

                state = statePrime
                
                rewards[i] += reward
                episode_over = terminated or truncated

        env.close()

        graphRewards(rewards, "Test Rewards")

def graphRewards(data, title):
    plt.figure()
    plt.suptitle(title)
    plt.ylabel('Rewards')
    plt.plot(data)
    
    fileName = title + ".png"
    plt.savefig(fileName)

if __name__ == '__main__':
    pacMan = FunctionApproximation()
    
    env = gym.make('ALE/MsPacman-v5', obs_type="ram", render_mode=None)
    env = ObservationWrapper(env)

    start = time.time()
    w = pacMan.SARSA(env, 50000)
    # w = pacMan.QLearning(env, 50000)
    end = time.time()
    print(f"Took {end-start:.3f} seconds")
    env.close()
    
    input("Press the Enter key to continue: ") 
    pacMan.test(w)