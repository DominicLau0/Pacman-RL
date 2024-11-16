import gymnasium as gym
import ale_py
import numpy as np
import numpy 
import matplotlib.pyplot as plt
import random
import torch

class PacManSARSA():

    def train(self, episodes=100, render=False):
        # hyperparameters
        epsilon = 1
        decay = 0.00001
        alpha = 0.1
        gamma = 0.8
        d = 210*160*3 # state features --> 100,800
        k = 9 # actions
        w = np.random.rand(d, k).astype(numpy.float64)
        
        env = gym.make('ALE/MsPacman-v5', render_mode=None)
        
        rewards = np.zeros(episodes)
        counter = 0
        
        for i in range(episodes):
            counter += 1
            print(f"Episode {counter}, epsilon={epsilon}")
            obs, info = env.reset()
            state = obs.flatten()
            episode_over = False
            randVal = random.random()
                
            # take random action
            if randVal < epsilon:
                action = env.action_space.sample()
            # take the greedy action
            else:
                action = np.argmax(q)
            
            while not episode_over:
                q = w.T @ state
                qSum = np.sum(q)
                q = np.divide(q, qSum) # divide by sum to avoid overflow errors

                obs, reward, terminated, truncated, info = env.step(action)
                statePrime = obs.flatten()

                # don't decay epsilon past 10%
                if(epsilon < 0.1):
                    epsilon = 0.1

                randVal = random.random()
                # take random action
                if randVal < epsilon:
                    actionPrime = env.action_space.sample()
                # take the greedy action
                else:
                    actionPrime = np.argmax(q)
                
                qSA = w.T[action] @ state
                qSAPrime = w.T[actionPrime] @ statePrime
                
                qSA = np.divide(qSA, qSum) # divide by sum to avoid overflow errors
                
                qSAPrime = np.divide(qSAPrime, qSum) # divide by sum to avoid overflow errors
                
                rewards[i] += reward
                episode_over = terminated or truncated

                values = torch.tensor(w.T[action])
                semiGradient = torch.gradient(values)
                semiGradient = semiGradient[0].numpy()

                change = alpha * (reward + gamma * (qSAPrime) - qSA) * semiGradient
                w.T[action] = w.T[action] + change

                epsilon -= decay
                action = actionPrime

        env.close()

        graphRewards(rewards, "SARSA_Rewards")

def graphRewards(data, title):
    # Create new graph 
    plt.figure(1)
    plt.suptitle(title)
    plt.xlabel('Timestep')
    plt.ylabel('Rewards')
    plt.plot(data)
    
    fileName = title + ".png"
    plt.savefig(fileName)

# if __name__ == '__main__':
#     pacMan = PacManSARSA()
#     pacMan.train(20)