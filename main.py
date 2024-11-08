from pacmanEnv import PacMan
from pacman_dql import pacmanDQL

def createEnv():
    env = PacMan()
    executePacmanDQL(env)
    env.close()

def executePacmanDQL(env):
    pacmanDQL(env)

if __name__ == "__main__":
    createEnv()