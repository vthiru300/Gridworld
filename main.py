

from gridworld import *


if __name__ == '__main__':
    #runtime = 100
    filename_state = 'state'
    filename_qtable = 'qtable'
    filename_results = 'results'
    number_of_robots = 1
    number_of_interestpoints = 1
    dimension = 4
    env = GridWorld(dimension)

    number_of_episodes = 500
    env.train(number_of_episodes)
    env.visualize_training(filename_state, number_of_episodes)
    env.close()
