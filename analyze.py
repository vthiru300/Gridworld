import pickle
import numpy as np
import matplotlib.pyplot as plt


def analyze_convergence():
    filehandler = open('convergence', 'rb')
    data = []
    number_of_episodes = 1

    while True:
        try:
            [episode, instant, reward] = pickle.load(filehandler)
            entry = [episode, instant, reward]
            data.append(entry)
            number_of_episodes = episode + 1
        except EOFError:
            break

    filehandler.close()

    average_reward = np.zeros(number_of_episodes)
    average_time = np.zeros(number_of_episodes)
    average_episode_reward = 0
    average_episode_time = 0
    episode_count = 0

    for item in data:
        episode, instant, reward = item

        if episode == episode_count:
            average_episode_reward += reward
            average_episode_time = instant
        else:
            average_reward[episode_count] = average_episode_reward / (instant + 1)
            average_time[episode_count] = average_episode_time + 1
            average_episode_reward = reward
            average_episode_time = 0
            episode_count += 1

    x = range(number_of_episodes)
    #plt.rcParams['figure.figsize'] = [10, 10]
    plt.plot(x, average_reward)
    plt.xlabel('Episode')
    plt.ylabel('Reward Accrued')
    plt.title('Reward per Episode')
    plt.savefig('QLreward.eps', dpi=2400)
    plt.clf()

    plt.plot(x, average_time)
    plt.xlabel('Episode')
    plt.ylabel('Number of Steps')
    plt.title('Steps per Episode')
    plt.savefig("QLtime.eps", dpi=2400)
    plt.clf()
