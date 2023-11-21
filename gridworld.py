import gym
import random
import numpy as np
import pickle
import matplotlib.pyplot as plt
import math

from torch import nn
import torch
import torch.nn.functional as F
from collections import deque
import itertools


from gym import *
from gym.spaces import Discrete, MultiDiscrete, flatdim

from imagemaker import *

class GridWorld(Env):
    def __init__(self, dim):
        super(GridWorld, self).__init__()

        # Hyperparameters
        self.alpha = 0.3
        self.gamma = 0.6

        self.max_epsilon = 0.9
        self.min_epsilon = 0.0025
        self.decay = 0.998
        self.epsilon = 0.9

        self.batch_size = 32
        self.buffer_size = 50000
        self.min_replay_size = 1000
        self.target_update_freq = 1000

        self.dim = dim
        self.reward = 0
        self.boundarycount = 0

        imagefolder = 'Images'
        self.empty_imagefolder(imagefolder)

        position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])
        self.robot = Robot(1, position, self.dim)
        print('Robot ID = {}: Robot Starting Position = [{}, {}]'.format(1, position[0], position[1]))

        position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])
        self.interestpoint=InterestPoint(1,position, self.dim)
        print('Interest Point ID = {}: Starting Position = [{}, {}]'.format(1, position[0], position[1]))

        self.observation_space = self.build_observation_space()

        self.number_of_actions = 4
        self.list_of_actions = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.rew_buffer = deque([0.0], maxlen=100)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('running on the GPU')
        else:
            device = torch.device('cpu')
            print('running on the CPU')

        self.online_net = Network(4, self.number_of_actions).to(device)
        self.target_net = Network(4, self.number_of_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=5e-4)

    def build_observation_space(self):
        return np.concatenate((self.robot.position,self.interestpoint.position), axis=None)

    def reset(self):
        self.robot.position = np.array([random.randint(0, self.dim-1), random.randint(0, self.dim-1)])
        self.interestpoint.visited = False
        self.interestpoint.position = np.array([random.randint(0, self.dim - 1), random.randint(0, self.dim - 1)])

    def compute_reward(self):
        is_interestpoint = np.zeros([self.dim, self.dim], dtype=bool)
        reward = 0

        is_interestpoint[tuple(self.interestpoint.position)] = True

        if is_interestpoint[tuple(self.robot.position)]:
            reward += 20
        elif self.robot.boundary:
            reward -= 10
            self.robot.boundary = False
        else:
            reward -= 1

        self.reward = reward

    def random_action(self):
        return np.random.choice(range(self.action_space))

    def epsilon_greedy_action(self, obs):
        if random.uniform(0, 1) < self.epsilon:
            return self.random_action()
        else:
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        return self.online_net.act(obs)

    def terminal_condition(self):
        is_interestpoint = np.zeros([self.dim, self.dim], dtype=bool)
        is_interestpoint[tuple(self.interestpoint.position)] = True
        #is_occupied[tuple(robot.position)] = True


        if self.is_interestpoint[tuple(self.robot.position)]:
            return True
        else:
            return False

    def step(self, action):
        self.robot.move(action)
        self.compute_reward()

    def train(self, number_of_episodes):
        filehandler_state = open('state', 'wb')
        filehandler_convergence = open('convergence', 'wb')

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        #Initialize Replay Buffer
        self.reset()
        obs = self.observation_space
        for _ in range(self.min_replay_size):
            action = self.random_action()
            self.step(action)
            reward = self.reward
            self.observation_space = self.build_observation_space()
            new_obs = self.observation_space
            done = self.terminal_condition()
            transition = (obs, action, reward, done, new_obs)
            #print('Observation = {}: Action = {}: New Observation = {}'.format(obs, actions, new_obs))
            self.replay_buffer.append(transition)
            obs = new_obs

            if done:
                self.reset()
                obs = self.observation_space

        step = 0

        for episode in range(number_of_episodes):
            self.reset()
            obs = self.observation_space
            reward = 0
            episode_reward = 0

            self.epsilon = max(self.min_epsilon, self.epsilon, self.decay)

            while not self.terminal_condition():
                action = self.epsilon_greedy_action(obs)

                if episode == number_of_episodes - 1:
                    pickle.dump([episode, step, self.robot, self.interestpoint], filehandler_state)

                self.step(actions)


                reward = self.reward
                self.observation_space = self.build_observation_space()
                new_obs = self.observation_space
                done = self.terminal_condition()

                transition = (obs, action, reward, done, new_obs)
                self.replay_buffer.append(transition)
                obs = new_obs

                transitions = random.sample(self.replay_buffer, self.batch_size)

                episode_reward += reward

                if done:
                    self.rew_buffer.append(episode_reward)
                    episode_reward = 0

                obses = np.asarray([t[0] for t in transitions])
                actions = np.asarray([t[1] for t in transitions])
                rews = np.asarray([t[2] for t in transitions])
                dones = np.asarray([t[3] for t in transitions])
                new_obses = np.asarray([t[4] for t in transitions])

                obses_t = torch.as_tensor(obses, dtype=torch.float32).to(device)
                actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
                rews_t = torch.as_tensor(rews, dtype=torch.float32).unsqueeze(-1).to(device)
                dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
                new_obses_t = torch.as_tensor(new_obses, dtype=torch.float32).to(device)

                target_q_values = self.target_net(new_obses_t)
                #print('Target Q Values = ', target_q_values)
                max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
                targets = rews_t + self.gamma * (1 - dones_t) * max_target_q_values

                q_values = self.online_net(obses_t)
                action_q_values = torch.gather(input=q_values, dim=1, index=actions_t)

                loss = nn.functional.smooth_l1_loss(action_q_values, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                step += 1

                if step % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.online_net.state_dict())

                if step % 1000 == 0:
                    print()
                    print('Step', step)
                    print('Episode', episode)
                    print('Avg Rew', np.mean(self.rew_buffer))
                    print('Epsilon', self.epsilon)

            pickle.dump([episode, step, self.robot, self.interestpoint], filehandler_state)

        filehandler_convergence.close()
        filehandler_state.close()

    def evaluate(self, filename_qtable):
        self.reset()
        step = 0
        episode = 0

        filehandler_qtable = open(filename_qtable, 'rb')
        filehandler_evaluation = open('evaluation', 'wb')
        [self.qtable, interestpoints, robot, dim] = pickle.load(filehandler_qtable)

        while not self.terminal_condition():
            state_index = self.flatten_multidimensional_index()
            action_index = self.greedy_action(state_index)
            actions = self.n_to_base(action_index, self.number_of_actions)
            self.step(actions)
            step += 1

            pickle.dump([episode, step, self.robot, self.interestpoint], filehandler_evaluation)

        filehandler_qtable.close()
        filehandler_evaluation.close()

    def visualize_training(self, filename, number_of_episodes):
        filehandler = open(filename, 'rb')
        while True:
            try:
                [episode, step, robot, interestpoint] = pickle.load(filehandler)
                if episode == number_of_episodes-1:
                    image_maker(step, robot, interestpoint, self.dim)
            except EOFError:
                break

        movie_maker()

    def visualize(self, filename):
        filehandler = open(filename, 'rb');
        while True:
            try:
                [episode, instant, robot, interestpoint] = pickle.load(filehandler)
                image_maker(instant, robot, interestpoint, self.dim)
            except EOFError:
                break

        movie_maker()

    def empty_imagefolder(self, imagefolder):
        for filename in os.listdir(imagefolder):
            file_path = os.path.join(imagefolder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s.  Reason: %s' % (file_path, e))


class Point(object):
    def __init__(self, name, position, dim):
        self.name = name
        self.dim = dim
        self.set_position(position)

    def set_position(self, position):
        temp_x, temp_y = position
        x_min, x_max, y_min, y_max = [0, self.dim-1, 0, self.dim-1]
        x = self.clamp(temp_x, x_min, x_max)
        y = self.clamp(temp_y, y_min, y_max)
        self.position = np.array([x, y])

    def get_position(self):
        return self.position

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)


class Robot(Point):
    def __init__(self, name, position, dim):
        super(Robot, self).__init__(name, position, dim)

        self.action_key = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        self.boundary = False

    def move(self, action):
        heading = 0
        if action == 0:
            heading = np.array([0, 1])
        elif action == 1:
            heading = np.array([0, -1])
        elif action == 2:
            heading = np.array([-1, 0])
        elif action == 3:
            heading = np.array([1, 0])

        position = self.position
        position += heading
        self.set_position(position)
        if np.linalg.norm(position-self.position) > 0.01:
            self.boundary = True

    def get_action_meanings(self, action):
        return self.action_key[action]


class InterestPoint(Point):
    def __init__(self, name, position, dim):
        super(InterestPoint, self).__init__(name, position, dim)

        self.visited = False


class Network(nn.Module):
    def __init__(self, observation_size, action_space_size):
        super().__init__()

        in_features = int(observation_size)

        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 4)

        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def act(self, obs):
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
        q_values = self(obs_t.unsqueeze(0))
        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
