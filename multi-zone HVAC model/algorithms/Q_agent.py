import numpy as np
import torch
from model_env_cooling import model
import pandas as pd
import random
import matplotlib.pyplot as plt
from weather_data import  *

class Q_learning(object):

    def __init__(self, gamma, learning_rate, epsilon, actions):

        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.actions = actions
        self.table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.epsilon_decay = 0.0005
    def choose_action(self, observation):
        self.check_observation(observation)
        if self.epsilon > 0.001:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0
        if np.random.uniform() > self.epsilon:
            state_action = np.array(self.table.loc[observation, :])
            b = np.where(state_action == np.max(state_action))[0]
            state_action2 = np.random.choice(b, 1)
        else:
            state_action2 = random.choice(self.actions)
        return state_action2

    def learn(self, s_, s, a, r, done):

        self.check_observation(s_)
        q_eval = self.table.loc[s, a]
        if done:
            q_target = r
        else:
            q_target = r + self.gamma * self.table.loc[s_, :].max()

        self.table.loc[s, a] += self.learning_rate * (q_target - q_eval)

    def check_observation(self, observation):

        if observation not in self.table.index:

            self.table = self.table.append(pd.Series(
                0 * len(self.actions),
                index=self.table.columns,
                name=observation,
            ))
        else:
            pass
    def save(self):
        self.table.to_csv("Q_table.csv")

actions = []
for i in np.arange(23, 27, 0.5):
    for j in np.arange(23, 27, 0.5):
        for l in np.arange(23, 27, 0.5):
            actions.append([i, j, l])
actions_ = [i for i in range(729)]
