import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from norm import *

class Actor_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.normal_(0.01)
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.normal_(0.01)
        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.normal_(0, 0.01)
        self.fc3.bias.data.normal_(0.01)

    def forward(self, x):

        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = torch.tanh(self.fc3(out))
        return out

class Critic_Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.normal_(0.01)
        self.fc2 = nn.Linear(action_dim, 128)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.normal_(0.01)
        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data.normal_(0, 0.01)
        self.fc3.bias.data.normal_(0.01)
        self.fc4 = nn.Linear(64, 1)
        self.fc4.weight.data.normal_(0, 0.01)
        self.fc4.bias.data.normal_(0.01)

    def forward(self, s, a):

        x = self.fc1(s)
        y = self.fc2(a)
        out = F.relu(x + y)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class DDPG_agent1(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A, LR_C,
                 gamma, TAU):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory1 = 0
        self.index_memory2 = 0
        self.loss_c = []
        self.loss_a = []
        self.memory1 = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))
        self.memory2 = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))
        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters), \
                                                     Actor_Net(self.state_counters, self.action_counters)

        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(parm.data)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(parm.data)
        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()
        return action.numpy()

    def store_memory(self, s, a, r, s_, bed, sit):

        if bed == 1:
            memory = np.hstack((s, a, [r], s_))
            index1 = self.index_memory1 % self.memory_size
            self.memory1[index1, :] = memory
            self.index_memory1 += 1
        if sit == 1:
            memory = np.hstack((s, a, [r], s_))
            index2 = self.index_memory2 % self.memory_size
            self.memory2[index2, :] = memory
            self.index_memory2 += 1

    def learn(self, bed, sit):
        if bed == 1:
            sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
            sample_memory = self.memory1[sample_memory_index, :]
            sample_memory = torch.FloatTensor(sample_memory)
            sample_memory_s = sample_memory[:, : self.state_counters]
            sample_memory_s_ = sample_memory[:, - self.state_counters:]
            sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
            sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

            a = self.Actor_Net_target(sample_memory_s_)
            a_s = self.Actor_Net_eval(sample_memory_s)
            q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
            q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
            loss_c = self.loss(q_target, q_eval)
            self.loss_c.append(loss_c.item())
            self.optimizer_C.zero_grad()
            loss_c.backward()
            self.optimizer_C.step()

            loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
            self.loss_a.append(loss_a.item())
            self.optimizer_A.zero_grad()
            loss_a.backward()
            self.optimizer_A.step()
        if sit == 1:
            sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
            sample_memory = self.memory2[sample_memory_index, :]
            sample_memory = torch.FloatTensor(sample_memory)
            sample_memory_s = sample_memory[:, : self.state_counters]
            sample_memory_s_ = sample_memory[:, - self.state_counters:]
            sample_memory_a = sample_memory[:, self.state_counters: self.state_counters + self.action_counters]
            sample_memory_r = sample_memory[:, - self.state_counters - 1: - self.state_counters]

            a = self.Actor_Net_target(sample_memory_s_)
            a_s = self.Actor_Net_eval(sample_memory_s)
            q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
            q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
            loss_c = self.loss(q_target, q_eval)
            self.loss_c.append(loss_c.item())
            self.optimizer_C.zero_grad()
            loss_c.backward()
            self.optimizer_C.step()

            loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
            self.loss_a.append(loss_a.item())
            self.optimizer_A.zero_grad()
            loss_a.backward()
            self.optimizer_A.step()

        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

class DDPG_agent3(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters), \
                                                     Actor_Net(self.state_counters, self.action_counters)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()

        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, r, s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]

        a = self.Actor_Net_target(sample_memory_s_)
        a_s = self.Actor_Net_eval(sample_memory_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        loss_c = self.loss(q_target, q_eval)
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()

        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()


        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)

class DDPG_agent5(object):

    def __init__(self, state_counters, action_counters, batch_size, memory_size, LR_A=0.001, LR_C=0.001,
                 gamma=0.99, TAU=0.005):

        self.state_counters = state_counters
        self.action_counters = action_counters
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.TAU = TAU
        self.index_memory = 0
        self.loss_c = []
        self.loss_a = []
        self.memory = np.zeros((memory_size, self.state_counters * 2 + self.action_counters + 1))

        self.Actor_Net_eval, self.Actor_Net_target = Actor_Net(self.state_counters, self.action_counters), \
                                                     Actor_Net(self.state_counters, self.action_counters)

        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(parm.data)

        self.Critic_Net_eval, self.Critic_Net_target = Critic_Net(self.state_counters, self.action_counters), \
                                                       Critic_Net(self.state_counters, self.action_counters)

        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(parm.data)
        self.optimizer_A = torch.optim.Adam(self.Actor_Net_eval.parameters(), lr=LR_A)

        self.optimizer_C = torch.optim.Adam(self.Critic_Net_eval.parameters(), lr=LR_C)

        self.loss = nn.MSELoss()

    def choose_action(self, observation):

        observation = torch.FloatTensor(observation)
        observation = torch.unsqueeze(observation, 0)
        action = self.Actor_Net_eval(observation)[0].detach()
        return action.numpy()

    def store_memory(self, s, a, r, s_):

        memory = np.hstack((s, a, [r], s_))
        index = self.index_memory % self.memory_size
        self.memory[index, :] = memory
        self.index_memory += 1

    def learn(self):

        sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
        sample_memory = self.memory[sample_memory_index, :]
        sample_memory = torch.FloatTensor(sample_memory)
        sample_memory_s = sample_memory[:, : self.state_counters]
        sample_memory_s_ = sample_memory[:, - self.state_counters:]
        sample_memory_a = sample_memory[:, self.state_counters : self.state_counters + self.action_counters]
        sample_memory_r = sample_memory[:, - self.state_counters -1 : - self.state_counters]
        # print(sample_memory)
        a = self.Actor_Net_target(sample_memory_s_)
        # print(a)
        a_s = self.Actor_Net_eval(sample_memory_s)
        # print(a_s)
        q_target = sample_memory_r + self.gamma * self.Critic_Net_target(sample_memory_s_, a)
        # print(q_target)
        q_eval = self.Critic_Net_eval(sample_memory_s, sample_memory_a)
        # print(q_eval)
        loss_c = self.loss(q_target, q_eval)
        self.loss_c.append(loss_c.item())
        self.optimizer_C.zero_grad()
        loss_c.backward()
        self.optimizer_C.step()
        loss_a = - torch.mean(self.Critic_Net_eval(sample_memory_s, a_s))
        # print(self.Critic_Net_eval(sample_memory_s, a_s))
        self.loss_a.append(loss_a.item())
        self.optimizer_A.zero_grad()
        loss_a.backward()
        self.optimizer_A.step()

        for parm, target_parm in zip(self.Actor_Net_eval.parameters(), self.Actor_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
        for parm, target_parm in zip(self.Critic_Net_eval.parameters(), self.Critic_Net_target.parameters()):
            target_parm.data.copy_(self.TAU * parm.data + (1 - self.TAU) * target_parm.data)
