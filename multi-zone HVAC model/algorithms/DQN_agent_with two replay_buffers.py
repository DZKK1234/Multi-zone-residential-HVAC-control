import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class Q_Network(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc1.bias.data.normal_(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc2.bias.data.normal_(0.1)
        self.fc3 = nn.Linear(64, action_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc3.bias.data.normal_(0.1)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class DQN(object):

    def __init__(self, state_dim, action_dim, gamma, memory_size, learning_rate, epsilon, batch_size, delay_update):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.memory_size = memory_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay = 0.00005
        self.batch_size = batch_size
        self.delay_update = delay_update
        self.loss = []
        self.counter = 0
        self.Q_eval_network, self.Q_target_network = Q_Network(self.state_dim, self.action_dim), \
                                                     Q_Network(self.state_dim, self.action_dim)
        self.replay_memory1 = np.zeros((self.memory_size, self.state_dim * 2 + 1 + 2))
        self.replay_memory2 = np.zeros((self.memory_size, self.state_dim * 2 + 1 + 2))
        self.index_memory1 = 0
        self.index_memory2 = 0
        self.optimizer = torch.optim.Adam(self.Q_eval_network.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def choose_action(self, observation):

        s = torch.unsqueeze(torch.FloatTensor(observation), 0)
        a = self.Q_eval_network(s)
        if self.epsilon > 0.0001:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = 0
        if np.random.uniform() > self.epsilon:
            action = torch.max(a, 1)[1].detach().numpy()[0]
        else:
            action = np.random.randint(0, self.action_dim)

        return action

    def store_memory(self, s, a, r, s_, d, bed, sit):
        if bed == 1:
            memory = np.hstack((s, a, [r], s_, [d]))
            index1 = self.index_memory1 % self.memory_size
            self.replay_memory1[index1, :] = memory
            self.index_memory1 += 1
        if sit == 1:
            memory = np.hstack((s, a, [r], s_, [d]))
            index2 = self.index_memory2 % self.memory_size
            self.replay_memory2[index2, :] = memory
            self.index_memory2 += 1
    def sample_memory(self, bed, sit):
        if bed == 1:
            sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
            sample_memory = self.replay_memory1[sample_memory_index, :]
            sample_memory_s = torch.FloatTensor(sample_memory[:, : self.state_dim])
            sample_memory_a = torch.LongTensor(sample_memory[:, self.state_dim: 1 + self.state_dim])
            sample_memory_r = torch.FloatTensor(sample_memory[:, - self.state_dim - 2: - self.state_dim - 1])
            sample_memory_s_ = torch.FloatTensor(sample_memory[:, - self.state_dim - 1: -1])
            sample_memory_d = torch.FloatTensor(sample_memory[:, -1:])
            return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_d
        if sit == 1:
            sample_memory_index = np.random.choice(self.memory_size, self.batch_size)
            sample_memory = self.replay_memory2[sample_memory_index, :]
            sample_memory_s = torch.FloatTensor(sample_memory[:, : self.state_dim])
            sample_memory_a = torch.LongTensor(sample_memory[:, self.state_dim: 1 + self.state_dim])
            sample_memory_r = torch.FloatTensor(sample_memory[:, - self.state_dim - 2: - self.state_dim - 1])
            sample_memory_s_ = torch.FloatTensor(sample_memory[:, - self.state_dim - 1: -1])
            sample_memory_d = torch.FloatTensor(sample_memory[:, -1:])
            return sample_memory_s, sample_memory_a, sample_memory_r, sample_memory_s_, sample_memory_d
    def learn(self, bed, sit):
        self.counter += 1
        # 每过delay_update步长跟新target_Q
        if self.counter % self.delay_update == 0:
            for parm, target_parm in zip(self.Q_eval_network.parameters(), self.Q_target_network.parameters()):
                target_parm.data.copy_(parm.data)
        s, a, r, s_, d = self.sample_memory(bed, sit)
        # 根据a来选取对应的Q(s,a)
        q_eval = self.Q_eval_network(s).gather(1, a)
        # 计算target_Q
        q_target = self.gamma * torch.max(self.Q_target_network(s_), 1)[0].reshape(self.batch_size, 1)
        y = r + (1 - d) * q_target

        # 网络跟新
        loss = self.loss_function(y, q_eval)
        self.loss.append(loss.item())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()



