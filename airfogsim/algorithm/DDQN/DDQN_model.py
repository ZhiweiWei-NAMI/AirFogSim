import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections  # 队列
import random


# ----------------------------------- #
# （1）经验回放池
# ----------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):
        # 创建一个队列，先进先出，队列长度不变
        self.buffer = collections.deque(maxlen=capacity)

    # 填充经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch组样本数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 分别取出这些数据，*获取list中的所有值
        state, action, reward, next_state, done = zip(*transitions)
        # 将state变成数组，后面方便计算
        return np.array(state), action, reward, np.array(next_state), done

    # 队列的长度
    def size(self):
        return len(self.buffer)


# ----------------------------------- #
# （2）构造网络，训练网络和目标网络共用该结构
# ----------------------------------- #

class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Net, self).__init__()
        # 只有一个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_actions]
        return x


# ----------------------------------- #
# （3）模型构建
# ----------------------------------- #

class Double_DQN:
    # （1）初始化
    def __init__(self, n_states, n_hiddens, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):
        # 属性分配
        self.n_states = n_states
        self.n_hiddens = n_hiddens
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.device = device
        # 记录迭代次数
        self.count = 0

        # 实例化训练网络
        self.q_net = Net(self.n_states, self.n_hiddens, self.n_actions)
        # 实例化目标网络
        self.target_q_net = Net(self.n_states, self.n_hiddens, self.n_actions)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # （2）动作选择
    def take_action(self, state):
        # numpy[n_states]-->[1, n_states]-->Tensor
        state = torch.Tensor(state[np.newaxis, :])
        # print('--------------------------')
        # print(state.shape)
        # 如果小于贪婪系数就取最大值reward最大的动作
        if np.random.random() < self.epsilon:
            # 获取当前状态下采取各动作的reward
            actions_value = self.q_net(state)
            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()
        # 如果大于贪婪系数就随即探索
        else:
            action = np.random.randint(self.n_actions)
        return action

    # （3）获取每个状态对应的最大的state_value
    def max_q_value(self, state):
        # list-->tensor[3]-->[1,3]
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        # 当前状态对应的每个动作的reward的最大值 [1,3]-->[1,11]-->int
        max_q = self.q_net(state).max().item()
        return max_q

    # （4）网络训练
    def update(self, transitions_dict):
        # 当前状态，array_shape=[b,4]
        states = torch.tensor(transitions_dict['states'], dtype=torch.float)
        # 当前状态的动作，tuple_shape=[b]==>[b,1]
        actions = torch.tensor(transitions_dict['actions'], dtype=torch.int64).view(-1, 1)
        # 选择当前动作的奖励, tuple_shape=[b]==>[b,1]
        rewards = torch.tensor(transitions_dict['rewards'], dtype=torch.float).view(-1, 1)
        # 下一个时刻的状态array_shape=[b,4]
        next_states = torch.tensor(transitions_dict['next_states'], dtype=torch.float)
        # 是否到达目标 tuple_shape=[b,1]
        dones = torch.tensor(transitions_dict['dones'], dtype=torch.float).view(-1, 1)

        # 当前状态[b,4]-->当前状态采取的动作及其奖励[b,2]-->actions中是每个状态下的动作索引
        # -->当前状态s下采取动作a得到的state_value
        q_values = self.q_net(states).gather(1, actions)
        # 获取动作索引
        # .max(1)输出tuple每个特征的最大state_value及其索引，[1]获取的每个特征的动作索引shape=[b]
        max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
        # 下个状态的state_value。下一时刻的状态输入到目标网络，得到每个动作对应的奖励，使用训练出来的action索引选取最优动作
        max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        # 目标网络计算出的，当前状态的state_value
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 预测值和目标值的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 梯度清零
        self.optimizer.zero_grad()
        # 梯度反传
        dqn_loss.backward()
        # 更新训练网络的参数
        self.optimizer.step()

        # 更新目标网络参数
        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        # 迭代计数+1
        self.count += 1