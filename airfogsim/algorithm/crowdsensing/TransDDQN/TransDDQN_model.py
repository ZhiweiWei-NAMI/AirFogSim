import os

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections  # 队列
import random
from .TransDDQN_relay_buffer import ReplayBuffer
from .TransDDQN_network import Net

# ----------------------------------- #
# 模型构建
# ----------------------------------- #
class TransDDQN:
    # （1）初始化
    def __init__(self, dim_args, train_args):
        # 训练超参数
        self.learning_rate = train_args.learning_rate
        self.gamma = train_args.gamma
        self.epsilon = train_args.epsilon
        self.eps_min = train_args.eps_end
        self.eps_dec = train_args.eps_dec
        self.target_update = train_args.target_update
        self.buffer_size = train_args.buffer_size
        self.batch_size=train_args.batch_size
        self.train_min_size = train_args.train_min_size
        self.tau = train_args.tau
        self.device = torch.device(train_args.device)
        # 记录迭代次数
        self.steps_done = 0

        # 实例化训练网络
        self.q_net = Net(dim_args)
        self.q_net.to(self.device)
        # 实例化目标网络
        self.target_q_net = Net(dim_args)
        self.target_q_net.to(self.device)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(params=self.q_net.parameters(), lr=self.learning_rate)
        # 损失函数
        self.criterion = torch.nn.MSELoss()

        # 经验池
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, train_min_size=self.train_min_size)

    # 目标网络更新
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        for q_target_params, q_params in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            q_target_params.data.copy_(tau * q_params + (1 - tau) * q_target_params)

    def remember(self, node_state, mission_state, sensor_state, sensor_mask, action, mask, reward, next_state, next_mask, done):
        self.memory.add(node_state, mission_state, sensor_state, sensor_mask, action, mask, reward, next_state, next_mask, done)

    # 动作选择
    def take_action(self,node_state, mission_state, sensor_state, sensor_mask):
        """
        Args:
            node_state: [m1, dim_node]
            mission_state: [dim_mission]
            sensor_state: [m_uv, max_sensors, dim_sensor]
            sensor_mask: [m_uv, max_sensors], 1 for valid sensors, 0 for others
        """
        # numpy[m1, dim_node]-->[1, m1, dim_node]-->Tensor
        node_state = torch.Tensor(node_state[np.newaxis, :])
        # numpy[dim_mission]-->[1, dim_mission]-->Tensor
        mission_state = torch.Tensor(mission_state[np.newaxis, :])
        # numpy[m_uv, max_sensors, dim_sensor]-->[1, m_uv, max_sensors, dim_sensor]-->Tensor
        sensor_state = torch.Tensor(sensor_state[np.newaxis, :])
        # numpy[m_uv, max_sensors]-->[1, m_uv, max_sensors]-->Tensor
        sensor_mask = torch.Tensor(sensor_mask[np.newaxis, :])
        # 获取当前状态下采取各动作的q值
        q_values = self.q_net(node_state, mission_state, sensor_state, sensor_mask)
        # 非法动作置为最小值
        flatten_mask=sensor_mask.flatten()
        flatten_q_values=q_values.flatten()
        masked_q_values = torch.where(flatten_mask, flatten_q_values, torch.tensor(float('-inf')))
        # 对每个样本找到最大 Q 值对应的动作的索引
        max_q_value, max_action_index = torch.max(masked_q_values, dim=-1)
        # 如果小于贪婪系数就取最大值reward最大的动作
        if np.random.random() < self.epsilon:
            is_random = False
            # 获取reward最大值对应的动作索引
            action = masked_q_values.argmax().item()
        # 如果大于贪婪系数就随即探索
        else:
            is_random = True
            valid_action_indices = torch.nonzero(flatten_mask, as_tuple=False).squeeze(1)
            action = valid_action_indices[torch.randint(0, len(valid_action_indices), (1,))].item()

        return is_random, max_q_value, action

    def decrement_epsilon(self):
        if self.epsilon > self.eps_min:
            self.epsilon = self.epsilon - self.eps_dec
        else:
            self.epsilon = self.eps_min

    # 获取每个状态对应的最大的state_value
    # def max_q_value(self, state):
    #     # list-->tensor[3]-->[1,3]
    #     state = torch.tensor(state, dtype=torch.float).view(1, -1)
    #     # 当前状态对应的每个动作的reward的最大值 [1,3]-->[1,11]-->int
    #     max_q = self.q_net.forward(state).max().item()
    #     return max_q

    # 网络训练
    def update(self):
        if not self.memory.ready():
            return
        states, actions, masks, rewards, next_states, next_masks, dones = self.memory.sample(self.batch_size)

        # 当前状态，array_shape=[b,4]
        states = torch.tensor(states, dtype=torch.float)
        # 当前状态的动作，tuple_shape=[b]==>[b,1]
        actions = torch.tensor(actions, dtype=torch.int64).view(-1, 1)
        # 动作掩码（布尔张量，True为有效动作，False为无效动作）
        masks = torch.tensor(masks, dtype=torch.bool).view(-1, 1)
        # 选择当前动作的奖励, tuple_shape=[b]==>[b,1]
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1)
        # 下一个时刻的状态array_shape=[b,4]
        next_states = torch.tensor(next_states, dtype=torch.float)
        # 动作掩码（布尔张量，True为有效动作，False为无效动作）
        next_masks = torch.tensor(next_masks, dtype=torch.bool).view(-1, 1)
        # 是否到达目标 tuple_shape=[b,1]
        dones = torch.tensor(dones, dtype=torch.bool).view(-1, 1)

        with torch.no_grad():
            next_q_values = self.q_net.forward(next_states)
            next_masked_q_values = torch.where(next_masks, next_q_values, torch.tensor(float('-inf')))
            max_next_actions = torch.argmax(next_masked_q_values, dim=-1)
            next_q_targets = self.target_q_net.forward(next_states)
            td_q_targets = rewards + self.gamma * next_q_targets.gather(1, max_next_actions) * (1 - dones)
        q_values = self.q_net(states).gather(1, actions)

        # 预测值和目标值的均方误差损失(取一个batch的平均值)
        dqn_loss = torch.mean(F.mse_loss(q_values, td_q_targets.detach()))
        # 梯度清零
        self.optimizer.zero_grad()
        # 梯度反传
        dqn_loss.backward()
        # 更新训练网络的参数
        self.optimizer.step()

        # 更新目标网络参数
        if self.steps_done % self.target_update == 0 and self.steps_done>0:
            self.update_network_parameters(self.tau)
        self.decrement_epsilon()
        # 迭代计数+1
        self.steps_done += 1

    def save_models(self, episode,base_dir):
        file_dir=f"{base_dir}/episode_{episode}/"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.q_net.save_model(file_dir + f'TransDDQN_Q_net.pth')
        print('Saving TransDDQN_Q_net network successfully!')
        self.target_q_net.save_model(file_dir + f'TransDDQN_Q_target.pth')
        print('Saving TransDDQN_Q_target network successfully!')

    def load_models(self, episode,base_dir):
        file_dir = f"{base_dir}/episode_{episode}/"
        self.q_net.load_model(file_dir + f'TransDDQN_Q_net.pth')
        print('Loading TransDDQN_Q_net network successfully!')
        self.target_q_net.load_model(file_dir + f'TransDDQN_Q_target.pth')
        print('Loading TransDDQN_Q_target network successfully!')
