import os

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import collections  # 队列
import random
import logging
from .TransD3QN_relay_buffer import ReplayBuffer
from .TransD3QN_network import Net

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(source_param.data)

# ----------------------------------- #
# 模型构建
# ----------------------------------- #
class TransD3QN:
    # （1）初始化
    def __init__(self, dim_args, train_args):
        # 训练超参数
        self.lr = train_args.learning_rate
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
        self.optimizer = torch.optim.Adam(params=self.q_net.parameters(), lr=self.lr)
        # 损失函数
        self.criterion = torch.nn.MSELoss()

        # 经验池
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, train_min_size=self.train_min_size)

    def remember(self, node_state, mission_state, sensor_state, sensor_mask, action,  reward, next_node_state, next_mission_state, next_sensor_state, next_sensor_mask, done):
        self.memory.add(node_state, mission_state, sensor_state, sensor_mask, action, reward, next_node_state, next_mission_state, next_sensor_state, next_sensor_mask, done)

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
        node_state = torch.tensor(node_state[np.newaxis, :], dtype=torch.float).to(self.device)
        # numpy[dim_mission]-->[1, dim_mission]-->Tensor
        mission_state = torch.tensor(mission_state[np.newaxis, :], dtype=torch.float).to(self.device)
        # numpy[m_uv, max_sensors, dim_sensor]-->[1, m_uv, max_sensors, dim_sensor]-->Tensor
        sensor_state = torch.tensor(sensor_state[np.newaxis, :], dtype=torch.float).to(self.device)
        # numpy[m_uv, max_sensors]-->[1, m_uv, max_sensors]-->Tensor
        sensor_mask = torch.tensor(sensor_mask[np.newaxis, :], dtype=torch.bool).to(self.device)

        with torch.no_grad():
            # 获取当前状态下采取各动作的q值
            q_values = self.q_net(node_state, mission_state, sensor_state, sensor_mask)

        # 非法动作置为最小值
        flatten_mask=sensor_mask.flatten()
        flatten_q_values=q_values.flatten()
        # masked_q_values = torch.where(flatten_mask, flatten_q_values, torch.tensor(float('-inf')))
        flatten_q_values.copy_(torch.where(flatten_mask, flatten_q_values, torch.tensor(float('-inf'))))
        # 对每个样本找到最大 Q 值对应的动作的索引
        max_q_value, max_action_index = torch.max(flatten_q_values, dim=-1)
        # 如果小于贪婪系数就取最大值reward最大的动作
        if np.random.random() < self.epsilon:
            is_random = False
            # 获取reward最大值对应的动作索引
            # action = masked_q_values.argmax().item()
            if max_q_value.item() == float('-inf'):
                action = None
            else:
                action = max_action_index.item()
        # 如果大于贪婪系数就随机探索
        else:
            is_random = True
            valid_action_indices = torch.nonzero(flatten_mask, as_tuple=False).squeeze(1)
            if valid_action_indices.numel() == 0:
                action = None
            else:
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
        node_states, mission_states, sensor_states, sensor_masks, actions, rewards, next_node_states, next_mission_states, next_sensor_states, next_sensor_masks, dones = self.memory.sample(self.batch_size)

        # 当前状态
        # numpy[batch_size, m1, dim_node]-->Tensor[batch_size, m1, dim_node]
        node_states = torch.tensor(node_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, dim_mission]-->Tensor[batch_size, dim_mission]
        mission_states = torch.tensor(mission_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, m_uv, max_sensors, dim_sensor]-->Tensor[batch_size, m_uv, max_sensors, dim_sensor]
        sensor_states = torch.tensor(sensor_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, m_uv, max_sensors]-->Tensor[batch_size, m_uv, max_sensors]
        sensor_masks = torch.tensor(sensor_masks, dtype=torch.bool).to(self.device)

        # 当前动作索引
        # numpy[batch_size]-->Tensor[batch_size]
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)

        # 当前动作的奖励
        # numpy[batch_size]-->Tensor[batch_size]
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)

        # 下一状态
        # numpy[batch_size, m1, dim_node]-->Tensor[batch_size, m1, dim_node]
        next_node_states = torch.tensor(next_node_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, dim_mission]-->Tensor[batch_size, dim_mission]
        next_mission_states = torch.tensor(next_mission_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, m_uv, max_sensors, dim_sensor]-->Tensor[batch_size, m_uv, max_sensors, dim_sensor]
        next_sensor_states = torch.tensor(next_sensor_states, dtype=torch.float).to(self.device)
        # numpy[batch_size, m_uv, max_sensors]-->Tensor[batch_size, m_uv, max_sensors]
        next_sensor_masks = torch.tensor(next_sensor_masks, dtype=torch.bool).to(self.device)

        # 是否到达目标
        # numpy[batch_size]-->Tensor[batch_size]
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        flatten_next_masks=next_sensor_masks.reshape(self.batch_size,-1) # shape(b, m_uv * max_sensors)
        with torch.no_grad():
            next_q_values = self.q_net.forward(next_node_states,next_mission_states,next_sensor_states,next_sensor_masks)
            # next_masked_q_values = torch.where(flatten_next_masks, next_q_values, torch.tensor(float('-inf')))
            next_q_values.copy_(torch.where(flatten_next_masks, next_q_values, torch.tensor(float('-inf'))))
            # [batch_size] -> [batch_size,index_size(1)]
            max_next_actions = torch.argmax(next_q_values, dim=1).unsqueeze(-1)
            next_q_targets = self.target_q_net.forward(next_node_states,next_mission_states,next_sensor_states,next_sensor_masks)
            td_q_targets = rewards + self.gamma * next_q_targets.gather(1, max_next_actions) * (1 - dones.int())
        actions=actions.unsqueeze(-1)
        q_values = self.q_net(node_states,mission_states,sensor_states,sensor_masks).gather(1, actions)

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
            soft_update(self.target_q_net,self.q_net,self.tau)
        self.decrement_epsilon()
        # 迭代计数+1
        self.steps_done += 1

        return dqn_loss.detach().item()

    def save_models(self, episode,base_dir,final):
        if final is True:
            file_dir = f"{base_dir}/final"
        else:
            file_dir=f"{base_dir}/episode_{episode}"
        model_type = "final" if final is True else "checkpoint"

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        logging.basicConfig(
            level=logging.INFO,  # 日志级别
            format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
            filename=f'{file_dir}/model.log',  # 日志文件名
            filemode='a'  # 追加模式写入日志文件
        )

        self.q_net.save_model(file_dir + f'/TransD3QN_Q_net.pth')
        print(f'Saving {model_type} episode_{episode} TransD3QN_Q_net network successfully!')
        logging.info(f'Saving {model_type} episode_{episode} TransD3QN_Q_net network successfully!')
        self.target_q_net.save_model(file_dir + f'/TransD3QN_Q_target.pth')
        print(f'Saving {model_type} episode_{episode} TransD3QN_Q_target network successfully!')
        logging.info(f'Saving {model_type} episode_{episode} TransD3QN_Q_target network successfully!')
        self.memory.save(file_dir+f'/TransD3QN_memory.pkl')
        print(f'Saving {model_type} episode_{episode} TransD3QN memory successfully!')
        logging.info(f'Saving {model_type} episode_{episode} TransD3QN memory successfully!')

    def load_models(self, episode,base_dir,final):
        if final is True:
            file_dir = f"{base_dir}/final"
        else:
            file_dir=f"{base_dir}/episode_{episode}"
        model_type = "final" if final is True else "checkpoint"

        logging.basicConfig(
            level=logging.INFO,  # 日志级别
            format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
            filename=f'{file_dir}/model.log',  # 日志文件名
            filemode='a'  # 追加模式写入日志文件
        )

        self.q_net.load_model(file_dir + f'/TransD3QN_Q_net.pth')
        print(f'Loading {model_type} episode_{episode} TransD3QN_Q_net network successfully!')
        logging.info(f'Loading {model_type} episode_{episode} TransD3QN_Q_net network successfully!')
        self.target_q_net.load_model(file_dir + f'/TransD3QN_Q_target.pth')
        print(f'Loading {model_type} episode_{episode} TransD3QN_Q_target network successfully!')
        logging.info(f'Loading {model_type} episode_{episode} TransD3QN_Q_target network successfully!')
        self.memory.load(file_dir+f'/TransD3QN_memory.pkl')
        print(f'Loading {model_type} episode_{episode} TransD3QN memory successfully!')
        logging.info(f'Loading {model_type} episode_{episode} TransD3QN memory successfully!')
