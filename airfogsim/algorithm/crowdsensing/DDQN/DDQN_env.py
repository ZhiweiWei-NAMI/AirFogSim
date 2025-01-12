import os

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .DDQN_model import Double_DQN

# self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# # 输入层神经元数量（状态空间大小）
# self.dim_states = dim_states
# # 输出层神经元数量（动作空间大小）
# self.dim_actions = dim_actions
#
# # 超参数
# self.buffer_size = 500  # 经验池容量
# self.lr = 2e-3  # 学习率
# self.gamma = 0.9  # 折扣因子
# self.epsilon = 0.9  # 探索系数
# self.eps_end = 0.01  # 最低探索系数
# self.eps_dec = 5e-7  # 探索系数衰减率
# self.target_update = 200  # 目标网络的参数的更新频率
# self.batch_size = 32  # 每次训练选取的经验数量
# self.dim_hidden = 128  # 隐含层神经元个数
# self.train_min_size = 200  # 经验池超过200后再训练(train_min_size>batch_size)
# self.tau = 0.995  # 目标网络软更新平滑因子（策略网络权重）
# self.smooth_factor = 0.995  # 最大q值平滑因子（旧值权重）
class DDQN_Env:

    def __init__(self, dim_args, train_args):
        # self.return_list = []  # 记录每次迭代的return，即链上的reward之和
        self.smooth_factor=train_args.smooth_factor
        self.max_q_value = 0  # 最大state_value
        self.max_q_value_list = []  # 保存所有最大的state_value

        # 模型文件路径
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.model_base_dir = os.path.join(current_dir, "model")
        self.model_base_dir=train_args.model_base_dir

        # 实例化 Double-DQN
        self.agent = Double_DQN(dim_args, train_args)

    def takeAction(self, state, mask):
        # 状态state时做动作选择，action为动作索引
        is_random, max_q_value, action = self.agent.take_action(state,mask)
        # 平滑处理最大state_value
        self.max_q_value = max_q_value * (1 - self.smooth_factor) + self.max_q_value * self.smooth_factor
        # 保存每次迭代的最大state_value
        # self.max_q_value_list.append(self.max_q_value)
        return is_random,self.max_q_value,action

    def addExperience(self, state, action,mask, reward, next_state,next_mask, done):
        # 添加经验池
        self.agent.remember(state, action,mask, reward, next_state,next_mask, done)

    def train(self):
        self.agent.update()

    def saveModel(self,episode):
        self.agent.save_models(episode,self.model_base_dir)

    def loadModel(self,episode):
        self.agent.load_models(episode,self.model_base_dir)


