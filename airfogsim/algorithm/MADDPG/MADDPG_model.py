from .MADDPG_network import Critic, Actor
import torch
from copy import deepcopy
from .MADDPG_relay_buffer import ReplayBuffer
from torch.optim import Adam
from torch.nn import functional as F
import torch.nn as nn
import numpy as np


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act,lr, gamma, buffer_size, batch_size,
                 episodes_before_train, train_min_size, tau, device):
        self.n_agents = n_agents
        self.dim_states = dim_obs
        self.dim_actions = dim_act
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        # self.use_cuda = th.cuda.is_available()
        self.device=device
        self.train_min_size = train_min_size
        self.episodes_before_train = episodes_before_train

        self.lr=lr
        self.gamma = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]  # 动作探索随机噪声

        # 实例化策略训练网络*n
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        # 实例化价值训练网络*n
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        # 实例化目标策略网络*n
        self.actors_target = deepcopy(self.actors)
        # 实例化目标价值网络*n
        self.critics_target = deepcopy(self.critics)
        # 策略训练网络优化器
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=self.lr) for x in self.critics]
        # 目标训练网络优化器
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=self.lr) for x in self.actors]

        # 经验池
        self.memory = ReplayBuffer(buffer_size=self.buffer_size, train_min_size=self.train_min_size)

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def update(self):
        if not self.memory.ready():
            return
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return

        c_loss = []
        a_loss = []
        for agent in range(self.n_agents):
            # 同一时间的全局state,action,next_state,reward
            states,actions,next_states,rewards = self.memory.sample(self.batch_size)
            # 转换为 PyTorch 张量
            states = torch.tensor(states, dtype=torch.float)
            actions = torch.tensor(actions, dtype=torch.float).view(-1,1)
            rewards = torch.tensor(rewards, dtype=torch.float).view(-1,1)
            next_states = torch.tensor(next_states, dtype=torch.float)


            whole_states = states.view(self.batch_size, -1)
            whole_actions = actions.view(self.batch_size, -1)

            # 1.Critic 更新(agent_i)
            q_values_i = self.critics[agent](whole_states, whole_actions) #实际采取的动作
            with torch.no_grad():
                next_whole_states = next_states.view(self.batch_size, -1)
                # all agents next actions
                next_whole_actions = [
                    self.actors_target[i](next_states[:, i, :]) for i in range(self.n_agents)
                ]
                next_whole_actions = torch.cat(next_whole_actions, dim=-1)
                q_targets_i = self.critics_target[agent](next_whole_states, next_whole_actions).squeeze()
                td_targets_i = rewards[:, agent] + self.gamma * q_targets_i

            critic_loss = F.mse_loss(q_values_i, td_targets_i.detach())
            self.critic_optimizer[agent].zero_grad()
            critic_loss.backward()
            self.critic_optimizer[agent].step()

            # 2.Actor 更新(agent_i)
            state_i=states[:, agent, :]
            online_action_i = self.actors[agent](state_i)  # 非经验里state_i对应的实际动作，是重新采样的
            online_action = actions.clone()
            online_action[:, agent, :] = online_action_i
            whole_online_actions = online_action.view(self.batch_size, -1)

            # loss采用-q，评价越好loss越小
            actor_loss = -self.critics[agent](whole_states, whole_online_actions).mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            self.actor_optimizer[agent].step()

            c_loss.append(critic_loss.item())
            a_loss.append(actor_loss.item())

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, agents_state):
        # agents_state: n_agents x state_dim
        for i in range(self.n_agents):
            state = agents_state[i, :].detach()
            actions = self.actors[i](state.unsqueeze(0)).squeeze()
            actions += torch.from_numpy(np.random.randn(2) * self.var[i])
            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998
            # act = th.clamp(act, -1.0, 1.0)
            # actions[i, :] = act
        self.steps_done += 1

        return actions
