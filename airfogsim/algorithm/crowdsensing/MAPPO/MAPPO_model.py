import argparse
import os

from .MAPPO_network import Critic, Actor
import torch
from copy import deepcopy
from .MAPPO_memory import Memory
from torch.optim import Adam
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
import logging

def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),source.parameters()):
        target_param.data.copy_(source_param.data)

def compute_advantage(gamma, gae_lambda, td_error,device):
    # 用通用优势估计（Generalized Advantage Estimator, GAE）构建时序优势
    td_error = td_error.detach().cpu().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_error[::-1]:
        advantage = gamma * gae_lambda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float,device=device)

class MAPPO:
    def __init__(self, dim_args, train_args):
        # 维度超参数
        self.n_agents = dim_args.n_agents
        self.dim_hiddens = dim_args.dim_hiddens
        self.dim_obs = dim_args.dim_observation
        self.dim_act = dim_args.dim_action


        # 训练超参数
        self.lr = train_args.learning_rate
        self.gamma = train_args.gamma
        self.gae_lambda = train_args.gae_lambda
        self.epsilon = train_args.epsilon
        self.epoch = train_args.epoch
        self.device = train_args.device

        # 实例化策略网络*n
        self.actors = [Actor(self.dim_obs, self.dim_hiddens) for i in range(self.n_agents)]
        self.old_actors = deepcopy(self.actors) # 旧行为策略
        # 实例化价值网络*n
        self.critics = [Critic(self.n_agents, self.dim_obs, self.dim_act, self.dim_hiddens) for i in range(self.n_agents)]
        self.old_critics = deepcopy(self.critics) # 旧评价网络
        # 策略训练网络优化器
        self.critic_optimizer = [Adam(x.parameters(), lr=self.lr) for x in self.critics]
        # 目标训练网络优化器
        self.actor_optimizer = [Adam(x.parameters(), lr=self.lr) for x in self.actors]

        # 经验池
        self.memory = [Memory() for i in range(self.n_agents)] # 每个agent构建独立经验池

        for x in self.actors:
            x.to(self.device)
        for x in self.old_actors:
            x.to(self.device)
        for x in self.critics:
            x.to(self.device)
        for x in self.old_critics:
            x.to(self.device)


        # 记录迭代次数
        self.steps_done = 0

    def remember(self,agent_id, state, action, reward, next_state):
        self.memory[agent_id].add(state, action, reward, next_state)

    def take_action(self,agents_state):
        # agents_state: [n_agents, state_dim]
        agents_state=torch.tensor(agents_state,dtype=torch.float).to(self.device)
        actions=[]
        with torch.no_grad():
            for i in range(self.n_agents):
                state = agents_state[i, :].detach()
                mu,sigma = self.old_actors[i](state.unsqueeze(0))
                mu=mu.squeeze(0) # 压缩batch维度
                sigma=sigma.squeeze(0) # 压缩batch维度
                dis = torch.distributions.normal.Normal(mu, sigma)
                action = dis.sample()
                action= torch.clamp(action,0,1) # action裁剪到normalization范围
                actions.append(action)
        self.steps_done += 1

        return actions

    def update(self):
        c_loss = []
        a_loss = []
        for _ in range(self.epoch):
            for agent_idx in range(self.n_agents):
                # 同一时间的全局state,action,next_state,reward
                states, actions, rewards, next_states= self.memory[agent_idx].get_all()
                # 转换为 PyTorch 张量
                # numpy[batch_size, n_agents, state_dim]-->Tensor[batch_size, n_agents, state_dim]
                states = torch.tensor(states, dtype=torch.float).to(self.device)
                # numpy[batch_size, action_dim]-->Tensor[batch_size, n_agents, action_dim]
                actions = torch.tensor(actions, dtype=torch.float).to(self.device)
                # numpy[batch_size]-->Tensor[batch_size, n_agents]
                rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
                # numpy[batch_size, n_agents, state_dim]-->Tensor[batch_size, n_agents, state_dim]
                next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)

                whole_states = states.view(states.shape[0], -1)
                whole_next_states = states.view(next_states.shape[0], -1)
                with torch.no_grad():
                    td_targets = rewards+self.old_critics[agent_idx](whole_next_states)
                    mu,sigma = self.old_actors[agent_idx](states[:,agent_idx,:])
                    old_dis = torch.distributions.normal.Normal(mu, sigma)
                    log_prob_old = old_dis.log_prob(actions)
                    td_errors = rewards + self.gamma * self.critics[agent_idx](whole_next_states)  - self.critics[agent_idx](whole_states)
                    adv=compute_advantage(self.gamma, self.gae_lambda, td_errors,self.device)

                # 1.更新actor
                mu, sigma = self.actors[agent_idx](states[:,agent_idx,:])
                new_dis = torch.distributions.normal.Normal(mu, sigma)
                log_prob_new = new_dis.log_prob(actions)
                ratio = torch.exp(log_prob_new - log_prob_old)
                L1 = ratio * adv
                L2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * adv
                loss_actor = -torch.min(L1, L2).mean()
                self.actor_optimizer[agent_idx].zero_grad()
                loss_actor.backward()
                self.actor_optimizer[agent_idx].step()

                # 2.更新critic，next_state估值使用旧价值网络
                q_values = self.critics[agent_idx](whole_states)
                loss_critic = F.mse_loss(q_values,td_targets.detach())
                self.critic_optimizer[agent_idx].zero_grad()
                loss_critic.backward()
                self.critic_optimizer[agent_idx].step()

                a_loss.append(loss_actor.detach().item())
                c_loss.append(loss_critic.detach().item())

        for i in range(self.n_agents):
            hard_update(self.critics[i], self.old_critics[i])
            hard_update(self.actors[i], self.old_actors[i])
            self.memory[i].clear()

        return a_loss,c_loss


    def save_models(self, episode, base_dir,final):
        if final is True:
            file_dir = f"{base_dir}/final"
        else:
            file_dir=f"{base_dir}/episode_{episode}"
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        logging.basicConfig(
            level=logging.INFO,  # 日志级别
            format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
            filename=f'{file_dir}/model.log',  # 日志文件名
            filemode='a'  # 追加模式写入日志文件
        )

        for i in range(self.n_agents):
            self.critics[i].save_model(file_dir + f'/MAPPO_critics_{i}.pth')
            print(f'Saving MAPPO_critics_{i} network successfully!')
            logging.info(f'Saving MAPPO_critics_{i} network successfully!')
            self.actors[i].save_model(file_dir + f'/MAPPO_actors_{i}.pth')
            print(f'Saving MAPPO_actors_{i} network successfully!')
            logging.info(f'Saving MAPPO_actors_{i} network successfully!')
            self.old_critics[i].save_model(file_dir + f'/MAPPO_old_critics_{i}.pth')
            print(f'Saving MAPPO_old_critics_{i} network successfully!')
            logging.info(f'Saving MAPPO_old_critics_{i} network successfully!')
            self.old_actors[i].save_model(file_dir + f'/MAPPO_old_actors_{i}.pth')
            print(f'Saving MAPPO_old_actors_{i} network successfully!')
            logging.info(f'Saving MAPPO_old_actors_{i} network successfully!')


    def load_models(self, episode, base_dir,final):
        if final is True:
            file_dir = f"{base_dir}/final"
        else:
            file_dir=f"{base_dir}/episode_{episode}"

        logging.basicConfig(
            level=logging.INFO,  # 日志级别
            format='%(asctime)s [%(levelname)s] %(message)s',  # 日志格式
            datefmt='%Y-%m-%d %H:%M:%S',  # 时间格式
            filename=f'{file_dir}/model.log',  # 日志文件名
            filemode='a'  # 追加模式写入日志文件
        )

        for i in range(self.n_agents):
            self.critics[i].load_model(file_dir + f'/MAPPO_critics_{i}.pth')
            print(f'Loading MAPPO_critics_{i} network successfully!')
            logging.info(f'Loading MAPPO_critics_{i} network successfully!')
            self.actors[i].load_model(file_dir + f'/MAPPO_actors_{i}.pth')
            print(f'Loading MAPPO_actors_{i} network successfully!')
            logging.info(f'Loading MAPPO_actors_{i} network successfully!')
            self.old_critics[i].load_model(file_dir + f'/MAPPO_old_critics_{i}.pth')
            print(f'Loading MAPPO_old_critics_{i} network successfully!')
            logging.info(f'Loading MAPPO_old_critics_{i} network successfully!')
            self.old_actors[i].load_model(file_dir + f'/MAPPO_old_actors_{i}.pth')
            print(f'Loading MAPPO_old_actors_{i} network successfully!')
            logging.info(f'Loading MAPPO_old_actors_{i} network successfully!')

