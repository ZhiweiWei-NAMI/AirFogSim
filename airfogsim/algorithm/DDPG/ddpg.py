from .ac_network import ActorNetwork, CriticNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..replay_buffer import ReplayBuffer
import os

# 线程数量=4
torch.set_num_threads(4)
class DDPG_Agent:
    def __init__(self, args):
        self.args = args
        self.d_node = args.d_node
        self.d_task = args.d_task
        self.max_tasks = args.max_tasks
        self.m1 = args.m1
        self.m2 = args.m2
        self.d_model = args.d_model
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.device = args.device
        self.tau = args.tau
        # state包括task_node, compute_node, reward
        self.state_dim = self.d_node * self.m1 + self.d_node * self.m2 + 1 
        self.action_dim = self.m1 * self.max_tasks * (self.m2+1) # 每个task选择一个compute node，或者是本地计算
        # d_node, d_task, max_tasks, m1, m2, d_model
        self.actor_network = ActorNetwork(self.d_node, self.d_task, self.max_tasks, self.m1, self.m2, self.d_model).to(self.device)
        self.critic_network = CriticNetwork(self.d_node, self.d_task, self.max_tasks, self.m1, self.m2, self.d_model).to(self.device)
        if args.mode == 'test':
            self.actor_network.load_state_dict(torch.load(args.actor_model_path))
            self.critic_network.load_state_dict(torch.load(args.critic_model_path))
                
        self.target_actor_network = ActorNetwork(self.d_node, self.d_task, self.max_tasks, self.m1, self.m2, self.d_model).to(self.device)
        self.target_critic_network = CriticNetwork(self.d_node, self.d_task, self.max_tasks, self.m1, self.m2, self.d_model).to(self.device)
        self.target_actor_network.load_state_dict(self.actor_network.state_dict())
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        self.gamma = args.gamma
        # self.optimizer_actor = torch.optim.SGD(self.actor_network.parameters(), lr=args.lr_actor, momentum=0.9)
        # self.optimizer_critic = torch.optim.SGD(self.critic_network.parameters(), lr=args.lr_critic, momentum=0.9)
        self.optimizer_actor = torch.optim.Adam(self.actor_network.parameters(), lr=args.lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic_network.parameters(), lr=args.lr_critic)
        self.criterion = nn.HuberLoss()
        self.replay_buffer = ReplayBuffer(args.replay_buffer_capacity)
        self.update_cnt = 0

    def saveModel(self):
        torch.save(self.actor_network.state_dict(), os.path.join(self.args.model_dir, f'actor_model_{self.update_cnt}.pth'))
        torch.save(self.critic_network.state_dict(), os.path.join(self.args.model_dir, f'critic_model_{self.update_cnt}.pth'))

    def select_action(self, task_node, task_data, compute_node, task_mask, compute_node_mask):
        task_node = torch.FloatTensor(task_node).to(self.device).unsqueeze(0)
        task_data = torch.FloatTensor(task_data).to(self.device).unsqueeze(0)
        compute_node = torch.FloatTensor(compute_node).to(self.device).unsqueeze(0)
        task_mask = torch.FloatTensor(task_mask).to(self.device).unsqueeze(0)
        compute_node_mask = torch.FloatTensor(compute_node_mask).to(self.device).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.actor_network(task_node, task_data, compute_node, task_mask, compute_node_mask).squeeze(0)
        q_values = q_values.reshape(self.m1 * self.max_tasks, self.m2+1)
        # add noise
        q_values += torch.randn_like(q_values) * self.args.epsilon 

        # action_probs = F.gumbel_softmax(q_values, tau=self.args.gumbel_tau, hard=False) # tau is a temperature parameter
        action_probs = F.softmax(q_values, dim=1)
        # 把compute_node_mask对应的action概率置为0。需要注意的是，action_probs的第一列是本地计算的概率，所以不需要置为0
        action_probs_1col = action_probs[:, 1:] # [m1 * max_tasks, m2]
        action_probs_1col = action_probs_1col * compute_node_mask.squeeze(0)
        action_probs = torch.cat([action_probs[:, 0].view(-1, 1), action_probs_1col], dim=1)
        # action_probs最后维度都是0，则全部设为1
        action_probs = action_probs + 1e-5
        # 按照概率重新采样
        actions = torch.multinomial(action_probs, 1) # [m1 * max_tasks, 1]
        # turn to np
        actions = actions.cpu().numpy()
        action_probs = action_probs.cpu().detach().numpy()
        return actions, action_probs
    
    def add_experience(self, task_node, task_data, compute_node, task_mask, compute_node_mask, action, reward, next_task_node, next_task_data, next_compute_node, next_task_mask, next_compute_node_mask, done):
        self.replay_buffer.add((task_node, task_data, compute_node, task_mask, compute_node_mask, action, reward, next_task_node, next_task_data, next_compute_node, next_task_mask, next_compute_node_mask, done))
    
    def update(self, writer):
        # if test, return
        if self.args.mode == 'test':
            return
        batch_size = self.args.batch_size
        if self.replay_buffer.size() < batch_size:
            return
        self.update_cnt += 1
        if self.update_cnt % self.args.replay_buffer_update_freq != 0:
            return
        if self.update_cnt % self.args.save_model_freq == 0:
            self.saveModel()
        experiences = self.replay_buffer.sample(batch_size)
        task_node, task_data, compute_node, task_mask, compute_node_mask, action, reward, next_task_node, next_task_data, next_compute_node, next_task_mask, next_compute_node_mask, done = zip(*experiences)
        task_node = torch.FloatTensor(np.asarray(task_node)).to(self.device)
        task_data = torch.FloatTensor(np.asarray(task_data, dtype=np.float32)).to(self.device).squeeze()
        compute_node = torch.FloatTensor(np.asarray(compute_node)).to(self.device)
        task_mask = torch.FloatTensor(np.asarray(task_mask)).to(self.device)
        compute_node_mask = torch.FloatTensor(np.asarray(compute_node_mask)).to(self.device)
        action = torch.FloatTensor(np.asarray(action)).to(self.device) # [batch_size, m1, max_tasks]
        action = action.view(batch_size, -1) # [batch_size, m1 * max_tasks]
        reward = torch.FloatTensor(np.asarray(reward, dtype=np.float32)).to(self.device)
        next_task_node = torch.FloatTensor(np.asarray(next_task_node)).to(self.device)
        next_task_data = torch.FloatTensor(np.asarray(next_task_data)).to(self.device)
        next_compute_node = torch.FloatTensor(np.asarray(next_compute_node)).to(self.device)
        next_task_mask = torch.FloatTensor(np.asarray(next_task_mask)).to(self.device)
        next_compute_node_mask = torch.FloatTensor(np.asarray(next_compute_node_mask)).to(self.device)
        done = torch.FloatTensor(np.asarray(done, dtype=np.float32)).to(self.device)

        certain_reward = (reward).squeeze()
        action = action.view(batch_size, -1)  # [batch_size, m1 * max_tasks*(m2+1)]

        # ---------------------- Optimize Critic ----------------------
        with torch.no_grad():
            next_action_logits = self.target_actor_network(next_task_node, next_task_data, next_compute_node, next_task_mask, next_compute_node_mask)
            next_action_logits = next_action_logits.view(batch_size, self.m1 * self.max_tasks, self.m2 + 1)
            # next_action_probs = F.gumbel_softmax(next_action_logits, tau=self.args.gumbel_tau, hard=False)
            next_action_probs = F.softmax(next_action_logits, dim=2)
            next_action_probs_1col = next_action_probs[:, :, 1:] # [batch_size, m1 * max_tasks, m2]
            next_action_probs_1col = next_action_probs_1col * next_compute_node_mask
            next_action_probs = torch.cat([next_action_probs[:, :, 0].view(batch_size, -1, 1), next_action_probs_1col], dim=2)
            target_q = self.target_critic_network(next_task_node, next_task_data, next_compute_node, next_task_mask, next_compute_node_mask, next_action_probs.view(batch_size, -1))
            target_q = certain_reward + (1 - done) * self.gamma * target_q.squeeze()

        current_q = self.critic_network(task_node, task_data, compute_node, task_mask, compute_node_mask, action.float()) # Use float() to match the expected input type of critic

        critic_loss = self.criterion(current_q.squeeze(), target_q)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # ---------------------- Optimize Actor ----------------------
        # Compute actor loss
        action_with_grad = self.actor_network(task_node, task_data, compute_node, task_mask, compute_node_mask)
        # mask
        action_with_grad = action_with_grad.view(batch_size, self.m1 * self.max_tasks, self.m2 + 1)
        action_with_grad_1col = action_with_grad[:, :, 1:]
        action_with_grad_1col = action_with_grad_1col * compute_node_mask
        action_with_grad = torch.cat([action_with_grad[:, :, 0].view(batch_size, -1, 1), action_with_grad_1col], dim=2)
        
        actor_q = self.critic_network(task_node, task_data, compute_node, task_mask, compute_node_mask, action_with_grad.view(batch_size, -1))
        actor_loss = -actor_q.mean()

        # Update the actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # ------------------- Update Target Networks -------------------
        self.soft_update(self.target_actor_network, self.actor_network)
        self.soft_update(self.target_critic_network, self.critic_network)
        
        writer.add_scalar('Loss/Actor', actor_loss.item(), self.update_cnt)
        writer.add_scalar('Loss/Critic', critic_loss.item(), self.update_cnt)

    def soft_update(self, source_network, target_network):
        # 目标网络的软更新：θ_target = τ * θ_source + (1 - τ) * θ_target
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)