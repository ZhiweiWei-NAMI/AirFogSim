from .trans_network import TransformerActor, TransformerCritic
import torch
import torch.nn as nn
import numpy as np

class MADDPG_Agent:
    def __init__(self, agent_id, obs_shape, act_shape, args):
        self.agent_id = agent_id
        self.args = args
        self.actor = TransformerActor(obs_shape, act_shape, args)
        self.critic = TransformerCritic(obs_shape, act_shape, args)
        self.target_actor = TransformerActor(obs_shape, act_shape, args)
        self.target_critic = TransformerCritic(obs_shape, act_shape, args)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)
        self.critic_loss = nn.MSELoss()
        self.act_limit = args.act_limit # action limit, 1.0
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = args.device

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        action = self.actor(obs)
        action = action.cpu().data.numpy()
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, replay_buffer, batch_size):
        obs, act, rew, next_obs, done = replay_buffer.sample(batch_size)
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        rew = torch.FloatTensor(rew).to(self.device).unsqueeze(-1)
        next_obs = torch.FloatTensor(next_obs).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(-1)

        with torch.no_grad():
            target_act = self.target_actor(next_obs)
            target_q = self.target_critic(next_obs, target_act)
            target_q = rew + self.gamma * (1 - done) * target_q

        current_q = self.critic(obs, act)
        critic_loss = self.critic_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)