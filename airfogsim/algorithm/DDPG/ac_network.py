
import torch
from torch import nn
from torch.nn import functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, action_dim)

    def forward(self, state):
        a = F.tanh(self.l1(state))
        a = F.tanh(self.l2(a))
        a = F.tanh(self.l3(a)) # tanh activation function, not the probability distribution
        # a: [batch_size, m1 * max_tasks * (m2+1)]
        return a
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 1024)
        self.l2 = nn.Linear(1024, 512)
        self.l3 = nn.Linear(512, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.tanh(self.l1(sa))
        q1 = F.tanh(self.l2(q1))
        q1 = self.l3(q1)
        return q1