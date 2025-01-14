
import torch
from torch import nn
from torch.nn import functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim, 10240)
        self.l2 = nn.Linear(10240, 4096)
        self.l3 = nn.Linear(4096, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = self.l3(a)
        # a: [batch_size, m1 * max_tasks * (m2+1)]
        return a
    
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 10240)
        self.l2 = nn.Linear(10240, 4096)
        self.l3 = nn.Linear(4096, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q = F.relu(self.l1(sa))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        # q: [batch_size, 1]
        return q