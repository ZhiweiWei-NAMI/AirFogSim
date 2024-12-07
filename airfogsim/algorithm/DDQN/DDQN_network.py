
import torch
from torch import nn

# ----------------------------------- #
# 构造网络，训练网络和目标网络共用该结构
# ----------------------------------- #
class Net(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(Net, self).__init__()
        # 有两个隐含层
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = self.fc2(x)  # [b,n_hiddens]-->[b,n_hiddens]
        x = self.fc3(x)  # [b,n_hiddens]-->[b,n_actions]
        return x

    def save_model(self, file_dir):
        torch.save(self.state_dict(), file_dir, _use_new_zipfile_serialization=False)

    def load_model(self, file_dir):
        self.load_state_dict(torch.load(file_dir))