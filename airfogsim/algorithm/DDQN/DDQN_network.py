
import torch
from torch import nn

# ----------------------------------- #
# 构造网络，训练网络和目标网络共用该结构
# ----------------------------------- #
class Net(nn.Module):
    def __init__(self, dim_states, dim_hiddens, dim_actions):
        super(Net, self).__init__()
        # 有两个隐含层
        self.fc1 = nn.Linear(dim_states, dim_hiddens)
        self.fc2 = nn.Linear(dim_hiddens, dim_hiddens)
        self.fc3 = nn.Linear(dim_hiddens, dim_actions)

    # 前向传播
    def forward(self, x):
        x = self.fc1(x)  # [b,dim_states]-->[b,dim_hiddens]
        x = self.fc2(x)  # [b,dim_hiddens]-->[b,dim_hiddens]
        x = self.fc3(x)  # [b,dim_hiddens]-->[b,dim_actions]
        return x

    def save_model(self, file_dir):
        torch.save(self.state_dict(), file_dir, _use_new_zipfile_serialization=False)

    def load_model(self, file_dir):
        self.load_state_dict(torch.load(file_dir))