import os

from .MAPPO_model import MAPPO
import numpy as np
import torch


class MAPPO_Env:

    def __init__(self,dim_args, train_args):
        # 模型文件路径
        # current_dir = os.path.dirname(os.path.abspath(__file__))
        # self.model_base_dir = os.path.join(current_dir, "model")
        self.model_base_dir=train_args.model_base_dir

        # 实例化 MAPPO
        self.agent = MAPPO(dim_args, train_args)

    def takeAction(self, state):
        # 状态state时做动作选择
        actions = self.agent.take_action(state)
        return actions

    def addExperience(self, agent_id, state, action, reward, next_state):
        # 添加经验池
        self.agent.remember(agent_id,state, action, reward, next_state)

    def train(self):
        a_loss,c_loss=self.agent.update()
        return a_loss,c_loss

    def saveModel(self,episode,final=False):
        self.agent.save_models(episode,self.model_base_dir,final)

    def loadModel(self,episode,final=False):
        self.agent.load_models(episode,self.model_base_dir,final)
