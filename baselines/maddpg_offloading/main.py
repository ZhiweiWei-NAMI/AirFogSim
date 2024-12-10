import sys
import os
# 直到airfogsim的根目录
isAirFogSim = False
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
cnt = 0
while not isAirFogSim:
    cnt += 1
    if 'airfogsim' in os.listdir(root_path) or cnt > 10:
        isAirFogSim = True
    else:
        root_path = os.path.abspath(os.path.join(root_path, os.path.pardir))
sys.path.append(root_path)
dir_name = os.path.dirname(__file__)
from airfogsim import AirFogSimEnv, BaseAlgorithmModule
import numpy as np
import random
import yaml
from airfogsim.scheduler import RewardScheduler, TaskScheduler
from .maddpg import MaddpgOffloadingAlgorithm

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

# 1. Load the configuration file
    
config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'maddpg_airfogsim_config.yaml')
config = load_config(config_path)

# 2. Create the environment
# env = AirFogSimEnv(config, interactive_mode='graphic')
env = AirFogSimEnv(config, interactive_mode=None)

# 3. Get algorithm module
algorithm_module = BaseAlgorithmModule()
algorithm_module.initialize(env)
RewardScheduler.setModel(env, 'REWARD', '1/task_delay')
accumulated_reward = 0
np.random.seed(0)
random.seed(0)
while not env.isDone():
    algorithm_module.scheduleStep(env)
    env.step()
    accumulated_reward += algorithm_module.getRewardByTask(env)
    task_num = TaskScheduler.getDoneTaskNum(env)
    out_of_ddl_task_num = TaskScheduler.getOutOfDDLTasks(env)
    succ_ratio = task_num / max(1,task_num + out_of_ddl_task_num)
    env.render()
    print(f'Simulation time: {env.simulation_time}, ACC_Reward: {succ_ratio*accumulated_reward/max(1,task_num)}', end='\r')
env.close()