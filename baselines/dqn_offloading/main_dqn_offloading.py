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
from airfogsim.scheduler import RewardScheduler, TaskScheduler, EntityScheduler
from baselines.dqn_offloading.dqn_algorithm import DQNOffloadingAlgorithm

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

# 1. Load the configuration file
    
config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'dqn_airfogsim_config.yaml')
config = load_config(config_path)

# 2. Create the environment
# env = AirFogSimEnv(config, interactive_mode='graphic')
env = AirFogSimEnv(config, interactive_mode=None)
# 3. Get algorithm module
algorithm_module = DQNOffloadingAlgorithm()
algorithm_module.initialize(env, config)
RewardScheduler.setModel(env, 'REWARD', 'max((1+task_priority)*log(1+max(-0.9, task_deadline-task_delay))+0.2, 0.2*exp(task_deadline-task_delay))')
np.random.seed(0)
random.seed(0)
EPOCH_NUM = 500
for epoch in range(EPOCH_NUM):
    accumulated_reward = 0
    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()
        accumulated_reward += algorithm_module.getRewardByTask(env)
        task_num = TaskScheduler.getDoneTaskNum(env)
        total_task_num = TaskScheduler.getTotalTaskNum(env)
        succ_ratio = task_num / max(1,total_task_num)
        # veh_num = EntityScheduler.getNodeNumByType(env, 'vehicle')
        # uav_num = EntityScheduler.getNodeNumByType(env, 'uav')
        env.render()
        print(f'Epoch: {epoch}, Simulation time: {env.simulation_time}, Ratio: {succ_ratio} = {task_num}/{total_task_num}, Reward: {accumulated_reward}', end='\r')
    print()
    env.reset()
    algorithm_module.reset()
algorithm_module.saveModel()
env.close()