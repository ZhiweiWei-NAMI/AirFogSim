import sys
import os
import torch
import numpy as np
import random
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
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
from airfogsim import AirFogSimEnv
import yaml
from airfogsim.scheduler import RewardScheduler, TaskScheduler
from benchmarks.trans_dqn_offloading.dqn_algorithm import DQNOffloadingAlgorithm

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

# 1. Load the configuration file
    
config_path = os.path.join(os.path.dirname(__file__), '../configs/DAG_offloading_config.yaml')
config = load_config(config_path)

# 2. Create the environment
# env = AirFogSimEnv(config, interactive_mode='graphic')
env = AirFogSimEnv(config, interactive_mode=None)
# 3. Get algorithm module
algorithm_module = DQNOffloadingAlgorithm()
algorithm_module.initialize(env, config)
RewardScheduler.setModel(env, 'REWARD', '1/max(1e-1, task_delay)')
# RewardScheduler.setModel(env, 'REWARD', 'Piecewise((log(1 + task_deadline - task_delay) + 1, task_delay < task_deadline), (exp((task_deadline - task_delay)), True))')
EPOCH_NUM = 1000
for epoch in range(EPOCH_NUM):
    accumulated_reward = 0
    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()
        accumulated_reward += algorithm_module.getRewardByTask(env)
        task_num = TaskScheduler.getDoneTaskNum(env)
        # task_num = TaskScheduler.getDoneTaskNumLessThanSeconds(env, 2)
        total_task_num = TaskScheduler.getTotalTaskNum(env)
        succ_ratio = task_num / max(1,total_task_num)
        # env.render()
    algorithm_module.tensorboard_writer.add_scalar('Reward', accumulated_reward, env.simulation_time + epoch * env.max_simulation_time)
    algorithm_module.tensorboard_writer.add_scalar('Success ratio', succ_ratio, env.simulation_time + epoch * env.max_simulation_time)
    
    print(f'Epoch: {epoch}, Simulation time: {env.simulation_time:.2f}, Ratio: {succ_ratio:.2f} = {task_num}/{total_task_num}, Reward: {accumulated_reward:.2f}, TaskNodeNum: {len(env.task_node_ids)}, VNum: {len(env.vehicle_ids_as_index)}')
    
    env.reset()
    algorithm_module.reset()
algorithm_module.saveModel()
env.close()