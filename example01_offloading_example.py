from airfogsim import AirFogSimEnv, BaseAlgorithmModule
import numpy as np
import yaml
import sys
from airfogsim.scheduler import RewardScheduler, TaskScheduler

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config
    

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
config = load_config(config_path)

# 2. Create the environment
# env = AirFogSimEnv(config, interactive_mode='graphic')
env = AirFogSimEnv(config, interactive_mode=None)

# 3. Get algorithm module
algorithm_module = BaseAlgorithmModule()
algorithm_module.initialize(env)
RewardScheduler.setModel(env, 'REWARD', '-task_delay')
accumulated_reward = 0
while not env.isDone():
    algorithm_module.scheduleStep(env)
    env.step()
    accumulated_reward += algorithm_module.getRewardByTask(env)
    task_num = TaskScheduler.getDoneTaskNum(env)
    print(f"Simulation time: {env.simulation_time}, ACC_Reward: {accumulated_reward/max(1,task_num)}", end='\r')
    env.render()
env.close()