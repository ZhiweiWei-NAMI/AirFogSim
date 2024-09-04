from airfogsim import AirFogSimEnv, BaseAlgorithmModule
import numpy as np
import yaml
import sys

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
accumulated_reward = 0
while not env.isDone():
    algorithm_module.scheduleStep(env)
    env.step()
    accumulated_reward += algorithm_module.getReward(env)
    print(f"Simulation time: {env.simulation_time}, ACC_Reward: {accumulated_reward}", end='\r')
    env.render()
env.close()