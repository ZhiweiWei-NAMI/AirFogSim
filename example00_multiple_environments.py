from airfogsim import AirFogSimEnv, BaseAlgorithmModule, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
config = load_config(config_path)

# 2. Create the environment
envs = [AirFogSimEnv(config) for _ in range(2)]

# 3. Get algorithm module
algorithm_module = BaseAlgorithmModule()
# algorithm_module -1:N-> envs
for env in envs:
    algorithm_module.initialize(env)
accumulated_reward = {env.airfogsim_label: 0 for env in envs}
cnt = 0
while not env.isDone():
    cnt += 1
    for env in envs:
        algorithm_module.scheduleStep(env)
        env.step()
        accumulated_reward[env.airfogsim_label] += algorithm_module.getReward(env)
    if cnt % 10 == 0:
        print(f"Simulation time: {env.simulation_time}, ACC_Reward: {accumulated_reward}")
for env in envs:
    env.close()