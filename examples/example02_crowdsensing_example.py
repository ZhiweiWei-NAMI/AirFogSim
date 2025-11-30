import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

os.environ['useCUPY'] = 'False'
print('useCUPY:',os.environ['useCUPY'])
# When n_RB < 50, numpy is better than cupy; When n_RB >= 50, cupy is better than numpy.

from airfogsim import AirFogSimEnv, BaseAlgorithmModule, NVHAUAlgorithmModule, AirFogSimEvaluation
import numpy as np
import yaml
from pyinstrument import Profiler

def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config

# 启动性能监控
profiler=Profiler()
profiler.start()

# 1. Load the configuration file
config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. Create the environment
# env = AirFogSimEnv(config, interactive_mode='graphic')
env = AirFogSimEnv(config, interactive_mode=None)

# 3. Get algorithm module
algorithm_module = NVHAUAlgorithmModule()
algorithm_module.initialize(env)

# 4. Create the evaluation module
evaluation_module=AirFogSimEvaluation()

while not env.isDone():
    algorithm_module.scheduleStep(env)
    env.step()
    # accumulated_reward += algorithm_module.getRewardByMission(env)
    print(f"Simulation time: {env.simulation_time}", end='\r')
    # print(f"Simulation time: {env.simulation_time}, ACC_Reward: {accumulated_reward}")
    env.render()
    # evaluation_module.updateEvaluationIndicators(env,algorithm_module)
    # evaluation_module.printEvaluation()
env.close()

# 结束性能监控并打印报告
profiler.stop()
profiler.print()