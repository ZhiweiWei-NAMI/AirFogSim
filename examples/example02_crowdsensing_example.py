import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

os.environ['useCUPY'] = 'False'
print('useCUPY:', os.environ['useCUPY'])
# When n_RB < 50, numpy is better than cupy; When n_RB >= 50, cupy is better than numpy.

from airfogsim import AirFogSimEnv, AirFogSimEvaluation
from baselines.crowdsensing.greedy.greedy_algorithm import GreedyAlgorithmModule
import numpy as np
import yaml
from pyinstrument import Profiler

root=os.path.abspath(__file__)



def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config


last_episode = 0
max_episode = 1

# 启动性能监控
profiler = Profiler()
profiler.start()

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. Get algorithm module
# algorithm_module = DDQNAlgorithmModule(last_episode)
algorithm_module = GreedyAlgorithmModule()


# 3. Create the new environment
env = AirFogSimEnv(config, interactive_mode='graphic')

# 4. Initialize the algorithm module (initialize in every episode)
algorithm_module.initialize(env)

# 5. Create the evaluation module
evaluation_module = AirFogSimEvaluation()

for episode in range(last_episode + 1, max_episode+1):
    env.reset()
    algorithm_module.reset(env)

    while not env.isDone():
        algorithm_module.scheduleStep(env)
        env.step()
        # accumulated_reward += algorithm_module.getRewardByMission(env)
        # print(f"Simulation time: {env.simulation_time}", end='\r')
        # print(f"Simulation time: {env.simulation_time}, ACC_Reward: {accumulated_reward}")
        env.render()
        # algorithm_module.updateExperience(env)
        # algorithm_module.DDQN_env.train()
        evaluation_module.updateEvaluationIndicators(env, algorithm_module)
        evaluation_module.addToStepRecord()
        # evaluation_module.printEvaluation()

    # algorithm_module.DDQN_env.saveModel(episode)
    evaluation_module.toFile(episode)
    evaluation_module.addToEpisodeRecord()
    evaluation_module.drawAndResetStepRecord(episode)

evaluation_module.drawAndResetEpisodeRecord()
env.close()
# 结束性能监控并打印报告
profiler.stop()
profiler.print()
