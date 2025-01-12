import sys
import os
import faulthandler
faulthandler.enable()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
dir_name = os.path.dirname(__file__)

os.environ['useCUPY'] = 'False'
print('useCUPY:', os.environ['useCUPY'])
# When n_RB < 50, numpy is better than cupy; When n_RB >= 50, cupy is better than numpy.

from airfogsim import AirFogSimEnv, AirFogSimEvaluation
from baselines.crowdsensing.greedy.greedy_algorithm import GreedyAlgorithmModule
from baselines.crowdsensing.RL.TDDQN_MADDPG_algorithm import TransDDQN_MADDPG_AlgorithmModule
import numpy as np
import yaml
from pyinstrument import Profiler

root = os.path.abspath(__file__)


def load_config(path):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        return config


last_episode = 74
max_episode = 200

# 启动性能监控
profiler = Profiler()
profiler.start()

# 1. Load the configuration file
config_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), 'config.yaml')
config = load_config(config_path)

# 2. Get algorithm module
algorithm_module = GreedyAlgorithmModule()
# algorithm_module = TransDDQN_MADDPG_AlgorithmModule()
algorithm_tag = algorithm_module.getAlgorithmTag()

# 3. Create the new environment
env = AirFogSimEnv(config, interactive_mode=None)
# env = AirFogSimEnv(config, interactive_mode='graphic')

# 4. Initialize the algorithm module (initialize in every episode)
algorithm_module.initialize(env)

# 5. Create the evaluation module
evaluation_module = AirFogSimEvaluation(algorithm_module.getAlgorithmTag())

for episode in range(last_episode + 1, max_episode + 1):
    if algorithm_tag == 'Greedy':
        while not env.isDone():
            algorithm_module.scheduleStep(env)
            env.step()

            acc_reward = evaluation_module.getAccReward()
            avg_reward = evaluation_module.getAvgReward()
            succ_ratio = evaluation_module.getCompletionRatio()
            print(f"Simulation time: {env.simulation_time:.2f}, ACC_Reward: {acc_reward:.2f}, AVG_Reward: {avg_reward:.2f}, SUCC_Ratio: {succ_ratio:.2f}")
            env.render()
            evaluation_module.updateEvaluationIndicators(env, algorithm_module)
            evaluation_module.addToStepRecord()
            # evaluation_module.printEvaluation()

        evaluation_module.episodeRecordToFile(episode)
        evaluation_module.addToEpisodeRecord()
        evaluation_module.drawAndResetStepRecord(episode)

    elif algorithm_tag == 'TDDQN_MADDPG':
        while not env.isDone():
            algorithm_module.scheduleStep(env)
            env.step()

            acc_reward = evaluation_module.getAccReward()
            avg_reward = evaluation_module.getAvgReward()
            succ_ratio = evaluation_module.getCompletionRatio()
            print(f"Simulation time: {env.simulation_time:.2f}, ACC_Reward: {acc_reward:.2f}, AVG_Reward: {avg_reward:.2f}, SUCC_Ratio: {succ_ratio:.2f}")
            env.render()

            algorithm_module.updateTAExperience(env)
            algorithm_module.TransDDQN_env.train()
            algorithm_module.updatePPExperience(env)
            algorithm_module.MADDPG_env.train()

            evaluation_module.updateEvaluationIndicators(env, algorithm_module)
            evaluation_module.addToStepRecord()
            # evaluation_module.printEvaluation()

        algorithm_module.TransDDQN_env.saveModel(episode)
        algorithm_module.MADDPG_env.saveModel(episode)
        evaluation_module.episodeRecordToFile(episode)
        evaluation_module.addToEpisodeRecord()
        evaluation_module.drawAndResetStepRecord(episode)

    # 构建车辆数据时不能reset，且只需要一个episode
    env.reset()
    algorithm_module.reset(env)

# evaluation_module.drawAndResetEpisodeRecord()
evaluation_module.drawEpisodeRecordsByFile()
env.close()
# 结束性能监控并打印报告
profiler.stop()
profiler.print()
