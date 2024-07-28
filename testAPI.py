# 测试评估airfogsim需要的api

from airfogsim import AirFogSimEnv, AirFogSimScheduler, AirFogSimEnvVisualizer
import numpy as np
import yaml
import sys

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

config = load_config('config.yaml')

config_path = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
config = load_config(config_path)
env = AirFogSimEnv(config)
env_wrapper = AirFogSimEnvVisualizer(env, config) # 可视化

# 接下来，测试AirFogSimEnv的API
# 整体框架是，算法优化层->调度层->环境层

# 算法优化层这里不涉及，直接从调度层开始。调度层的基础设置是在AirFogSimScheduler中完成的，如果不做设置，会使用默认值
# ----------------------------------------------------------------------------------------------------------
# 计算资源调度器，包括初始化和运行中的调度
compSched = AirFogSimScheduler.getComputationScheduler()
compSched.setComputationModel('M/M/1')

# 通信资源调度器
commSched = AirFogSimScheduler.getCommunicationScheduler()
commSched.setV2VFadingModel('Rayleigh')
commSched.setV2VPathLossModel('LogDistance')
commSched.setV2VShadowingModel('LogNormal')

# 区块链资源调度器
blockchainSched = AirFogSimScheduler.getBlockchainScheduler()
blockchainSched.setBlockchainModel('PoW')

# 任务调度器
taskSched = AirFogSimScheduler.getTaskScheduler()
taskSched.setTaskArrivalModel('Poisson') # Poisson, random, etc.
taskSched.setTaskArrivalRate(0.5) # 任务到达率, per second per task node
taskSched.setTaskSizeModel('Exponential') # Exponential, uniform, etc.
taskSched.setTaskSizeRange([0.5, 1.5]) # 任务大小范围, in MB
taskSched.setTaskCPUModel('Exponential') # Exponential, uniform, etc.
taskSched.setTaskCPURange([0.5, 1.5]) # 任务CPU需求范围, in MIPS
taskSched.setTaskDeadlineModel('Exponential') # Exponential, uniform, etc.
taskSched.setTaskDeadlineRange([0.5, 1.5]) # 任务截止时间范围, in seconds

#奖励调度器，设置每一个任务的奖励，解析数学公式来进行设置
rewardSched = AirFogSimScheduler.getRewardScheduler()
rewardSched.setRewardModel('1/log(1+delay)')

# Agent调度器，为FogNode或者TaskNode设置Agent
agentSched = AirFogSimScheduler.getAgentScheduler()
agentSched.setAgentModel('DQN')

while True:
    
    # 在每一步之前，都可以通过AirFogSimEnvScheduler调整策略，即state -> action
    compSched.scheduleCPUbyFogNodeName(env, 'Fog-V_1' [0.5, 0.5])
    commSched.scheduleBandwidthbyChannelType(env, 'V2V', [0.5, 0.5])
    # 此外的一系列调度操作，需要丰富，并且可以通过算法进行优化

    
    done = env.step()
    if done:
        break
