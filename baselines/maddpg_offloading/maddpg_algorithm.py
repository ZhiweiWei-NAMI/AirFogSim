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

import random

from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
# from .algorithm.DDQN.DDQN_env import  DDQN_Env
import numpy as np

class MADDPGOffloadingAlgorithm(BaseAlgorithmModule):
    """
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleReturning: Relay (only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct, select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: 
        UAV: Fly to next position in route list and stay for a period of time.
    '''

    def __init__(self):
        super().__init__()

    def initialize(self, env, config):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.min_position_x, self.max_position_x = 0, 2000
        self.min_position_y, self.max_position_y = 0, 2000
        self.min_position_z, self.max_position_z = 0, 200
        self.min_speed, self.max_speed = 0, 20
        self.task_min_cpu, self.task_max_cpu = config['task'].get('task_min_cpu', 0), config['task'].get('task_max_cpu', 1)
        self.task_min_size, self.task_max_size = config['task'].get('task_min_size', 0), config['task'].get('task_max_size', 1)
        self.task_min_deadline, self.task_max_deadline = config['task'].get('task_min_deadline', 0), config['task'].get('task_max_deadline', 1)
        self.task_min_priority, self.task_max_priority = config['task'].get('task_min_priority', 0), config['task'].get('task_max_priority', 1)
        self.rewardScheduler.setModel(env, 'REWARD', '-task_priority*task_delay')
        self.node_dim = 6
        self.task_dim = 7
        self.fog_type_dict = {
            'V': 0,
            'U': 1,
            'I': 2,
            'C': 3,
        }
        # # lifecycle includes to_generate, to_offload, to_return, computing, computed, returning, finished
        self.task_lifecycle_state_dict = {
            'to_generate': 0,
            'to_offload': 1,
            'to_return': 2,
            'computing': 3,
            'computed': 4,
            'returning': 5,
            'finished': 6
        }
            
    def _encode_node_state(self, node_state, node_type):
        # id, time, position_x, position_y, position_z, speed, fog_profile, node_type
        # ['UAV_9' 0 2655.3572089655477 1030.70353000605 100.0 0 {'lambda': 1} 'U']
        # 选取 position_x, position_y, position_z, speed, fog_profile, node_type, 6维
        # 注意，fog_profile要转为数字；node_type要转为encoding；position_x, position_y, position_z, speed要normalize
        # fog_profile: {'lambda': 1} -> 1
        if node_type == 'FN':
            profile = node_state[5].get('cpu', 0)
        elif node_type == 'TN':
            profile = node_state[5].get('lambda', 0)
        fog_type = self.fog_type_dict.get(node_state[6], -1)
        position_x = (node_state[2] - self.min_position_x) / (self.max_position_x - self.min_position_x)
        position_y = (node_state[3] - self.min_position_y) / (self.max_position_y - self.min_position_y)
        position_z = (node_state[4] - self.min_position_z) / (self.max_position_z - self.min_position_z)
        speed = (node_state[5] - self.min_speed) / (self.max_speed - self.min_speed)
        state = [position_x, position_y, position_z, speed, profile, fog_type]
        return np.asarray(state)
    
    def _encode_task_state(self, task_state):
        # ['task_id', 'time', 'task_node_id', 'task_size', 'task_cpu', 'required_returned_size', 'task_deadline', 'task_priority', 'task_arrival_time', 'task_lifecycle_state']
        # 选取 task_size, task_cpu, required_returned_size, task_deadline, task_priority, task_arrival_time, 6维
        # 注意，task_size, task_cpu, required_returned_size要normalize
        task_size = (task_state[3] - self.task_min_size) / (self.task_max_size - self.task_min_size)
        task_cpu = (task_state[4] - self.task_min_cpu) / (self.task_max_cpu - self.task_min_cpu)
        required_returned_size = (task_state[5] - self.task_min_size) / (self.task_max_size - self.task_min_size)
        task_deadline = (task_state[6] - self.task_min_deadline) / (self.task_max_deadline - self.task_min_deadline)
        task_priority = (task_state[7] - self.task_min_priority) / (self.task_max_priority - self.task_min_priority)
        task_arrival_time = (task_state[1] - task_state[8]) / task_deadline # time - task_arrival_time, 即任务已经等待的时间
        task_lifecycle_state = self.task_lifecycle_state_dict.get(task_state[9], -1)
        state = [task_size, task_cpu, required_returned_size, task_deadline, task_priority, task_arrival_time, task_lifecycle_state]
        return np.asarray(state)


    def scheduleStep(self, env: AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleReturning(env)
        self.scheduleOffloading(env) # 这里使用MADDPG
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleMission(env)
        self.scheduleTraffic(env)

    def scheduleMission(self, env: AirFogSimEnv):
        return

    def scheduleReturning(self, env: AirFogSimEnv):
        # 直接返回给task node
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                return_route = [task.getTaskNodeId()]
                self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)

    def scheduleTraffic(self, env: AirFogSimEnv):
        """The UAV traffic scheduling logic. Should be implemented by the subclass. Default is move to the nearest
         mission sensing or task position. If there is no mission allocated to UAV, movement is random.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        UAVs_info = self.trafficScheduler.getUAVTrafficInfos(env)
        UAVs_mobile_pattern = {}
        for UAV_id, UAV_info in UAVs_info.items():
            current_position = UAV_info['position']
            target_position = self.trafficScheduler.getRandomTargetPositionForUAV(env, UAV_id)
            mobility_pattern = self.trafficScheduler.getDefaultUAVMobilityPattern(env, UAV_id, current_position, target_position)
            UAVs_mobile_pattern[UAV_id] = mobility_pattern
        self.trafficScheduler.setUAVMobilityPatterns(env, UAVs_mobile_pattern)

    def scheduleOffloading(self, env: AirFogSimEnv):
        all_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        for task_dict in all_task_infos:
            task_node_id = task_dict['task_node_id']
            task_id = task_dict['task_id']
            neighbor_infos = self.entityScheduler.getNeighborNodeInfosById(env, task_node_id, sorted_by='distance', max_num=5)
            if len(neighbor_infos) > 0:
                nearest_node_id = neighbor_infos[0]['id']
                furthest_node_id = neighbor_infos[-1]['id']
                flag = self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, nearest_node_id)
                assert flag

    def scheduleCommunication(self, env: AirFogSimEnv):
        n_RB = self.commScheduler.getNumberOfRB(env)
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        avg_RB_nos = max(1, n_RB // max(1, len(all_offloading_task_infos)))
        RB_ctr = 0
        for task_dict in all_offloading_task_infos:
            # 从RB_ctr到RB_ctr+avg_RB_nos-1分配给task；多出的部分mod n_RB，allocated_RB_nos是RB编号的列表
            allocated_RB_nos = [(RB_ctr + i) % n_RB for i in range(avg_RB_nos)]
            RB_ctr = (RB_ctr + avg_RB_nos) % n_RB
            self.commScheduler.setCommunicationWithRB(env, task_dict['task_id'], allocated_RB_nos)

    def scheduleComputing(self, env: AirFogSimEnv):
        all_computing_task_infos = self.taskScheduler.getAllComputingTaskInfos(env)
        appointed_fog_node_set = set()
        for task_dict in all_computing_task_infos:
            task_id = task_dict['task_id']
            task_node_id = task_dict['task_node_id']
            assigned_node_id = task_dict['assigned_to']
            assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
            if assigned_node_info is None or assigned_node_id in appointed_fog_node_set:
                continue
            appointed_fog_node_set.add(assigned_node_id)
            # 所有cpu分配给task
            self.compScheduler.setComputingWithNodeCPU(env, task_id, assigned_node_info.get('fog_profile', {}).get('cpu', 0)) 

    def getRewardByTask(self, env: AirFogSimEnv):
        return super().getRewardByTask(env)

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)
