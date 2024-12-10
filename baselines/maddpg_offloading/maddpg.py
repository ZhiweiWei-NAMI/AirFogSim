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

class MaddpgOffloadingAlgorithm(BaseAlgorithmModule):
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

    def initialize(self, env: AirFogSimEnv):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD', '-task_delay')
        # 1. 获取一个template的fog node和task node信息，建模并且初始化state_dim和action_dim
        fog_node_template = self.entityScheduler.getFogNodeTemplate(env)

    def scheduleStep(self, env: AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleReturning(env)
        self.scheduleOffloading(env)
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleMission(env)
        self.scheduleTraffic(env)

    def scheduleMission(self, env: AirFogSimEnv):
        """The mission scheduling logic.
        Mission: Missions assigned to both vehicles and UAVs, each type has a probability of sum of 1.
        Sensor: Assigned to vehicle, select the sensor closest to PoI from the idle sensors with accuracy higher than required(Distance First).
                Assigned to RSU, select the sensor with the lowest accuracy from the idle sensors with accuracy higher than required(Accuracy Lowerbound).

        Args:
            env (AirFogSimEnv): The environment object.

        """
        return

    def scheduleReturning(self, env: AirFogSimEnv):
        """The returning logic. Relay or direct is controlled by probability.
        Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
        Direct, select nearest RSU, return_route=[RSU]

        Args:
            env (AirFogSimEnv): The environment object.
        """
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                current_node_id = task.getCurrentNodeId()
                current_node_type = self.entityScheduler.getNodeTypeById(env, current_node_id)
                vehicle_num = self.entityScheduler.getNodeNumByType(env, 'V')
                UAV_num = self.entityScheduler.getNodeNumByType(env, 'U')
                RSU_num = self.entityScheduler.getNodeNumByType(env, 'R')
                if current_node_type == 'V':
                    if UAV_num > 0:
                        V2U_distance = np.zeros((UAV_num))
                        for u_idx in range(UAV_num):
                            u_id = self.entityScheduler.getNodeInfoByIndexAndType(env, u_idx, 'U')['id']
                            distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, u_id)
                            V2U_distance[u_idx] = distance
                        nearest_u_distance = np.max(V2U_distance)
                        nearest_u_idx = np.unravel_index(np.argmax(V2U_distance), V2U_distance.shape)
                        nearest_u_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_u_idx[0]), 'U')[
                            'id']

                    if RSU_num > 0:
                        V2R_distance = np.zeros((RSU_num))
                        for r_idx in range(RSU_num):
                            r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'R')['id']
                            distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                            V2R_distance[r_idx] = distance
                        nearest_r_distance = np.max(V2R_distance)
                        nearest_r_idx = np.unravel_index(np.argmax(V2R_distance), V2R_distance.shape)
                        nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'R')[
                            'id']

                    relay_probability = 0.5
                    if random.random() < relay_probability and UAV_num > 0:
                        return_route = [nearest_u_id, nearest_r_id]
                    else:
                        return_route = [nearest_r_id]

                elif current_node_type == 'U':
                    U2R_distance = np.zeros((RSU_num))
                    for r_idx in range(RSU_num):
                        r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'R')['id']
                        distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                        U2R_distance[r_idx] = distance
                    nearest_r_distance = np.max(U2R_distance)
                    nearest_r_idx = np.unravel_index(np.argmax(U2R_distance), U2R_distance.shape)
                    nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'R')['id']
                    return_route = [nearest_r_id]
                else:
                    raise TypeError('Node type is invalid')

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
            target_position = self.trafficScheduler.getNextPositionOfUav(UAV_id)
            if target_position is None:
                # 悬停
                mobility_pattern = {'angle': 0, 'phi': 0, 'speed': 0}
                UAVs_mobile_pattern[UAV_id] = mobility_pattern
            else:
                # Update stay time at first position in route
                distance_threshold = self.trafficScheduler.getConfig(env, 'distance_threshold')
                distance = np.linalg.norm(np.array(target_position) - np.array(current_position))
                if distance < distance_threshold:
                    self.trafficScheduler.updateRoute(env, UAV_id, self.trafficScheduler.getTrafficInterval())

                delta_x = target_position[0] - current_position[0]
                delta_y = target_position[1] - current_position[1]
                delta_z = target_position[2] - current_position[2]

                # 计算 xy 平面的方位角
                angle = np.arctan2(delta_y, delta_x)

                # 计算 z 相对于 xy 平面的仰角
                distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
                phi = np.arctan2(delta_z, distance_xy)

                mobility_pattern = {'angle': angle, 'phi': phi}
                UAV_speed_range = self.trafficScheduler.getConfig(env, 'UAV_speed_range')
                mobility_pattern['speed'] = random.uniform(UAV_speed_range[0], UAV_speed_range[1])
                UAVs_mobile_pattern[UAV_id] = mobility_pattern
        self.trafficScheduler.setUAVMobilityPatterns(env, UAVs_mobile_pattern)

    def scheduleOffloading(self, env: AirFogSimEnv):
        super().scheduleOffloading(env)

    def scheduleCommunication(self, env: AirFogSimEnv):
        super().scheduleCommunication(env)

    def scheduleComputing(self, env: AirFogSimEnv):
        super().scheduleComputing(env)

    def getRewardByTask(self, env: AirFogSimEnv):
        return super().getRewardByTask(env)

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)
