import random

from .airfogsim_scheduler import AirFogSimScheduler
from .airfogsim_env import AirFogSimEnv
import numpy as np


class BaseAlgorithmModule:
    """Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: Not used.
    scheduleComputing: Not used.
    scheduleCommunication: Randomly allocate three RBs
    scheduleMission: 
        Mission: Missions assigned to only UAVs.
        Sensor: Use the sensor with the lowest accuracy among sensors with accuracy higher than the required accuracy.
    scheduleReturning: Select the nearest RSU.
    scheduleTraffic: UAV flys to the sensing position closest to the current location among all sensing missions assigned to oneself
    '''

    def __init__(self):
        self.compScheduler = AirFogSimScheduler.getComputationScheduler()
        self.commScheduler = AirFogSimScheduler.getCommunicationScheduler()
        self.entityScheduler = AirFogSimScheduler.getEntityScheduler()
        self.rewardScheduler = AirFogSimScheduler.getRewardScheduler()
        self.taskScheduler = AirFogSimScheduler.getTaskScheduler()
        self.blockchainScheduler = AirFogSimScheduler.getBlockchainScheduler()
        self.missionScheduler = AirFogSimScheduler.getMissionScheduler()
        self.sensorScheduler = AirFogSimScheduler.getSensorScheduler()
        self.trafficScheduler = AirFogSimScheduler.getTrafficScheduler()
        self.algorithmScheduler = AirFogSimScheduler.getAlgorithmScheduler()

    def initialize(self, env: AirFogSimEnv, config={}):
        """Initialize the algorithm with the environment. Should be implemented by the subclass. Including setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD', '-task_delay')
        self.rewardScheduler.setModel(env, 'PUNISH', '-1')

    def reset(self,env:AirFogSimEnv):
        pass

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

    def scheduleReturning(self, env: AirFogSimEnv):
        """The returning logic. Should be implemented by the subclass.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        RSU_infos = self.trafficScheduler.getRSUTrafficInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                distance_dict = {}
                current_node_id = task.getCurrentNodeId()
                for RSU_id, RSU_info in RSU_infos.items():
                    distance_dict[RSU_id] = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, RSU_id)
                distance_list = sorted(distance_dict.items(), key=lambda d: d[1],
                                       reverse=False)  # distance_list[idx][0]:key [1]:value
                return_route = [distance_list[0][0]]  # Select the nearest RSU
                self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)

    def scheduleMission(self, env: AirFogSimEnv):
        """The mission scheduling logic. Should be implemented by the subclass. Default is selecting the idle sensor
        with lowest(but higher than mission_accuracy) accuracy (Only assigned to UAV).
        
        Args:
            env (AirFogSimEnv): The environment object.

        """
        cur_time = self.trafficScheduler.getCurrentTime(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []

        for mission_profile in new_missions_profile:
            mission_sensor_type = mission_profile['mission_sensor_type']
            mission_accuracy = mission_profile['mission_accuracy']

            appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.sensorScheduler.getLowestAccurateIdleSensorOnUAV(
                env, mission_sensor_type, mission_accuracy, excluded_sensor_ids)

            if appointed_node_id != None and appointed_sensor_id != None:
                mission_profile['appointed_node_id'] = appointed_node_id
                mission_profile['appointed_sensor_id'] = appointed_sensor_id
                mission_profile['appointed_sensor_accuracy'] = appointed_sensor_accuracy
                mission_profile['mission_start_time'] = cur_time
                for _ in mission_profile['mission_routes']:
                    task_set = []
                    mission_task_profile = {
                        'task_node_id': appointed_node_id,
                        'task_deadline': mission_profile['mission_deadline'],
                        'arrival_time': mission_profile['mission_arrival_time'],
                        'return_size': mission_profile['mission_size']
                    }
                    new_task = self.taskScheduler.generateTaskOfMission(env, mission_task_profile)
                    task_set.append(new_task)
                    mission_profile['mission_task_sets'].append(task_set)
                self.missionScheduler.generateAndAddMission(env, mission_profile)

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)
        self.missionScheduler.deleteBeAssignedMissionsProfile(env, delete_mission_profile_ids)

    def scheduleTraffic(self, env: AirFogSimEnv):
        """The UAV traffic scheduling logic. Should be implemented by the subclass. Default is move to the next
         mission sensing or task position. If there is no mission allocated to UAV, movement is random.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        UAVs_info = self.trafficScheduler.getUAVTrafficInfos(env)
        UAVs_mobile_pattern = {}
        for UAV_id, UAV_info in UAVs_info.items():
            current_position = UAV_info['position']
            target_position = self.trafficScheduler.getNextPositionOfUav(env, UAV_id)

            if target_position is None:
                # 在 [0, 2π) 范围内生成一个随机角度（弧度）
                random_angle = np.random.uniform(0, 2 * np.pi)
                mobility_pattern = {}
                mobility_pattern['angle'] = random_angle
                mobility_pattern['phi'] = 0
                UAV_speed_range = self.trafficScheduler.getConfig(env, 'UAV_speed_range')
                mobility_pattern['speed'] = random.uniform(UAV_speed_range[0], UAV_speed_range[1])
                UAVs_mobile_pattern[UAV_id] = mobility_pattern
            else:
                delta_x = target_position[0] - current_position[0]
                delta_y = target_position[1] - current_position[1]
                delta_z = target_position[2] - current_position[2]

                # 计算 xy 平面的方位角
                angle = np.arctan2(delta_y, delta_x)

                # 计算 z 相对于 xy 平面的仰角
                distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
                phi = np.arctan2(delta_z, distance_xy)

                mobility_pattern = {}
                mobility_pattern['angle'] = angle
                mobility_pattern['phi'] = phi
                UAV_speed_range = self.trafficScheduler.getConfig(env, 'UAV_speed_range')
                mobility_pattern['speed'] = random.uniform(UAV_speed_range[0], UAV_speed_range[1])
                UAVs_mobile_pattern[UAV_id] = mobility_pattern
        self.trafficScheduler.setUAVMobilityPatterns(env, UAVs_mobile_pattern)

    def scheduleOffloading(self, env: AirFogSimEnv):
        """The offloading scheduling logic. Should be implemented by the subclass. Default is to offload the task to the nearest node.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
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
        """The communication scheduling logic. Should be implemented by the subclass. Default is random.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        n_RB = self.commScheduler.getNumberOfRB(env)
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        for task_dict in all_offloading_task_infos:
            allocated_RB_nos = np.random.choice(n_RB, 3, replace=False)
            self.commScheduler.setCommunicationWithRB(env, task_dict['task_id'], allocated_RB_nos)

    def scheduleComputing(self, env: AirFogSimEnv):
        """The computing scheduling logic. Should be implemented by the subclass. Default is evenly distributing the computing resources to the tasks.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        def alloc_cpu_callback(computing_tasks, **kwargs):
            # _computing_tasks: {task_id: task_dict}
            # simulation_interval: float
            # current_time: float
            # 返回值是一个字典，key是task_id，value是分配的cpu
            # 本函数的目的是将所有的cpu分配给task
            appointed_fog_node_dict = {}
            task_list = []
            for tasks in computing_tasks.values():
                for task in tasks:
                    task_dict = task.to_dict()
                    assigned_node_id = task_dict['assigned_to']
                    assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
                    task_num = appointed_fog_node_dict.get(assigned_node_id, 0)
                    if assigned_node_info is None or task_num>=3:
                        continue
                    appointed_fog_node_dict[assigned_node_id] = task_num + 1
                    task_list.append(task_dict)
            # 所有cpu分配给task
            alloc_cpu_dict = {}
            for task_dict in task_list:
                task_id = task_dict['task_id']
                assigned_node_id = task_dict['assigned_to']
                alloc_cpu = assigned_node_info.get('fog_profile', {}).get('cpu', 0) / max(1, appointed_fog_node_dict[assigned_node_id])
                alloc_cpu_dict[task_id] = alloc_cpu
            return alloc_cpu_dict
        self.compScheduler.setComputingCallBack(env, alloc_cpu_callback) 

    def getRewardByTask(self, env: AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done tasks in last time.
        
        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_task_infos = self.taskScheduler.getLastStepSuccTaskInfos(env)
        last_step_fail_task_infos = self.taskScheduler.getLastStepFailTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos+last_step_fail_task_infos:
            reward += self.rewardScheduler.getRewardByTask(env, task_info)
        return reward

    def getRewardByMission(self, env: AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done missions in last time.

        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_mission_infos = self.missionScheduler.getLastStepSuccMissionInfos(env)
        last_step_fail_mission_infos = self.missionScheduler.getLastStepFailMissionInfos(env)
        last_step_early_fail_mission_infos= self.missionScheduler.getLastStepEarlyFailMissionInfos(env)
        sum_reward = 0
        reward = 0
        punish = 0
        for mission_info in last_step_succ_mission_infos:
            mission_reward = self.rewardScheduler.getRewardByMission(env, mission_info)
            reward += mission_reward
            sum_reward += mission_reward
        for mission_info in last_step_fail_mission_infos:
            mission_punish = self.rewardScheduler.getPunishByMission(env, mission_info)
            punish += mission_punish
            sum_reward += mission_punish
        for mission_info in last_step_early_fail_mission_infos:
            mission_punish = self.rewardScheduler.getPunishByMission(env, mission_info)
            punish += mission_punish
            sum_reward += mission_punish
        return reward, punish, sum_reward






