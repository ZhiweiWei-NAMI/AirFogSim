import random

from .airfogsim_scheduler import AirFogSimScheduler
from .airfogsim_env import AirFogSimEnv
# from .algorithm.DDQN.DDQN_env import  DDQN_Env
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

    def initialize(self, env: AirFogSimEnv, config={}):
        """Initialize the algorithm with the environment. Should be implemented by the subclass. Including setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD', '-task_delay')
        self.rewardScheduler.setModel(env, 'PUNISH', '-1')

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

        # 1. generate mission (according to Poisson)
        # missionScheduler = AirFogSimScheduler.getMissionScheduler()
        # mission_profiles = [{
        #     'id':'Mission-1',
        #     'position': (100,230),
        #     'duration': 100,
        #     'task_profiles': [{
        #         'id':'Task-1',
        #         'task_type':'classification',
        #         'task_node_id':'Node-1',
        #         'task_deadline': 100,
        #         'task_data_size': 100,
        #         'task_computation': 100,
        #         'task_offloading': 100
        #     }]
        #     } for _ in range(3)]
        # for mission_profile in mission_profiles:
        #     missionScheduler.generateMission(env, mission_profile)

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
            target_position = self.missionScheduler.getNearestMissionPosition(env, UAV_id, UAV_info['position'])
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
            # 获取CPU最大的节点
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
        all_offloading_task_infos = all_offloading_task_infos[:n_RB]
        avg_RB_nos = max(1, n_RB // max(1, len(all_offloading_task_infos)))
        RB_ctr = 0
        for task_dict in all_offloading_task_infos:
            # 从RB_ctr到RB_ctr+avg_RB_nos-1分配给task；多出的部分mod n_RB，allocated_RB_nos是RB编号的列表
            allocated_RB_nos = [(RB_ctr + i) % n_RB for i in range(avg_RB_nos)]
            RB_ctr = (RB_ctr + avg_RB_nos) % n_RB
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
        return reward, punish, sum_reward


class NVHAUAlgorithmModule(BaseAlgorithmModule):
    """Nearest Vehicle and Highest Accuracy Uav Algorithm Module
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleMission: 
        Mission: Missions assigned to both vehicles and UAVs, each type has a probability of sum of 1.
        Sensor: Assigned to vehicle, select the sensor closest to PoI from the idle sensors with accuracy higher than required(Distance First).
                Assigned to RSU, select the sensor with lowest accuracy from the idle sensors with accuracy higher than required(Accuracy Lowerbound).
    scheduleReturning: Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct, select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: BaseAlgorithm.
    '''

    def __init__(self):
        super().__init__()

    def initialize(self, env: AirFogSimEnv):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        super().initialize(env)

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
        UAV_probability = 0.5
        cur_time = self.trafficScheduler.getCurrentTime(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []
        for mission_profile in new_missions_profile:
            mission_sensor_type = mission_profile['mission_sensor_type']
            mission_accuracy = mission_profile['mission_accuracy']

            if random.random() < UAV_probability:
                appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.sensorScheduler.getLowestAccurateIdleSensorOnUAV(
                    env, mission_sensor_type, mission_accuracy, excluded_sensor_ids)
            else:
                sensing_position = mission_profile['mission_routes'][0]
                distance_threshold = self.missionScheduler.getConfig(env, 'distance_threshold')
                vehicle_infos = self.trafficScheduler.getVehicleInfosInRange(env, sensing_position, distance_threshold)
                appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.sensorScheduler.getNearestIdleSensorInNodes(
                    env, mission_sensor_type, mission_accuracy, sensing_position, vehicle_infos, excluded_sensor_ids)

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
                        'return_size': mission_profile['mission_size'],
                    }
                    new_task = self.taskScheduler.generateTaskOfMission(env, mission_task_profile)
                    task_set.append(new_task)
                    mission_profile['mission_task_sets'].append(task_set)
                self.missionScheduler.generateAndAddMission(env, mission_profile)

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)
        self.missionScheduler.deleteBeAssignedMissionsProfile(env, delete_mission_profile_ids)

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
                    # relay_comm_rate = np.zeros((1, UAV_num, RSU_num))
                    # direct_comm_rate = np.zeros((1, RSU_num))
                    # v_idx=self.entityScheduler.getNodeIdxById(env,current_node_id)
                    # for u_idx in range(UAV_num):
                    #     for r_idx in range(RSU_num):
                    #         V2U_rate = self.commScheduler.getSumRateByChannelType(env, v_idx, u_idx, 'V2U')
                    #         U2I_rate = self.commScheduler.getSumRateByChannelType(env, u_idx, r_idx, 'U2I')
                    #         avg_rate = V2U_rate * U2I_rate / (V2U_rate + U2I_rate)
                    #         relay_comm_rate[0][u_idx][r_idx] = avg_rate
                    # for r_idx in range(RSU_num):
                    #     V2I_rate = self.commScheduler.getSumRateByChannelType(env, v_idx, r_idx, 'V2I')
                    #     direct_comm_rate[0][r_idx] = V2I_rate
                    #
                    # relay_max_value = np.max(relay_comm_rate)
                    # relay_max_index = np.unravel_index(np.argmax(relay_comm_rate), relay_comm_rate.shape)
                    # relay_u_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(relay_max_index[1]), 'U')['id']
                    # relay_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(relay_max_index[2]), 'R')['id']
                    # relay_max_route = [relay_u_id, relay_r_id]
                    #
                    # direct_max_value = np.max(direct_comm_rate)
                    # direct_max_index = np.unravel_index(np.argmax(direct_comm_rate), direct_comm_rate.shape)
                    # direct_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(direct_max_index[1]), 'R')['id']
                    # direct_max_route = [direct_r_id]
                    # print('relay',relay_max_value,relay_max_route)
                    # print('direct',direct_max_value,direct_max_route)

                    # return_route = relay_max_route if relay_max_value > direct_max_value else direct_max_route

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
        super().scheduleTraffic(env)

    def scheduleOffloading(self, env: AirFogSimEnv):
        # super().scheduleOffloading(env)
        pass

    def scheduleCommunication(self, env: AirFogSimEnv):
        super().scheduleCommunication(env)

    def scheduleComputing(self, env: AirFogSimEnv):
        super().scheduleComputing(env)

    def getRewardByTask(self, env: AirFogSimEnv):
        return super().getRewardByTask(env)

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)

