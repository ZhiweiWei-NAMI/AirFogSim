import collections
import random

from .airfogsim_scheduler import AirFogSimScheduler
from .airfogsim_env import AirFogSimEnv
from .algorithm.DDQN.DDQN_env import DDQN_Env
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
        """The computing scheduling logic. Should be implemented by the subclass. Default is FIFS.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        all_computing_task_infos = self.taskScheduler.getAllComputingTaskInfos(env)
        for task_dict in all_computing_task_infos:
            task_id = task_dict['task_id']
            task_node_id = task_dict['task_node_id']
            assigned_node_id = task_dict['assigned_to']
            assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
            self.compScheduler.setComputingWithNodeCPU(env, task_id, 0.3)  # allocate cpu 0.3

    def getRewardByTask(self, env: AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done tasks in last time.
        
        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_task_infos = self.taskScheduler.getLastStepSuccTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos:
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

    def initialize(self, env: AirFogSimEnv, config={}):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD',
                                      '5 * log(10, 1 + (_mission_deadline-_mission_duration_sum)) * (1 / (1 + exp(-(_mission_deadline-_mission_duration_sum) / (_mission_finish_time - _mission_arrival_time-_mission_duration_sum))) - 1 / (1 + exp(-1)))')
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
        traffic_interval=self.trafficScheduler.getTrafficInterval(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []

        generate_num=0
        allocate_num=0
        for mission_profile in new_missions_profile:
            if mission_profile['mission_arrival_time']>cur_time-traffic_interval:
                generate_num+=1
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
                allocate_num+=1

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)

        self.missionScheduler.setMissionEvaluationIndicators(env,generate_num,allocate_num)
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
                assert current_node_type is not None
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


class DDQNAlgorithmModule(BaseAlgorithmModule):
    """
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleMission: 
        Mission: Missions assigned to both vehicles and UAVs, decided by DDQN model.
        Sensor: Decided by DDQN model.
    scheduleReturning: Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct, select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: 
        UAV: Fly to next position in route list and stay for a period of time.
    '''

    class ReplayBuffer:
        def __init__(self):
            # 创建一个队列，先进先出，队列长度不限
            self.buffer = {}

        def __expToFlattenArray(self, exp):
            state = exp['state']
            action = exp['action']
            mask = exp['mask']
            reward = exp['reward']
            next_state = exp['next_state']
            next_mask = exp['next_mask']
            done = exp['done']
            return np.array(state), action, mask, reward, np.array(next_state), next_mask, done

        def add(self, exp_id, state, action, mask, reward=None, next_state=None, next_mask=None, done=None):
            self.buffer[exp_id] = {'state': state, 'action': action, 'mask': mask, 'reward': reward,
                                   'next_state': next_state, 'next_mask': next_mask, 'done': done}

        def setNextState(self, exp_id, next_state, next_mask, done):
            assert exp_id in self.buffer, "State_id is invalid."
            self.buffer[exp_id]['next_state'] = next_state
            self.buffer[exp_id]['next_mask'] = next_mask
            self.buffer[exp_id]['done'] = done

        def completeAndPopExperience(self, exp_id, reward):
            assert exp_id in self.buffer, "exp_id is invalid."
            self.buffer[exp_id]['reward'] = reward
            packed_exp = self.__expToFlattenArray(self.buffer[exp_id])
            del self.buffer[exp_id]
            return packed_exp

        def size(self):
            return len(self.buffer)

    def __init__(self,last_episode=None):
        super().__init__()
        self.dim_state = 631
        self.dim_action = 10
        self.DDQN_env = DDQN_Env(self.dim_state, self.dim_action)
        if last_episode is not None:
            self.DDQN_env.loadModel(last_episode)


    def initialize(self, env: AirFogSimEnv, config={}):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD',
                                      '5 * log(10, 1 + (_mission_deadline-_mission_duration_sum)) * (1 / (1 + exp(-(_mission_deadline-_mission_duration_sum) / (_mission_finish_time - _mission_arrival_time-_mission_duration_sum))) - 1 / (1 + exp(-1)))')
        self.rewardScheduler.setModel(env, 'PUNISH', '-1')
        self.max_n_vehicles = env.traffic_manager.getConfig('max_n_vehicles')
        self.max_n_UAVs = env.traffic_manager.getConfig('max_n_UAVs')
        self.max_n_RSUs = env.traffic_manager.getConfig('max_n_RSUs')
        self.last_mission_id = None  # Last allocated mission id,used in next state update
        self.replay_buffer = self.ReplayBuffer()

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

        cur_time = self.trafficScheduler.getCurrentTime(env)
        traffic_interval=self.trafficScheduler.getTrafficInterval(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []

        UAVs_state = self.algorithmScheduler.getNodeStates(env, 'U', self.max_n_UAVs)
        vehicles_state = self.algorithmScheduler.getNodeStates(env, 'V', self.max_n_vehicles)
        RSUs_state = self.algorithmScheduler.getNodeStates(env, 'R', self.max_n_RSUs)

        generate_num=0
        allocate_num=0
        for mission_profile in new_missions_profile:
            if mission_profile['mission_arrival_time']>cur_time-traffic_interval:
                generate_num+=1
            mission_id = mission_profile['mission_id']
            mission_sensor_type = mission_profile['mission_sensor_type']
            mission_accuracy = mission_profile['mission_accuracy']
            mission_position = mission_profile['mission_routes'][0]

            mission_state = self.algorithmScheduler.getMissionStates(env, mission_profile)
            valid_sensor_num, nearest_10_sensors_state, mask = self.algorithmScheduler.getNearest10SensorStates(env,mission_sensor_type,mission_accuracy,mission_position,excluded_sensor_ids)
            if valid_sensor_num==0:
                continue
            state = np.concatenate([UAVs_state, vehicles_state, RSUs_state, mission_state, nearest_10_sensors_state])
            state = state.flatten()
            is_random, max_q_value, action_index = self.DDQN_env.takeAction(state, mask)

            if self.last_mission_id is not None:
                self.replay_buffer.setNextState(self.last_mission_id, state, mask, False)
            self.replay_buffer.add(mission_id, state, action_index, mask)
            self.last_mission_id = mission_id

            appointed_node_type,appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.algorithmScheduler.getSensorInfoByAction(
                env, action_index, nearest_10_sensors_state)
            if appointed_node_id != None and appointed_sensor_id != None:
                if appointed_node_type=='U':
                    self.trafficScheduler.addUAVRoute(env,appointed_node_id,mission_position)
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
                allocate_num+=1

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)

        self.missionScheduler.setMissionEvaluationIndicators(generate_num, allocate_num)
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
            target_position = self.trafficScheduler.getNextPositionOfUav(env, UAV_id)
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

    def updateExperience(self, env: AirFogSimEnv):
        last_step_succ_mission_infos = self.missionScheduler.getLastStepSuccMissionInfos(env)
        last_step_fail_mission_infos = self.missionScheduler.getLastStepFailMissionInfos(env)
        for mission_info in last_step_succ_mission_infos:
            reward = self.rewardScheduler.getRewardByMission(env, mission_info)
            exp = self.replay_buffer.completeAndPopExperience(mission_info['mission_id'], reward)
            self.DDQN_env.addExperience(*exp)
        for mission_info in last_step_fail_mission_infos:
            reward = self.rewardScheduler.getPunishByMission(env, mission_info)
            exp = self.replay_buffer.completeAndPopExperience(mission_info['mission_id'], reward)
            self.DDQN_env.addExperience(*exp)

