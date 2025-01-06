import random

from airfogsim.airfogsim_scheduler import AirFogSimScheduler
from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
import numpy as np

class GreedyAlgorithmModule(BaseAlgorithmModule):
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

    def reset(self,env: AirFogSimEnv):
        super().reset(env)

    def scheduleStep(self, env: AirFogSimEnv):
        """The algorithm logic.

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
                RSU_num = self.entityScheduler.getNodeNumByType(env, 'I')

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
                    # relay_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(relay_max_index[2]), 'I')['id']
                    # relay_max_route = [relay_u_id, relay_r_id]
                    #
                    # direct_max_value = np.max(direct_comm_rate)
                    # direct_max_index = np.unravel_index(np.argmax(direct_comm_rate), direct_comm_rate.shape)
                    # direct_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(direct_max_index[1]), 'I')['id']
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
                        V2R_distance = np.zeros(RSU_num)
                        for r_idx in range(RSU_num):
                            r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'I')['id']
                            distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                            V2R_distance[r_idx] = distance
                        nearest_r_distance = np.max(V2R_distance)
                        nearest_r_idx = np.unravel_index(np.argmax(V2R_distance), V2R_distance.shape)
                        nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'I')[
                            'id']

                    relay_probability = 0.5
                    if random.random() < relay_probability and UAV_num > 0:
                        return_route = [nearest_u_id, nearest_r_id]
                    else:
                        return_route = [nearest_r_id]

                elif current_node_type == 'U':
                    U2R_distance = np.zeros((RSU_num))
                    for r_idx in range(RSU_num):
                        r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'I')['id']
                        distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                        U2R_distance[r_idx] = distance
                    nearest_r_distance = np.max(U2R_distance)
                    nearest_r_idx = np.unravel_index(np.argmax(U2R_distance), U2R_distance.shape)
                    nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'I')['id']
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