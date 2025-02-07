import itertools

import numpy as np

from .base_sched import BaseScheduler


class AlgorithmScheduler(BaseScheduler):
    @staticmethod
    def getNodeStates(env, node_type=None, node_priority=None):
        """Get node states (shape:[node_num,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             node_type(str): Type in ['V','I','U'].

         Returns:
             int: node num,
             list: list of node states
             node_priority:  {{node_type: priority}, ...}

         Examples:
             algoSched.getNodeState(env,'U')
         """
        # dim:[id, type, is_mission_node, is_schedulable, x, y, z]
        # [1, 'U', True, True, 105.23, 568.15. 225.65]
        if node_type is not None:
            assert node_type in ['V', 'I', 'U']
        state_dim = 5
        feature_dict = {
            'V': {
                'is_mission_node': True,
                'is_schedulable': False
            },
            'U': {
                'is_mission_node': True,
                'is_schedulable': True
            },
            'I': {
                'is_mission_node': False,
                'is_schedulable': False
            }
        }

        states = []
        candidate_nodes = {}
        if node_type == 'V':
            node_infos = env.traffic_manager.getVehicleTrafficInfos()
            candidate_nodes[node_type] = node_infos
            # node_type_enum = NodeTypeEnum.VEHICLE
        elif node_type == 'I':
            node_infos = env.traffic_manager.getRSUInfos()
            candidate_nodes[node_type] = node_infos
            # node_type_enum = NodeTypeEnum.RSU
        elif node_type == 'U':
            node_infos = env.traffic_manager.getUAVTrafficInfos()
            candidate_nodes[node_type] = node_infos
            # node_type_enum = NodeTypeEnum.UAV
        else:
            vehicle_infos = env.traffic_manager.getVehicleTrafficInfos()
            RSU_infos = env.traffic_manager.getRSUInfos()
            UAV_infos = env.traffic_manager.getUAVTrafficInfos()
            candidate_nodes['V'] = vehicle_infos
            candidate_nodes['I'] = RSU_infos
            candidate_nodes['U'] = UAV_infos

        node_num = 0
        for node_type, node_infos in candidate_nodes.items():
            node_num += len(node_infos)
            for node_id, node_info in node_infos.items():
                index = int(node_id.split('_')[-1])  # 转换为整数
                x, y, z = node_info['position']
                is_mission_node = feature_dict[node_type]['is_mission_node']
                is_schedulable = feature_dict[node_type]['is_schedulable']
                node_state = [index, node_type, is_mission_node, is_schedulable, x, y, z]
                states.append(node_state)

        # 按type,id排序
        sorted_node_states = sorted(states, key=lambda x: (node_priority[x[1]], x[0]))

        return node_num, sorted_node_states

    @staticmethod
    def getMissionStates(env, mission_profiles):
        """Get mission states (shape:[mission_num,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             mission_profiles(dict):

         Returns:
             list: list of mission states

         Examples:
             algoSched.getMissionStates(env,mission_profiles)
         """
        # dim:[sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]
        # ['U',0.8,50,20,120,5,120.25,262.05,553.25,100]
        states = []
        for mission_profile in mission_profiles:
            # index = int(mission_profile['mission_id'].split('_')[-1])  # 转换为整数
            sensor_type = int(mission_profile['mission_sensor_type'].split('_')[-1])  # 转换为整数
            accuracy = mission_profile['mission_accuracy']
            return_size = mission_profile['mission_size']
            arrive_time = mission_profile['mission_arrival_time']
            TTL = mission_profile['mission_deadline']
            duration = sum(mission_profile['mission_duration'])
            x, y, z = mission_profile['mission_routes'][0]
            distance_threshold = mission_profile['distance_threshold']
            state = [sensor_type, accuracy, return_size, arrive_time, TTL, duration, x, y, z, distance_threshold]
            states.append(state)
        return states

    @staticmethod
    def getNearest10SensorStates(env, sensor_type, lower_bound_accuracy, base_position, excluded_sensor_ids):
        """Get sensor states (shape:[10,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             sensor_type(str): The required type of sensor
             lower_bound_accuracy(float): The lowest permitted accuracy of sensor
             base_position(list): The position of core entity
             excluded_sensor_ids(list): The sensor ids of occupied sensors in this step

         Returns:
             list: list of sensor states

         Examples:
             algoSched.getNearest10SensorStates(env,2,0.5,[100,100,200],[1,2,3,4])
         """
        # dim:[node_id,node_type,id,type, accuracy,candidate]
        # [1, 'U', 3, 0.8]
        sensor_max_num = 10

        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', sensor_type)
        sensor_states = []

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in candidate_sensors.items():
            node_type = env._getNodeTypeById(node_id)
            assert node_type is not None, "Node is invalid."
            node_position = env.traffic_manager.getNodePositionById(node_id)
            # if node_type == 'V':
            #     node_position = env.traffic_manager.getVehiclePosition(node_id)
            #     node_type = NodeTypeEnum.VEHICLE
            # elif node_type == 'U':
            #     node_position = env.traffic_manager.getUAVPosition(node_id)
            #     node_type = NodeTypeEnum.UAV
            distance = np.linalg.norm(np.array(base_position) - np.array(node_position))
            for sensor in sensors:
                if sensor.getSensorAccuracy() > lower_bound_accuracy and sensor.getSensorId() not in excluded_sensor_ids:
                    index = int(sensor.getSensorId().split('_')[-1])  # 转换为整数
                    sensor_type = int(sensor.getSensorType().split('_')[-1])
                    node_index = int(node_id.split('_')[-1])
                    accuracy = sensor.getSensorAccuracy()
                    sensor_states.append([distance, node_index, node_type, index, sensor_type, accuracy, True])
        sensor_states_sorted = sorted(sensor_states, key=lambda x: x[0])

        top_10_sensor_states = []
        for item in sensor_states_sorted[:sensor_max_num]:
            used_state = item[1:]
            top_10_sensor_states.append(used_state)

        # mask = np.array([True] * valid_sensor_num + [False] * (sensor_max_num - valid_sensor_num)).flatten()
        # if valid_sensor_num < sensor_max_num:
        #     top_10_sensor_states.extend([[0] * sensor_dim] * (sensor_max_num - valid_sensor_num))  # 补充零
        # state_array = np.array(top_10_sensor_states).flatten()

        return top_10_sensor_states

    @staticmethod
    def getSensorStates(env, sensor_type, lower_bound_accuracy, excluded_sensor_ids,current_position,TA_distance_Veh,TA_distance_UAV, node_priority):
        """Get sensor states (shape:[sensor_num,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             sensor_type(str): The required type of sensor
             lower_bound_accuracy(float): The lowest permitted accuracy of sensor
             excluded_sensor_ids(list): The sensor ids of occupied sensors in this step
             node_priority(dict): {{node_type: priority}, ...}

         Returns:
             list: list of sensor states

         Examples:
             algoSched.getSensorStates(env,2,0.5,[1,2,3,4])
         """
        # dim:[node_id,node_type,id,type, accuracy,candidate]
        # [1, 'U', 3, 2, 0.8, True]

        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', sensor_type)
        idle_sensors = env.sensor_manager.getSensorsByStateAndType('idle')
        busy_sensors = env.sensor_manager.getSensorsByStateAndType('busy')
        combined_sensors = itertools.chain(idle_sensors.items(), busy_sensors.items())
        sensor_states_dict = {}
        valid_sensor_num = 0

        candidate_sensor_ids=[sensor.getSensorId() for node_id, sensors in candidate_sensors.items() for sensor in sensors]

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in combined_sensors:
            node_sensor_states = sensor_states_dict.get(node_id, [])
            node_type = env._getNodeTypeById(node_id)
            assert node_type is not None, "Node is invalid."
            node=env._getNodeById(node_id)
            node_position = node.getPosition()
            distance = np.linalg.norm(np.asarray(node_position) - np.asarray(current_position))

            for sensor in sensors:
                sensor_id=sensor.getSensorId()
                if node_type=='V' and \
                        sensor.getSensorAccuracy() > lower_bound_accuracy and \
                        sensor_id not in excluded_sensor_ids and \
                        sensor_id in candidate_sensor_ids and \
                        distance < TA_distance_Veh: # Veh可分配距离阈值
                    candidate = True
                    valid_sensor_num += 1
                elif node_type=='U' and \
                        sensor.getSensorAccuracy() > lower_bound_accuracy and \
                        sensor_id not in excluded_sensor_ids and \
                        sensor_id in candidate_sensor_ids and \
                        distance < TA_distance_UAV: # UAV可分配距离阈值
                    candidate = True
                    valid_sensor_num += 1
                else:
                    candidate = False
                index = int(sensor.getSensorId().split('_')[-1])  # 转换为整数
                sensor_type = int(sensor.getSensorType().split('_')[-1])
                node_index = int(node_id.split('_')[-1])
                accuracy = sensor.getSensorAccuracy()
                state = [node_index, node_type, index, sensor_type, accuracy, candidate]
                node_sensor_states.append(state)
            sensor_states_dict[node_id] = node_sensor_states

        # 将dict转为list
        sensor_states = list(sensor_states_dict.values())
        # 按 node_type,node_id 排序
        sorted_sensor_states = sorted(sensor_states, key=lambda x: (node_priority[x[0][1]], x[0][0]))

        return valid_sensor_num, sorted_sensor_states

    @staticmethod
    def getNeighborUAVStates(env, base_position, distance_threshold, max_num):
        """Get neighbor UAV states in (shape:[max_num,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             base_position(str): The position of core UAV
             distance_threshold(float): The distance threshold of UAV search range
             max_num(list): The max num of searched UAVs

         Returns:
             list: list of neighbor UAV states

         Examples:
             algoSched.getNeighborUAVStates(env,[100,100,200],500,50)
         """
        # dim:[x, y, z]
        # [105.23, 568.15. 225.65]

        UAV_states = []
        UAV_infos = env.traffic_manager.getUAVTrafficInfos()
        for UAV_id, UAV_info in UAV_infos.items():
            node_index = int(UAV_id.split('_')[-1])
            UAV_position = UAV_info['position']
            x, y, z = UAV_position
            distance = np.linalg.norm(np.array(base_position) - np.array(UAV_position))

            if distance <= distance_threshold:
                state = [distance,node_index, x, y, z]
                UAV_states.append(state)
        states_sorted = sorted(UAV_states, key=lambda x: x[0])

        max_num_sensor_states = []
        for item in states_sorted[:max_num]:
            used_state = item[1:]
            max_num_sensor_states.append(used_state)

        return max_num_sensor_states

    @staticmethod
    def getTransMissionStates(env, base_position, distance_threshold, max_num):
        """Get neighbor mission states in (shape:[max_num,dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             base_position(str): The position of core UAV
             distance_threshold(float): The distance threshold of UAV search range
             max_num(list): The max num of searched UAVs

         Returns:
             list: list of neighbor mission states

         Examples:
             algoSched.getTransMissionStates(env,[100,100,200],500,50)
         """
        # [left_sensing_time, left_return_size, x, y, z]
        # [5,30,120.25,262.05,553.25]

        mission_states = []
        executing_missions = env.mission_manager.getExecutingMissions()
        for node_id, missions in executing_missions.items():
            for mission in missions:
                current_node_id = mission.getCurrentNodeId()
                node = env._getNodeById(current_node_id)
                node_position = env.traffic_manager.getNodePositionById(node_id)
                x, y, z = node_position
                left_sensing_time = mission.getLeftSensingTime()
                left_return_size = mission.getLeftReturnSize()
                distance = np.linalg.norm(np.array(base_position) - np.array(node_position))
                if distance < distance_threshold:
                    state = [distance, left_sensing_time, left_return_size, x, y, z]
                    mission_states.append(state)
        states_sorted = sorted(mission_states, key=lambda x: x[0])

        max_num_mission_states = []
        for item in states_sorted[:max_num]:
            used_state = item[1:]
            max_num_mission_states.append(used_state)

        return max_num_mission_states

    @staticmethod
    def getSelfUAVStates(env, UAV_id):
        """Get UAV state in (shape:[dim]).

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             UAV_id(str): The id of UAV

         Returns:
             list: list of UAV state

         Examples:
             algoSched.getTransMissionStates(env,1)
         """
        # dim:[x, y, z, energy]
        # [105.23, 568.15, 225.65, 12000]
        node_position = env.traffic_manager.getNodePositionById(UAV_id)
        x, y, z = node_position
        left_energy = env.energy_manager.getEnergyById(UAV_id)
        state = [x, y, z, left_energy]
        return state

    @staticmethod
    def getSensorInfoByAction(env, action_index, sensor_states):
        """Get sensor info from sensor states by action index.

         Args:
             env (AirFogSimEnv): The AirFogSim environment.
             action_index(int): The index of sensor select action, corresponding to sensor
             sensor_states(list): Sensor states, [node_num, node_sensor_num, sensor_dim]
             node_dict(dict): {type_str:type_int,...}

         Returns:
             list: list of UAV state

         Examples:
             algoSched.getTransMissionStates(env, 1, [[node_id, node_type, id, type, accuracy, candidate], ..., ...], {'U':0,'V':1,'I':2,'C':3})
         """
        # [node_id, node_type, id, type, accuracy, candidate]
        attr_num = 6
        node_index_bias = 0
        node_type_bias = 1
        sensor_index_bias = 2
        sensor_type_bias = 3
        accuracy_bias = 4
        candidate_bias = 5

        flattened_states = [sensor_state for node in sensor_states for sensor_state in node]

        # 打印形状
        node_id_num = int(flattened_states[action_index][node_index_bias])
        sensor_id_num = int(flattened_states[action_index][sensor_index_bias])
        accuracy = flattened_states[action_index][accuracy_bias]
        node_type = flattened_states[action_index][node_type_bias]
        # node_type = [type_str for type_str, type_int in node_dict.items() if type_int == node_type]

        sensor_id = env.sensor_manager.completeSensorId(sensor_id_num)
        node_id = env.traffic_manager.completeStrId(node_id_num, node_type)

        return node_type, node_id, sensor_id, accuracy

    @staticmethod
    def getUAVStepRecord(env):
        UAV_energy_consumptions = env.getUAVStepEnergyConsumption()
        UAV_trans_datas = env.getUAVStepTransmissionSize()
        UAV_sensing_datas = env.getUAVStepSensingData()

        return UAV_energy_consumptions, UAV_trans_datas, UAV_sensing_datas
