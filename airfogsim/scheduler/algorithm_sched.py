import numpy as np

from .base_sched import BaseScheduler


class AlgorithmScheduler(BaseScheduler):
    @staticmethod
    def getNodeStates(env, node_type):
        """Get node states (shape:[node_num,dim]).

         Args:
             node_type(str): Type in ['V','R','U'].

         Returns:
             list: list of node states

         Examples:
             algoSched.getNodeState(env,'U')
         """
        # dim:[id, type, is_mission_node, is_schedulable, x, y, z]
        # [1, 'U', True, True, 105.23, 568.15. 225.65]
        assert node_type in ['V', 'I', 'U']
        state_dim = 5

        state = []
        if node_type == 'V':
            node_infos = env.traffic_manager.getVehicleTrafficInfos()
            # node_type_enum = NodeTypeEnum.VEHICLE
        elif node_type == 'I':
            node_infos = env.traffic_manager.getRSUInfos()
            # node_type_enum = NodeTypeEnum.RSU
        elif node_type == 'U':
            node_infos = env.traffic_manager.getUAVTrafficInfos()
            # node_type_enum = NodeTypeEnum.UAV

        for node_id, node_info in node_infos.items():
            index = int(node_id.split('_')[-1])  # 转换为整数
            x, y, z = node_info['position']
            node_state = [index, node_type, x, y, z]  # 注意position解包
            state.append(node_state)

        node_num = len(node_info)

        # state = state[:node_max_num]
        # valid_node_num = len(state)
        # if valid_node_num < node_max_num:
        #     state.extend([[0] * state_dim] * (node_max_num - valid_node_num))  # 补充零

        return node_num, state

    @staticmethod
    def getMissionStates(env, mission_profiles):
        """Get mission states (shape:[mission_num,dim]).

         Args:
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
            top_10_sensor_states.append([item[1:]])

        # mask = np.array([True] * valid_sensor_num + [False] * (sensor_max_num - valid_sensor_num)).flatten()
        # if valid_sensor_num < sensor_max_num:
        #     top_10_sensor_states.extend([[0] * sensor_dim] * (sensor_max_num - valid_sensor_num))  # 补充零
        # state_array = np.array(top_10_sensor_states).flatten()

        return top_10_sensor_states

    @staticmethod
    def getSensorStates(env, sensor_type, lower_bound_accuracy, excluded_sensor_ids):
        """Get sensor states (shape:[sensor_num,dim]).

         Args:
             sensor_type(str): The required type of sensor
             lower_bound_accuracy(float): The lowest permitted accuracy of sensor
             excluded_sensor_ids(list): The sensor ids of occupied sensors in this step

         Returns:
             list: list of sensor states

         Examples:
             algoSched.getSensorStates(env,2,0.5,[1,2,3,4])
         """
        # dim:[node_id,node_type,id,type, accuracy,candidate]
        # [1, 'U', 3, 0.8, True]

        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', sensor_type)
        idle_sensors = env.sensor_manager.getSensorsByStateAndType('idle')
        busy_sensors = env.sensor_manager.getSensorsByStateAndType('busy')
        combined_sensors = idle_sensors + busy_sensors
        sensor_states = []

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in combined_sensors.items():

            node_type = env._getNodeTypeById(node_id)
            assert node_type is not None, "Node is invalid."
            for sensor in sensors:
                if sensor.getSensorAccuracy() > lower_bound_accuracy and \
                        sensor.getSensorId() not in excluded_sensor_ids and \
                        node_id in candidate_sensors:
                    candidate = True
                else:
                    candidate = False
                index = int(sensor.getSensorId().split('_')[-1])  # 转换为整数
                sensor_type = int(sensor.getSensorType().split('_')[-1])
                node_index = int(node_id.split('_')[-1])
                accuracy = sensor.getSensorAccuracy()
                state = [node_index, node_type, index, sensor_type, accuracy, candidate]
                sensor_states.append(state)

        return sensor_states

    @staticmethod
    def getNeighborUAVStates(env, base_position, distance_threshold, max_num):
        """Get neighbor UAV states in (shape:[max_num,dim]).

         Args:
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
            UAV_position = UAV_info['position']
            x, y, z = UAV_position
            distance = np.linalg.norm(np.array(base_position) - np.array(UAV_position))

            if distance <= distance_threshold:
                state = [distance, x, y, z]
                UAV_states.append(state)
        states_sorted = sorted(UAV_states, key=lambda x: x[0])

        max_num_sensor_states = []
        for item in states_sorted[:max_num]:
            max_num_sensor_states.append([item[1:]])

        return max_num_sensor_states

    @staticmethod
    def getTransMissionStates(env, base_position, distance_threshold, max_num):
        """Get neighbor mission states in (shape:[max_num,dim]).

         Args:
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
            max_num_mission_states.append([item[1:]])

        return max_num_mission_states

    @staticmethod
    def getSelfUAVStates(env, UAV_id):
        """Get UAV state in (shape:[dim]).

         Args:
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
        attr_num = 6
        node_index_bias = 0
        node_type_bias = 1
        sensor_index_bias = 2
        sensor_type_bias = 1
        accuracy_bias = 1
        candidate_bias = 1

        node_id_num = int(sensor_states[action_index * attr_num + node_index_bias])
        sensor_id_num = int(sensor_states[action_index * attr_num + sensor_index_bias])
        accuracy = sensor_states[action_index * attr_num + accuracy_bias]
        node_type = int(sensor_states[action_index * attr_num + node_type_bias])

        sensor_id = env.sensor_manager.completeSensorId(sensor_id_num)
        node_id = env.traffic_manager.completeStrId(node_id_num, node_type)

        return node_type, node_id, sensor_id, accuracy
