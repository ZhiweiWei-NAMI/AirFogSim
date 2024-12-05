import numpy as np

from .base_sched import BaseScheduler
from ..enum_const import NodeTypeEnum

class AlgorithmScheduler(BaseScheduler):
    @staticmethod
    def getNodeStates(env,node_type):
        """Get node state (shape:[1,n]).

         Args:
             node_type(str): Type in ['V','R','U'].

         Returns:
             ndarray: Array of node state

         Examples:
             algoSched.getNodeState(env,'U')
         """
        assert node_type in ['V','R','U']
        v_max_num=100
        u_max_num = 10
        r_max_num = 4
        state = []
        if node_type=='V':
            vehicle_infos=env.traffic_manager.getVehicleTrafficInfos()
            for vehicle_id, vehicle_info in vehicle_infos.items():
                index = int(vehicle_id.split('_')[-1])  # 转换为整数
                node_type=NodeTypeEnum.VEHICLE
                position=vehicle_info['position']
                vehicle_state=[index,node_type,*position]# 注意position解包
                state.append(vehicle_state)
        elif node_type=='U':
            UAV_infos=env.traffic_manager.getUAVTrafficInfos()
            for UAV_id, UAV_info in UAV_infos.items():
                index = int(UAV_id.split('_')[-1])  # 转换为整数
                node_type=NodeTypeEnum.UAV
                position=UAV_info['position']
                UAV_state=[index,node_type,*position] # 注意position解包
                state.append(UAV_state)
        elif node_type == 'R':
            RSU_infos = env.traffic_manager.getRSUInfos()
            for RSU_id, RSU_info in RSU_infos.items():
                index = int(RSU_id.split('_')[-1])  # 转换为整数
                node_type = NodeTypeEnum.RSU
                position = RSU_info['position']
                RSU_state = [index, node_type, *position]# 注意position解包
                state.append(RSU_state)
        state_array=np.array(state).flatten()
        state_array.reshape(1,-1)
        return state_array

    @staticmethod
    def getMissionStates(env,mission_profile):
        index = int(mission_profile['mission_id'].split('_')[-1])  # 转换为整数
        accuracy = mission_profile['mission_accuracy']
        sensor_type = int(mission_profile['mission_sensor_type'].split('_')[-1])  # 转换为整数
        return_size = mission_profile['mission_size']
        arrive_time = mission_profile['mission_arrival_time']
        deadline = mission_profile['mission_deadline']
        duration = sum(mission_profile['mission_duration'])
        position = mission_profile['mission_routes'][0]
        distance_threshold = mission_profile['distance_threshold']
        state=[index,accuracy,sensor_type,return_size,arrive_time,deadline,duration,*position,distance_threshold]
        state_array=np.array(state).flatten()
        state_array.reshape(1,-1)
        return state_array

    @staticmethod
    def getNearest10SensorStates(env,sensor_type,lower_bound_accuracy,target_position,excluded_sensor_ids):
        sensor_max_num=10

        candidate_sensors = env.sensor_manager.getSensorsByStateAndType('idle', sensor_type)
        sensor_states=[]

        # Choose the sensor with the highest accuracy among the idle sensors
        for node_id, sensors in candidate_sensors.items():
            sensors = candidate_sensors.get(node_id, [])
            node_type=env._getNodeTypeById(node_id)
            if node_type=='V':
                node_position = env.traffic_manager.getVehiclePosition(node_id)
                node_type=NodeTypeEnum.VEHICLE
            elif node_type=='U':
                node_position=env.traffic_manager.getUAVPosition(node_id)
                node_type = NodeTypeEnum.UAV
            distance = np.linalg.norm(np.array(target_position) - np.array(node_position))
            for sensor in sensors:
                if sensor.getSensorAccuracy() > lower_bound_accuracy and sensor.getSensorId() not in excluded_sensor_ids:
                    index = int(sensor.getSensorId().split('_')[-1])  # 转换为整数
                    sensor_type=int(sensor.getSensorType().split('_')[-1])
                    node_index=int(node_id.split('_')[-1])
                    accuracy = sensor.getSensorAccuracy()
                    sensor_states.append([distance,index,sensor_type,accuracy,node_index,node_type])
        sensor_states_sorted = sorted(sensor_states, key=lambda x: x[0])

        top_10_sensor_states = []
        for item in sensor_states_sorted[:sensor_max_num]:
            top_10_sensor_states.extend([item[1:]])
        sensor_dim = len(sensor_states_sorted[0][1:])
        valid_sensor_num=len(top_10_sensor_states)
        mask = np.array([True] * valid_sensor_num + [False] * (sensor_max_num - valid_sensor_num))
        while len(top_10_sensor_states) < sensor_max_num:
            top_10_sensor_states.append([0] * sensor_dim)  # 补充零
        state_array=np.array(top_10_sensor_states).flatten()
        state_array.reshape(1,-1)
        return state_array,mask

    @staticmethod
    def getSensorInfoByAction(env,action_index,sensor_states):
        attr_num=5
        node_id_bias=3
        node_type_bias=4
        sensor_id_bias=0
        accuracy_bias=1

        states=sensor_states.copy().flatten()
        node_id_num=int(states[action_index*attr_num+node_id_bias])
        sensor_id_num=int(states[action_index*attr_num+sensor_id_bias])
        accuracy=states[action_index*attr_num+accuracy_bias]
        node_type=int(states[action_index*attr_num+node_type_bias])

        sensor_id=env.sensor_manager.completeSensorId(sensor_id_num)
        if node_type==NodeTypeEnum.VEHICLE:
            node_id=env.traffic_manager.completeStrId(node_id_num,'V')
        elif node_type==NodeTypeEnum.UAV:
            node_id = env.traffic_manager.completeStrId(node_id_num, 'U')

        return node_id,sensor_id,accuracy






