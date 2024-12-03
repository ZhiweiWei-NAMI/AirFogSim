import numpy as np

from .base_sched import BaseScheduler
from ..enum_const import NodeTypeEnum

class AlgorithmScheduler(BaseScheduler):
    @staticmethod
    def getNodesState(env,node_type):
        """Get node state (shape:[1,n]).

         Args:
             node_type(str): Type in ['V','R','U'].

         Returns:
             ndarray: Array of node state

         Examples:
             algoSched.getNodeState(env,'U')
         """
        assert node_type in ['V','R','U']
        state = []
        if node_type=='V':
            vehicle_infos=env.traffic_manager.getVehicleTrafficInfos()
            for vehicle_id, vehicle_info in vehicle_infos.items():
                index = int(vehicle_id.split('_')[1])  # 转换为整数
                node_type=NodeTypeEnum.VEHICLE
                position=vehicle_info['position']
                vehicle_state=[index,node_type,position]
                state.append(vehicle_state)
        elif node_type=='U':
            UAV_infos=env.traffic_manager.getUAVTrafficInfos()
            for UAV_id, UAV_info in UAV_infos.items():
                index = int(UAV_id.split('_')[1])  # 转换为整数
                node_type=NodeTypeEnum.UAV
                position=UAV_info['position']
                UAV_state=[index,node_type,position]
                state.append(UAV_state)
        elif node_type == 'R':
            RSU_infos = env.traffic_manager.getRSUInfos()
            for RSU_id, RSU_info in RSU_infos.items():
                index = int(RSU_id.split('_')[1])  # 转换为整数
                node_type = NodeTypeEnum.RSU
                position = RSU_info['position']
                RSU_state = [index, node_type, position]
                state.append(RSU_state)
        state_array=np.array(state).flatten()
        state_array.reshape(1,-1)
        return state_array

    @staticmethod
    def getMissionsState(env,mission_profile):
        index=mission_profile['mission_id']
        accuracy = mission_profile['mission_accuracy']
        sensor_type = mission_profile['mission_sensor_type']
        return_size = mission_profile['mission_size']
        arrive_time = mission_profile['mission_arrival_time']
        deadline = mission_profile['mission_deadline']
        duration = sum(mission_profile['mission_duration'])
        position = mission_profile['mission_routes']
        distance_threshold = mission_profile['distance_threshold']
        state=[index,accuracy,sensor_type,return_size,arrive_time,deadline,duration,position,distance_threshold]
        state_array=np.array(state).flatten()
        state_array.reshape(1,-1)
        return state_array

    @staticmethod
    def getSensorsState(env,sensor_type,lower_bound_accuracy):
        pass


