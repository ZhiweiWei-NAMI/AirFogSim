import numpy as np

from .base_sched import BaseScheduler

class TrafficScheduler(BaseScheduler):
    @staticmethod
    def getConfig(env,name):
        return env.traffic_manager.getConfig(name)

    @staticmethod
    def getCurrentTime(env):
        return env.traffic_manager.getCurrentTime()

    @staticmethod
    def getDistanceBetweenNodesById(env,node_id_1,node_id_2):
        return env.getDistanceBetweenNodesById(node_id_1,node_id_2)
    @staticmethod
    def getUAVTrafficInfos(env):
        return env.traffic_manager.getUAVTrafficInfos()

    @staticmethod
    def getRSUTrafficInfos(env):
        return env.traffic_manager.getRSUInfos()

    @staticmethod
    def setUAVMobilityPatterns(env,UAV_mobility_patterns):
        organized_patterns={}
        for UAV_id,UAV_mobile_pattern in UAV_mobility_patterns.items():
            organized_patterns[UAV_id]={}
            organized_patterns[UAV_id]['angle']=UAV_mobile_pattern['angle']
            organized_patterns[UAV_id]['phi']=UAV_mobile_pattern['phi']
            organized_patterns[UAV_id]['speed']=UAV_mobile_pattern['speed']
        env.uav_mobility_patterns =organized_patterns

    @staticmethod
    def getVehicleInfosInRange(env, target_position, distance_threshold):
        vehicle_infos=env.traffic_manager.getVehicleTrafficInfos()
        candidate_vehicle_infos={}
        for vehicle_id,vehicle_info in vehicle_infos.items():
            vehicle_position=vehicle_info['position']
            distance=np.linalg.norm(np.array(target_position) - np.array(vehicle_position))
            if distance<=distance_threshold:
                candidate_vehicle_infos[vehicle_id]=vehicle_info
        return candidate_vehicle_infos

    # @staticmethod
    # def setUAVSpeedAndDirectionByNodeId(env, node_id: str, speed: float, angle: float, phi: float):
    #     """Set the UAV speed and direction by the node id.
    #
    #     Args:
    #         env (AirFogSimEnv): The environment.
    #         node_id (str): The node id.
    #         speed (float): The speed.
    #         angle (float): The angle (angle in 2D plane).
    #         phi (float): The phi (angle in 3D plane).
    #     """
    #     env.traffic_manager.updateUAVMobilityPatternById(node_id, {'speed': speed, 'angle': angle, 'phi': phi})


