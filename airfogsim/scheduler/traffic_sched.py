import copy

import numpy as np

from .base_sched import BaseScheduler


class TrafficScheduler(BaseScheduler):
    @staticmethod
    def getConfig(env, name):
        return env.traffic_manager.getConfig(name)

    @staticmethod
    def getCurrentTime(env):
        return env.traffic_manager.getCurrentTime()

    @staticmethod
    def getTrafficInterval(env):
        return env.traffic_interval

    @staticmethod
    def getDistanceBetweenNodesById(env, node_id_1, node_id_2):
        return env.getDistanceBetweenNodesById(node_id_1, node_id_2)

    @staticmethod
    def getUAVTrafficInfos(env):
        return env.traffic_manager.getUAVTrafficInfos()

    @staticmethod
    def getRSUTrafficInfos(env):
        return env.traffic_manager.getRSUInfos()




    @staticmethod
    def setUAVMobilityPatterns(env, UAV_mobility_patterns):
        organized_patterns = {}
        for UAV_id, UAV_mobile_pattern in UAV_mobility_patterns.items():
            organized_patterns[UAV_id] = {}
            organized_patterns[UAV_id]['angle'] = UAV_mobile_pattern['angle']
            organized_patterns[UAV_id]['phi'] = UAV_mobile_pattern['phi']
            organized_patterns[UAV_id]['speed'] = UAV_mobile_pattern['speed']
        env.uav_mobility_patterns = organized_patterns

    @staticmethod
    def getVehicleInfosInRange(env, target_position, distance_threshold):
        vehicle_infos = env.traffic_manager.getVehicleTrafficInfos()
        candidate_vehicle_infos = {}
        vehicle_ids_list = list(vehicle_infos.keys())
        vehicle_positions = [vehicle_infos[vehicle_id]['position'] for vehicle_id in vehicle_ids_list]
        vehicle_positions = np.asarray(vehicle_positions)
        distances = np.linalg.norm(vehicle_positions - np.asarray(target_position), axis=1)
        selected_vehicle_ids = np.where(distances <= distance_threshold)[0]
        for idx in selected_vehicle_ids:
            vehicle_id = vehicle_ids_list[idx]
            candidate_vehicle_infos[vehicle_id] = vehicle_infos[vehicle_id]
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

    @staticmethod
    def getNextPositionOfUav(env, UAV_id):
        route = env.uav_routes.get(UAV_id, [])
        if len(route) == 0:
            return None
        else:
            return copy.deepcopy(route[0]['position'])

    @staticmethod
    def addUAVRoute(env, UAV_id, pos_with_time):
        route = env.uav_routes.get(UAV_id, [])
        route.append(pos_with_time)
        env.uav_routes[UAV_id] = route

    @staticmethod
    def updateRoute(env, UAV_id, stay_time):
        route = env.uav_routes.get(UAV_id, [])
        assert len(route) > 0, f"Route length of {UAV_id} should larger than 0."
        route[0]['to_stay_time'] = max(route[0]['to_stay_time'] - stay_time, 0)
        if route[0]['to_stay_time'] <= 0:
            del route[0]
        env.uav_routes[UAV_id] = route
