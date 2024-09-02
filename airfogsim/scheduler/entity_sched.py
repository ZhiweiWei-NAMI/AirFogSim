
from .base_sched import BaseScheduler
class EntityScheduler(BaseScheduler):
    """The entity scheduler for entities.
    """
    
    @staticmethod
    def getAllNodeIdsWithType(env):
        """Get all node ids with type.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            list: The node ids.
            list: The type list.
        """
        vehicle_ids = env.vehicles.keys()
        uav_ids = env.UAVs.keys()
        rsu_ids = env.RSUs.keys()
        cloud_server_ids = env.cloudServers.keys()
        all_ids = vehicle_ids + uav_ids + rsu_ids + cloud_server_ids
        type_list = ['vehicle'] * len(vehicle_ids) + ['uav'] * len(uav_ids) + ['rsu'] * len(rsu_ids) + ['cloud_server'] * len(cloud_server_ids)
        return all_ids, type_list
    
    @staticmethod
    def getNodeInfoByIndexAndType(env, idx: int, type: str):
        """Get the node by the index and type.

        Args:
            env (AirFogSimEnv): The environment.
            idx (int): The index.
            type (str): The type, ['vehicle' or 'v', 'uav' or 'u', 'rsu' or 'r', 'cloud_server' or 'c']

        Returns:
            dict: The node info.
        """
        type = type.lower()
        if type in ['vehicle', 'v']:
            node = env.vehicles[env.vehicle_ids_as_index[idx]]
        elif type in ['uav', 'u']:
            node = env.UAVs[env.uav_ids_as_index[idx]]
        elif type in ['rsu', 'r']:
            node = env.RSUs[env.rsu_ids_as_index[idx]]
        elif type in ['cloud_server', 'c']:
            node = env.cloudServers[env.cloud_server_ids_as_index[idx]]
        return node.to_dict()
    
    @staticmethod
    def getAllNodeInfos(env, type_list = ['vehicle', 'uav', 'rsu', 'cloud_server']):
        """Get all node infos. type_list = ['vehicle', 'uav', 'rsu', 'cloud_server']

        Args:
            env (AirFogSimEnv): The environment.
            type_list (list): The list of the required types.

        Returns:
            list: The list of the node infos.
        """
        all_nodes = []
        for required_type in type_list:
            if required_type == 'vehicle':
                all_nodes += list(env.vehicles.values())
            elif required_type == 'uav':
                all_nodes += list(env.UAVs.values())
            elif required_type == 'rsu':
                all_nodes += list(env.RSUs.values())
            elif required_type == 'cloud_server':
                all_nodes += list(env.cloudServers.values())
        all_node_infos = []
        for node in all_nodes:
            all_node_infos.append(node.to_dict())
        return all_node_infos

    @staticmethod
    def getAllTaskNodeIds(env):
        """Get all the task node ids.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of the task node ids. (copy of the original list)
        """
        return env.task_node_ids.copy()
    
    @staticmethod
    def setTaskNodeIds(env, task_node_ids):
        """Set the task node ids.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            task_node_ids (list): The list of the task node ids.
        """
        env.task_node_ids = task_node_ids
        
    @staticmethod
    def getNodeInfoById(env, node_id: str):
        """Get the node info by the node id.

        Args:
            env (AirFogSimEnv): The environment.
            node_id (str): The node id.

        Returns:
            dict: The node info.
        """
        node = env._getNodeById(node_id)
        return node.to_dict()

    @staticmethod
    def getNeighborNodeInfosById(env, node_id: str, sorted_by = 'distance', reverse = False):
        """Get the neighbor node infos by the node id.

        Args:
            env (AirFogSimEnv): The environment.
            node_id (str): The node id.
            sorted_by (str): The attribute to sort the neighbor node infos. ['distance', 'cpu']
            reverse (bool): The flag to indicate whether the neighbor node infos are sorted in reverse

        Returns:
            list: The neighbor node infos.
        """
        assert sorted_by in ['distance', 'cpu'], "sorted_by should be 'distance' or 'cpu'"
        node = env._getNodeById(node_id)
        all_nodes = list(env.vehicles.values()) + list(env.UAVs.values()) + list(env.RSUs.values()) + list(env.cloudServers.values())
        neighbor_node_infos = []
        cpu_list = []
        distance_list = []
        for n in all_nodes:
            if n.getId() != node_id:
                neighbor_node_infos.append(n.to_dict())
                cpu_list.append(n.getFogProfile()['cpu'])
                distance_list.append(env._getDistanceBetweenNodes(node, n))

        if sorted_by == 'distance':
            neighbor_node_infos = [x for _, x in sorted(zip(distance_list, neighbor_node_infos), key=lambda pair: pair[0], reverse=reverse)]
        elif sorted_by == 'cpu':
            neighbor_node_infos = [x for _, x in sorted(zip(cpu_list, neighbor_node_infos), key=lambda pair: pair[0], reverse=reverse)]
        return neighbor_node_infos
