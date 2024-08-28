from .entities.abstract.fog_node import FogNode
from .entities.abstract.task_node import TaskNode
from .entities.abstract.network_node import NetworkNode # the core network node
from .manager.traffic_manager import TrafficManager
from .manager.task_manager import TaskManager
from .manager.channel_manager import ChannelManager
from .entities.vehicle import Vehicle
from .entities.rsu import RSU
from .entities.uav import UAV
from .entities.cloud_server import CloudServer
from .entities.task import Task
from .entities.mission import Mission
from .manager.mission_manager import MissionManager
from .enum_const import EnumerateConstants
import numpy as np


class AirFogSimEnv():
    """AirFogSimEnv is the main class for the airfogsim environment. It provides the simulation of communication, computation, storage, battery, vehicle/UAV trajectory, cloud/cloudlet nodes, AI models for entities, blockchain, authentication, and privacy. It also provides the APIs for the agent to interact with the environment. The agent can be a DRL agent, a rule-based agent, or a human player.
    """

    def _init_(self, config):
        """The constructor of the AirFogSimEnv class. It initializes the environment with the given configuration.

        Args:
            config (dict): The configuration of the environment. Please follow standard YAML format.
        """
        self.vehicles = {}
        self.vehicle_ids_as_index = [] # vehicle ids as a list, used for indexing

        self.RSUs = {}
        self.rsu_ids_as_index = [] # RSU ids as a list, used for indexing

        self.UAVs = {}
        self.uav_ids_as_index = [] # UAV ids as a list, used for indexing

        self.cloudServers = {}
        self.cloud_server_ids_as_index = [] # cloud server ids as a list, used for indexing

        self.config = config

        self.simulation_time = 0
        self.max_simulation_time = config['simulation']['max_simulation_time']

        self.simulation_interval = config['simulation']['simulation_interval'] # TTI
        self.traffic_interval = config['simulation']['traffic_interval']


        self.traci_connection = self._connectToSUMO(config['sumo'])
        self.traffic_manager = TrafficManager(config['traffic'], self.traci_connection) # traffic interval is decided by sumo simulation step
        self.task_mananger = TaskManager()
        self.channel_manager = ChannelManager()
        self.mission_manager = MissionManager()

        # ----------------decisions, managed by schedulers----------------
        self.activated_offloading_tasks_with_RB_Nos = {} # dict, key是task_id, value 是RB的list
        self.new_missions_for_nodes = {} # dict, key是node_id, value是mission



    def _connectToSUMO(self, config):
        """Connect to the SUMO simulator.

        Args:
            config (dict): The configuration of the SUMO simulator.

        Returns:
            traci: The traci connection.
        """
        return None

    def isDone(self):
        """Check whether the environment is done.

        Returns:
            bool: The done signal. True if the episode is done, False otherwise.
        """
        return self.simulation_time >= self.max_simulation_time

    def step(self):
        """The step function of the environment. It simulates the environment for one time step.

        Returns:
            bool: The done signal. True if the episode is done, False otherwise.
        """
        # 1. Update the traffics (positions, speeds, routes, etc.)
        self._updateTraffics()
        # 2. Update the authentication and privacy
        self._updateAuthPrivacy()
        # 3. Update the AI models (e.g., enter a new region) for mobile entities. Neglect communication and computation for it.
        self._updateAIModels()
        # 4. Update the communication (wireless, V2V, V2I, V2U, etc.) for fog computing nodes.
        self._updateWirelessCommunication()
        # 5. Update the communication (wired, backhaul, fronthaul, etc.) for cloud computing network nodes.
        self._updateWiredCommunication()
        # 6. Update the computation
        self._updateComputation()
        # 7. Update the storage (cache, memory, etc.)
        self._updateStorage()
        # 8. Update the task
        self._updateTask()
        # 9. Update the battery
        self._updateBattery()
        # 10. Update the blockchain
        self._updateBlockchain()
        # 11. Update the mission (such as crowd sensing, data collection, etc.) and add new missions for the entities.
        self._updateMission()
        return self.isDone()
    
    def _updateMission(self):
        """Update the mission for the entities.
        """
        # check the location duration of each waypoint in the mission; check the task execution at each waypoint; check the mission deadline
        self.mission_manager.updateMissions(self.simulation_interval, self.simulation_time, self._getNodeById)
        for node_id, mission in self.new_missions_for_nodes.items():
            # the task for mission is executed locally instead of being offloaded
            self.mission_manager.addMission(node_id, mission, self.task_mananger)

    def _updateAuthPrivacy(self):
        """Update the authentication and privacy for the entities.
        """
        pass

    def _updateAIModels(self):
        """Update the AI models for the moving entities. Not training the AI models, just updating the AI models when the entities enter a new region.
        """
        pass

    def _updateWirelessCommunication(self):
        """Update the wireless communication for the fog computing nodes.
        """
        activated_task_dict = self._allocate_communication_RBs(self.activated_offloading_tasks_with_RB_Nos)
        self._compute_communication_rate(activated_task_dict)
        self._execute_communication(activated_task_dict)

    def _compute_communication_rate(self, activated_task_dict):
        """Compute the communication rate for the offloading tasks. The communication rate is computed based on the channel model, such as path loss, shadowing, fading, etc.

        Args:
            activated_task_dict (dict): The activated offloading tasks with the profiles. The key is the task ID, and the value is dict {tx_idx, rx_idx, channel_type, task}
        """
        self.channel_manager.computeRate(activated_task_dict)

    def _execute_communication(self, activated_task_dict):
        """Execute the communication for the offloading tasks. According to channel rate and SINR, the tasks are transmitted to the target nodes. If tasks need to transmit to multiple nodes, the tasks are transmitted one by one.

        Args:
            activated_task_dict (dict): The activated offloading tasks with the profiles. The key is the task ID, and the value is dict {tx_idx, rx_idx, channel_type, task}
        """
        tmp_succeed_tasks = []
        tmp_failed_tasks = []
        for task_idx, task_profile in enumerate(activated_task_dict):
            task = task_profile['task']
            assert isinstance(task, Task)
            channel_type = task_profile['channel_type']
            tx_idx = task_profile['tx_idx']
            rx_idx = task_profile['rx_idx']
            offload_objs = task.getToOffloadRoute()
            tx_id, rx_id = task.getCurrentNodeId(), offload_objs[0]
            TX_Node, RX_Node = self._getNodeById(tx_id), self._getNodeById(rx_id)
            if TX_Node is None or RX_Node is None:
                task.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_NODE)
                tmp_failed_tasks.append(task_profile)
                continue
            TX_Node.transmitting = False
            RX_Node.receiving = False # 务必每一回都要让这个玩意儿变成负数，然后每个TTI重新分配RB资源
            if task.wait_to_ddl(self.simulation_time):
                task.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_DDL)
                tmp_failed_tasks.append(task_profile)
                continue
            trans_data = np.sum(self.channel_manager.getRateByChannelType(tx_idx, rx_idx, channel_type)) * self.simulation_interval
            
            task.transmit_to_Node(rx_id, trans_data, self.simulation_time)
            if task.isTransmittedToAssignedNode():
                tmp_succeed_tasks.append(task_profile)

        for task_profile in tmp_succeed_tasks:
            flag = self.task_mananger.finishOffloadingTask(task_profile['task'])
            assert flag, 'Unexpected error occurs when finishing the offloading task! Possibly due to that task (node) id has been removed in task manager!'
        for task_profile in tmp_failed_tasks:
            flag = self.task_mananger.failOffloadingTask(task_profile['task'])
            assert flag, 'Unexpected error occurs when failing the offloading task! Possibly due to that task (node) id has been removed in task manager!'

    def _getNodeIdxById(self, node_id):
        """Get the node index by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            int: The index of the node.
        """
        if node_id in self.vehicles:
            return self.vehicle_ids_as_index.index(node_id)
        elif node_id in self.RSUs:
            return self.rsu_ids_as_index.index(node_id)
        elif node_id in self.UAVs:
            return self.uav_ids_as_index.index(node_id)
        elif node_id in self.cloudServers:
            return self.cloud_server_ids_as_index.index(node_id)
        else:
            return -1

    def _getNodeTypeById(self, node_id):
        """Get the node type by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            str: The type of the node. 'V' for vehicle, 'U' for UAV, 'I' for RSU, 'C' for cloud server.
        """
        if node_id in self.vehicles:
            return 'V'
        elif node_id in self.RSUs:
            return 'I'
        elif node_id in self.UAVs:
            return 'U'
        elif node_id in self.cloudServers:
            return 'C'
        else:
            return None
        
    def _getNodeById(self, node_id):
        """Get the node by the given id.

        Args:
            node_id (str): The id of the node.

        Returns:
            Node: The node.
        """
        node = self.UAVs.get(node_id, None)
        if node is None:
            node = self.vehicles.get(node_id, None)
        if node is None:
            node = self.RSUs.get(node_id, None)
        if node is None:
            node = self.cloudServers.get(node_id, None)
        return node

    def _allocate_communication_RBs(self, activated_offloading_tasks_with_RB_Nos:dict):
        """Allocate the communication resources (RBs) for the offloading tasks.
        
        Args:
            activated_offloading_tasks_with_RB_Nos (dict): The activated offloading tasks with RB numbers. The key is the task ID, and the value is the list of RB numbers.

        Returns:
            dict: The activated offloading tasks profiles. The key is the task ID, and the value is the dict {tx_idx, rx_idx, channel_type, task}

        Examples:
            activated_offloading_tasks_with_RB_Nos = {
                'Task_1': [1, 2, 3],
                'Task_2': [4, 5, 6],
                'Task_3': [7, 8, 9]
            }

        """
        offloading_tasks, total_num = self.task_mananger.getOffloadingTasks() # dict, key是node_id, value是task
        if total_num == 0:
            return
     
        # 初始化
        activated_offloading_tasks_with_RB_Nos = np.array(activated_offloading_tasks_with_RB_Nos, dtype='bool')
        self.channel_manager.resetActiveLinks()
        activated_tasks = {}
        # 遍历激活连接
        for node_id, offloading_tasks in enumerate(offloading_tasks): # offloading_tasks是一个list，每个元素是Task对象
            for task in offloading_tasks:
                assert isinstance(task, Task)
                assert not task.isExecutedLocally(), '任务已经在本地执行，不需要分配RB！'
                path = task.getToOffloadRoute()
                assert len(path)>0
                task_id = task.getTaskId()
                allocated_RBs = activated_offloading_tasks_with_RB_Nos[task_id]
                tx, rx = task.getCurrentNodeId(), path[0]
                tx_idx = self._getNodeIdxById(tx)
                rx_idx = self._getNodeIdxById(rx)
                TX_Node = self._getNodeById(tx)
                RX_Node = self._getNodeById(rx)
                TX_Node.setTransmitting(True)
                RX_Node.setReceiving(True)
                TX_Node_type = self._getNodeTypeById(tx) # 'V', 'U', 'I'
                RX_Node_type = self._getNodeTypeById(rx) # 'V', 'U', 'I'
                assert TX_Node_type in ['V', 'U', 'I'], 'TX_Node_type不在["Vehicle", "UAV", "RSU"]中！'
                assert RX_Node_type in ['V', 'U', 'I'], 'RX_Node_type不在["Vehicle", "UAV", "RSU"]中！'
                channel_type=f'{TX_Node_type}2{RX_Node_type}'
                self.channel_manager.activateLink(tx_idx, rx_idx, allocated_RBs, channel_type)
                activated_tasks[task_id] = {
                    'tx_idx': tx_idx,
                    'rx_idx': rx_idx,
                    'channel_type': channel_type,
                    'task': task
                }
        return activated_tasks

    def _updateWiredCommunication(self):
        """Update the wired communication for the cloud computing network nodes.
        """
        pass

    def _updateComputation(self):
        """Update the computation for the entities.
        """
        pass

    def _updateStorage(self):
        """Update the storage for the entities.
        """
        pass

    def _updateTask(self):
        """Update the task for the entities. Classify the successful tasks and failed tasks, also the reasons for the failed tasks.
        """
        pass

    def _updateBattery(self):
        """Update the battery for the entities.
        """
        pass

    def _updateBlockchain(self):
        """Update the blockchain for the entities.
        """
        pass

    def getNodeById(self, id):
        """Get the node by the given id.

        Args:
            id (str): The id of the node. Unique in the environment.

        Returns:
            Node: The node.
        """
        if id in self.vehicles:
            return self.vehicles[id]
        elif id in self.RSUs:
            return self.RSUs[id]
        elif id in self.UAVs:
            return self.UAVs[id]
        elif id in self.cloudServers:
            return self.cloudServers[id]
        else:
            return None
    
    def getVehicleIds(self):
        """Get the vehicle ids.

        Returns:
            list: The vehicle ids.
        """
        return list(self.vehicles.keys())
    
    def getRSUIds(self):
        """Get the RSU ids.

        Returns:
            list: The RSU ids.
        """
        return list(self.RSUs.keys())
    
    def getUAVIds(self):
        """Get the UAV ids.

        Returns:
            list: The UAV ids.
        """
        return list(self.UAVs.keys())
    
    def getCloudServerIds(self):
        """Get the cloud server ids.

        Returns:
            list: The cloud server ids.
        """
        return list(self.cloudServers.keys())
    
    def getVehicleById(self, id):
        """Get the vehicle by the given id.

        Args:
            id (str): The id of the vehicle.

        Returns:
            Vehicle: The vehicle.
        """
        return self.vehicles[id]
    
    def getRSUById(self, id):
        """Get the RSU by the given id.

        Args:
            id (str): The id of the RSU.

        Returns:
            RSU: The RSU.
        """
        return self.RSUs[id]
    
    def getUAVById(self, id):
        """Get the UAV by the given id.

        Args:
            id (str): The id of the UAV.

        Returns:
            UAV: The UAV.
        """
        return self.UAVs[id]
    
    def getCloudServerById(self, id):
        """Get the cloud server by the given id.

        Args:
            id (str): The id of the cloud server.

        Returns:
            CloudServer: The cloud server.
        """
        return self.cloudServers[id]
    
    def _initVehicle(self, vehicle_traffic_info):
        """Initialize the vehicle.

        Args:
            vehicle_traffic_info (dict): The vehicle traffic information.

        Returns:
            Vehicle: The vehicle.
        """
        vehicle = Vehicle(vehicle_traffic_info['id'], vehicle_traffic_info['position'], vehicle_traffic_info['speed'], vehicle_traffic_info['angle'], vehicle_traffic_info['routeId'])
        return vehicle
    
    def _updateTraffics(self):
        """Update the vehicle traffics.
        """
        vehicle_traffic_infos = self.traffic_manager.getVehicleTrafficInfos(self.simulation_time)
        uav_traffic_infos = self.traffic_manager.getUAVTrafficInfos(self.simulation_time)
        rsu_infos = self.traffic_manager.getRSUInfos()
        for vehicle_id, vehicle_traffic_info in vehicle_traffic_infos.items():
            if vehicle_id not in self.vehicles:
                self.vehicles[vehicle_id] = self._initVehicle(vehicle_traffic_info)
            self.vehicles[vehicle_id].update(vehicle_traffic_info, self.simulation_time)
        
        for uav_id, uav_traffic_info in uav_traffic_infos.items():
            if uav_id not in self.UAVs:
                self.UAVs[uav_id] = UAV(uav_traffic_info['id'], uav_traffic_info['position'], uav_traffic_info['speed'], uav_traffic_info['angle'], uav_traffic_info['routeId'])
            self.UAVs[uav_id].update(uav_traffic_info, self.simulation_time)

        for rsu_id, rsu_info in rsu_infos.items():
            if rsu_id not in self.RSUs:
                self.RSUs[rsu_id] = RSU(rsu_info['id'], rsu_info['position'], rsu_info['coverage'])
            self.RSUs[rsu_id].update(rsu_info)

    def _removeVehicle(self, vehicle_id):
        """Remove the vehicle safely by the given id. The tasks of the vehicle will be removed as well.

        Args:
            vehicle_id (str): The id of the vehicle.
        """
        pass
