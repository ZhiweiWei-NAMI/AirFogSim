from .manager.traffic_manager import TrafficManager
from .manager.task_manager import TaskManager
from .manager.channel_manager import ChannelManager
from .manager.block_manager import BlockchainManager
from .entities.vehicle import Vehicle
from .entities.rsu import RSU
from .entities.uav import UAV
from .entities.cloud_server import CloudServer
from .entities.task import Task
from .manager.mission_manager import MissionManager
from .enum_const import EnumerateConstants
import traci
import numpy as np
import time

class AirFogSimEnv():
    """AirFogSimEnv is the main class for the airfogsim environment. It provides the simulation of communication, computation, storage, battery, vehicle/UAV trajectory, cloud/cloudlet nodes, AI models for entities, blockchain, authentication, and privacy. It also provides the APIs for the agent to interact with the environment. The agent can be a DRL agent, a rule-based agent, or a human player.
    """

    def __init__(self, config):
        """The constructor of the AirFogSimEnv class. It initializes the environment with the given configuration.

        Args:
            config (dict): The configuration of the environment. Please follow standard YAML format.
        """
        self.vehicles = {}
        self.vehicle_ids_as_index = [] # vehicle ids as a list, used for indexing
        self.removed_vehicles = []

        self.RSUs = {}
        self.rsu_ids_as_index = [] # RSU ids as a list, used for indexing

        self.UAVs = {}
        self.uav_ids_as_index = [] # UAV ids as a list, used for indexing

        self.cloudServers = {}
        self.cloud_server_ids_as_index = [] # cloud server ids as a list, used for indexing

        self.config = config
        
        self.fog_profile = {}
        self.fog_profile['vehicle'] = {'cpu': config['fog_profile']['vehicle']['cpu'], 'memory': config['fog_profile']['vehicle']['memory'], 'storage': config['fog_profile']['vehicle']['storage']}
        self.fog_profile['uav'] = {'cpu': config['fog_profile']['uav']['cpu'], 'memory': config['fog_profile']['uav']['memory'], 'storage': config['fog_profile']['uav']['storage']}
        self.fog_profile['rsu'] = {'cpu': config['fog_profile']['rsu']['cpu'], 'memory': config['fog_profile']['rsu']['memory'], 'storage': config['fog_profile']['rsu']['storage']}
        self.fog_profile['cloud'] = {'cpu': config['fog_profile']['cloud']['cpu'], 'memory': config['fog_profile']['cloud']['memory'], 'storage': config['fog_profile']['cloud']['storage']}

        self.simulation_time = 0
        self.max_simulation_time = config['simulation']['max_simulation_time']

        self.simulation_interval = config['simulation']['simulation_interval'] # TTI
        self.traffic_interval = config['simulation']['traffic_interval']

        assert self.traffic_interval >= self.simulation_interval, "The traffic interval should be greater than or equal to the simulation interval!"


        self.traci_connection = self._connectToSUMO(config['sumo'])
        self.traffic_manager = TrafficManager(config['traffic'], self.traci_connection) # traffic interval is decided by sumo simulation step
        self._initRSUsAndCloudServers()
        self.task_manager = TaskManager(predictable_seconds=self.traffic_interval) # suppose tasks are generated every traffic interval
        self.channel_manager = ChannelManager(n_RSU=self.traffic_manager.getNumberOfRSUs(), n_UAV=self.traffic_manager.getNumberOfUAVs(), n_Veh=self.traffic_manager.getNumberOfVehicles(), RSU_positions=self.traffic_manager.getRSUPositions(), simulation_interval=self.simulation_interval)
        self.mission_manager = MissionManager()
        self.blockchain_manager = BlockchainManager(self.RSUs)
        
        self.max_task_node_num = 0
        self.task_node_types = []
        self.task_node_threshold_poss = 0.0

        # ----------------decisions, managed by schedulers----------------
        self.vehicle_mobility_patterns = {} # dict, key是vehicle_id, value是mobility pattern={speed}
        self.uav_mobility_patterns = {} # dict, key是uav_id, value是mobility pattern={angle, phi, speed}
        self.new_missions_for_nodes = {} # dict, key是node_id, value是mission
        self.activated_offloading_tasks_with_RB_Nos = {} # dict, key是task_id, value 是RB的list
        self.compute_tasks_with_cpu = {} # dict, key是task_id, value是对应assigned node分配的cpu
        self.task_node_ids = [] # list, 存储所有的task node id。可以在每个决策时隙更新，即每个车辆在不同的时候可能是fog node或task node。注意，每次生成的任务数量是按照"predictable_seconds"来预先存储的，所以可能t=0.1s时，车辆是task node，t=0.2s时，车辆是fog node，但是此时还有该车辆的任务需要进行卸载或计算。
        self.revenue_and_punishment_for_tasks = {} # dict, key是task_id, value是{node_id, amount}
        self.update_AI_models = {} # dict, key是node_id, value是{"model_name": AI model}

    @property
    def airfogsim_label(self):
        return self._sumo_label

    def close(self):
        """Close the environment.
        """
        traci.close()

    def _connectToSUMO(self, config):
        """Connect to the SUMO simulator with a generated label (e.g., airfogsim_{timestamp}).

        Args:
            config (dict): The configuration of the SUMO simulator.
        """
        self._sumo_label = "airfogsim_"+str(time.time())
        traci.start(["sumo", "--no-step-log", "--no-warnings", "--log", "sumo.log", "-c", config['sumo_config']], port=config['sumo_port'], label=self._sumo_label)
        assert self.traffic_interval == traci.simulation.getDeltaT(), "The traffic interval is not equal to the simulation step in SUMO!"
        traci_connection = traci.getConnection(self._sumo_label)
        return traci_connection

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
        # 0. Update the traffics (positions, speeds, routes, etc.)
        self._updateTraffics()
        # 1. Update the AI models (Federated Learning, Transfer Learning, etc.)
        self._updateAIModels()
        sim_step_per_traffic_step = int(self.traffic_interval / self.simulation_interval)
        for _ in range(sim_step_per_traffic_step):
            # 2. Update the authentication and privacy
            self._updateAuthPrivacy()
            # 3. Update the communication (wireless, V2V, V2I, V2U, etc.) for fog computing nodes.
            self._updateWirelessCommunication()
            # 4. Update the communication (wired, backhaul, fronthaul, etc.) for cloud computing network nodes.
            self._updateWiredCommunication()
            # 5. Update the computation
            self._updateComputation()
            # 6. Update the storage (cache, memory, etc.)
            self._updateStorage()
            # 7. Update the task
            self._updateTask()
            # 8. Update the battery
            self._updateBattery()
            # 9. Update the blockchain
            self._updateBlockchain()
            # 10. Update the mission (such as crowd sensing, data collection, etc.) and add new missions for the entities.
            self._updateMission()
            # Update the simulation time
            self.simulation_time += self.simulation_interval
        # ensure the simulation time is the same as the traffic time
        self.simulation_time = self.traffic_manager.getCurrentTime()
        return self.isDone()
    
    def _updateMission(self):
        """Update the mission for the entities.
        """
        # check the location duration of each waypoint in the mission; check the task execution at each waypoint; check the mission deadline
        self.mission_manager.updateMissions(self.simulation_interval, self.simulation_time, self._getNodeById)
        for node_id, mission in self.new_missions_for_nodes.items():
            # the task for mission is executed locally instead of being offloaded
            self.mission_manager.addMission(node_id, mission, self.task_manager, self.simulation_time)

    def _updateAuthPrivacy(self):
        """Update the authentication and privacy for the entities.
        """
        # if vehicles are not authenticated, they are prohibited by task_manager, channel_manager, and mission_manager.
        pass

    def _updateAIModels(self):
        """Update the AI models. Not training the AI models, just updating the AI models when Federated Learning, Transfer Learning, in new regions, etc.
        """
        for node_id, model_dict in self.update_AI_models.items():
            node = self._getNodeById(node_id)
            for model_name, model in model_dict.items():
                node.updateAIModel(model_name, model)

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
        self.channel_manager.updateFastFading(self.UAVs, self.vehicles, self.vehicle_ids_as_index, self.uav_ids_as_index)
        self.channel_manager.computeRate(activated_task_dict)

    def _execute_communication(self, activated_task_dict):
        """Execute the communication for the offloading tasks. According to channel rate and SINR, the tasks are transmitted to the target nodes. If tasks need to transmit to multiple nodes, the tasks are transmitted one by one.

        Args:
            activated_task_dict (dict): The activated offloading tasks with the profiles. The key is the task ID, and the value is dict {tx_idx, rx_idx, channel_type, task}
        """
        tmp_succeed_tasks = []
        tmp_failed_tasks = []
        for task_idx, task_profile in activated_task_dict.items():
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
            flag = self.task_manager.finishOffloadingTask(task_profile['task'], self.simulation_time)
            assert flag, 'Unexpected error occurs when finishing the offloading task! Possibly due to that task (node) id has been removed in task manager!'
        for task_profile in tmp_failed_tasks:
            flag = self.task_manager.failOffloadingTask(task_profile['task'])
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
            SimpleNode: The node.
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
        offloading_tasks, total_num = self.task_manager.getOffloadingTasksWithNumber() # dict, key是node_id, value是task
        activated_tasks = {}
        if total_num == 0:
            return activated_tasks
     
        # 初始化
        self.channel_manager.resetActiveLinks()
        # 遍历激活连接
        for _, task_set in offloading_tasks.items():
            for task in task_set:
                assert isinstance(task, Task)
                assert not task.isExecutedLocally(), '任务已经在本地执行，不需要分配RB！'
                path = task.getToOffloadRoute()
                task_id = task.getTaskId()
                if task_id not in activated_offloading_tasks_with_RB_Nos:
                    continue
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

        @TODO: Implement the wired communication for the cloud computing network nodes. For example, the backhaul communication, the fronthaul communication, etc.
        """
        pass

    def _updateComputation(self):
        """Update the computation for the entities.
        """
        self.task_manager.computeTasks(self.compute_tasks_with_cpu, self.simulation_interval, self.simulation_time)
        
    def _updateStorage(self):
        """Update the storage for the entities.

        @TODO: Implement the storage update for the entities. For example, the cache update, the memory update, etc.
        """
        pass

    def _updateTask(self):
        """Update and generate the task for the entities. 
        """
        task_node_ids_kwardsDict = {}
        for task_node_ids in self.task_node_ids:
            task_node_ids_kwardsDict[task_node_ids] = self._getNodeById(task_node_ids).getTaskProfile()
        # generate task for each task node. It generates the future tasks, and stored in "to_generate_tasks" in the task manager. These tasks are viewed as ``predictable'' tasks.
        self.task_manager.generateAndCheckTasks(task_node_ids_kwardsDict, self.simulation_time, self.simulation_interval)

    def _updateBattery(self):
        """Update the battery for the entities.

        @TODO: Implement the battery update for the entities. For example, the battery consumption, the battery charging, etc.
        """
        pass

    def _updateBlockchain(self):
        """Update the blockchain for the entities.
        """
        self.payAndPunish(self.revenue_and_punishment_for_tasks)
        self.blockchain_manager.generateToMineBlocks(self.simulation_time)
        miner_and_revenues = self.blockchain_manager.chooseMiner()
        self.blockchain_manager.Mining(miner_and_revenues, self.simulation_interval, self.simulation_time)
    
    def payAndPunish(self, revenue_and_punishment_for_tasks):
        """
        Pay and punish the nodes according to the revenue and punishment for the tasks.

        Args:
            revenue_and_punishment_for_tasks (dict): The revenue and punishment for the tasks. The key is the task ID, and the value is the dict {node_id, amount}
        """
        for task_id, info_dict in revenue_and_punishment_for_tasks.items():
            node_id = info_dict['node_id']
            amount = info_dict['amount']
            node = self._getNodeById(node_id)
            node.setRevenue(amount)
            self.blockchain_manager.addTransaction(f"({node_id}, {task_id}, {amount})")

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
    
    def _initVehicle(self, vehicle_traffic_info, task_profile={}, fog_profile={}):
        """Initialize the vehicle.

        Args:
            vehicle_traffic_info (dict): The vehicle traffic information.
            task_profile (dict): The task profile of the vehicle, indicating the keywords of generation model. For example, lambda for Poisson process, low/high for Uniform distribution, etc. This should be consistent with the task generation model.
            fog_profile (dict): The fog profile of the vehicle, indicating cpu, memory, storage, etc.

        Returns:
            Vehicle: The vehicle.

        Examples:
            vehicle_traffic_info = {"id": "Vehicle_1", "position": (0, 0, 0), "speed": 10, "acceleration": 0, "angle": 0}
            task_profile = {"lambda": 0.1}
            fog_profile = {"cpu": 1, "memory": 1, "storage": 1}
        """
        vehicle = Vehicle(vehicle_traffic_info['id'], vehicle_traffic_info['position'], vehicle_traffic_info['speed'], vehicle_traffic_info['acceleration'], vehicle_traffic_info['angle'], task_profile, fog_profile)
        return vehicle
    
    def _getDistanceBetweenNodes(self, node1, node2):
        """Get the distance between two nodes.

        Args:
            node1 (SimpleNode): The first node.
            node2 (SimpleNode): The second node.

        Returns:
            float: The distance between two nodes.
        """
        return np.linalg.norm(np.array(node1.getPosition()) - np.array(node2.getPosition()))
    
    def _initRSUsAndCloudServers(self):
        rsu_infos = self.traffic_manager.getRSUInfos()
        cloudserver_infos = self.traffic_manager.getCloudServerInfos()
        for rsu_id, rsu_info in rsu_infos.items():
            if rsu_id not in self.RSUs:
                self.RSUs[rsu_id] = RSU(id=rsu_id, position=rsu_info['position'], fog_profile=self.fog_profile['rsu'], task_profile={})

        for cloudserver_id, cloudserver_info in cloudserver_infos.items():
            if cloudserver_id not in self.cloudServers:
                self.cloudServers[cloudserver_id] = CloudServer(id=cloudserver_id, position=cloudserver_info['position'], fog_profile=self.fog_profile['cloud'], task_profile={})

    def _updateTraffics(self):
        """Update the vehicle traffics.
        """
        self.traffic_manager.updateVehicleMobilityPatterns(self.vehicle_mobility_patterns)
        self.traffic_manager.updateUAVMobilityPatterns(self.uav_mobility_patterns)
        self.traffic_manager.stepSimulation()
        vehicle_traffic_infos = self.traffic_manager.getVehicleTrafficInfos()
        uav_traffic_infos = self.traffic_manager.getUAVTrafficInfos()
        existing_vehicle_ids = list(self.vehicles.keys())
        certain_vehicle_ids = list(vehicle_traffic_infos.keys())
            
        for vehicle_id, vehicle_traffic_info in vehicle_traffic_infos.items():
            if vehicle_id not in self.vehicles:
                self.vehicles[vehicle_id] = self._initVehicle(vehicle_traffic_info, fog_profile=self.fog_profile['vehicle'])
                # check if the vehicle_id should be in the task_node_ids
                if 'vehicle' in self.task_node_types and np.random.rand() < self.task_node_threshold_poss and len(self.task_node_ids) < self.max_task_node_num and vehicle_id not in self.task_node_ids:
                    self.task_node_ids.append(vehicle_id)
            self.vehicles[vehicle_id].update(vehicle_traffic_info, self.simulation_time)
        
        for uav_id, uav_traffic_info in uav_traffic_infos.items():
            if uav_id not in self.UAVs:
                self.UAVs[uav_id] = UAV(uav_id, uav_traffic_info['position'], uav_traffic_info['speed'], uav_traffic_info['acceleration'], uav_traffic_info['angle'], uav_traffic_info['phi'], fog_profile=self.fog_profile['uav'], task_profile={})
                # check if the uav_id should be in the task_node_ids
                if 'UAV' in self.task_node_types and np.random.rand() < self.task_node_threshold_poss and len(self.task_node_ids) < self.max_task_node_num and uav_id not in self.task_node_ids:
                    self.task_node_ids.append(uav_id)
            self.UAVs[uav_id].update(uav_traffic_info, self.simulation_time)

        to_delete_vehicle_ids = list(set(existing_vehicle_ids) - set(certain_vehicle_ids))
        for vehicle_id in to_delete_vehicle_ids.copy():
            self._removeVehicle(vehicle_id)
        n_Vehicles = len(self.vehicles)
        n_UAVs = len(self.UAVs)
        n_RSUs = len(self.RSUs)
        self.vehicle_ids_as_index = list(self.vehicles.keys())
        self.uav_ids_as_index = list(self.UAVs.keys())
        self.rsu_ids_as_index = list(self.RSUs.keys())
        self.cloud_server_ids_as_index = list(self.cloudServers.keys())
        self.channel_manager.updateNodes(n_Vehicles, n_UAVs, n_RSUs)

    def _removeVehicle(self, vehicle_id):
        """Remove the vehicle safely by the given id. The tasks of the vehicle will be removed as well.

        Args:
            vehicle_id (str): The id of the vehicle.
        """
        if vehicle_id in self.task_node_ids:
            self.task_node_ids.remove(vehicle_id)
        if vehicle_id in self.vehicles:
            self.task_manager.removeTasksByNodeId(vehicle_id)
            self.vehicle_ids_as_index.remove(vehicle_id)
            del self.vehicles[vehicle_id]
            self.removed_vehicles.append(vehicle_id)