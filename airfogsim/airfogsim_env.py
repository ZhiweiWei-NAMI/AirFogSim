from .entities.abstract.fog_node import FogNode
from .entities.abstract.task_node import TaskNode
from .entities.abstract.network_node import NetworkNode # the core network node
from .manager.traffic_manager import TrafficManager
from .entities.vehicle import Vehicle
from .entities.rsu import RSU
from .entities.uav import UAV
from .entities.cloud_server import CloudServer

class AirFogSimEnv():
    """AirFogSimEnv is the main class for the airfogsim environment. It provides the simulation of communication, computation, storage, battery, vehicle/UAV trajectory, cloud/cloudlet nodes, AI models for entities, blockchain, authentication, and privacy. It also provides the APIs for the agent to interact with the environment. The agent can be a DRL agent, a rule-based agent, or a human player.
    """

    def _init_(self, config):
        """The constructor of the AirFogSimEnv class. It initializes the environment with the given configuration.

        Args:
            config (dict): The configuration of the environment. Please follow standard YAML format.
        """
        self.vehicles = {}
        self.RSUs = {}
        self.UAVs = {}
        self.cloudServers = {}

        self.simulation_time = 0
        self.max_simulation_time = config['simulation']['max_simulation_time']


        self.traci_connection = self._connectToSUMO(config['sumo'])
        self.traffic_manager = TrafficManager(config['traffic'], self.traci_connection) # traffic interval is decided by sumo simulation step

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
        # 1. Update the mission (such as crowd sensing, data collection, etc.)
        self._updateMission()
        # 2. Update the traffics (positions, speeds, routes, etc.)
        self._updateTraffics()
        # 3. Update the authentication and privacy
        self._updateAuthPrivacy()
        # 4. Update the AI models (e.g., enter a new region) for mobile entities. Neglect communication and computation for it.
        self._updateAIModels()
        # 5. Generate the task
        self._generateTask()
        # 6. Update the communication (wireless, V2V, V2I, V2U, etc.) for fog computing nodes.
        self._updateWirelessCommunication()
        # 7. Update the communication (wired, backhaul, fronthaul, etc.) for cloud computing network nodes.
        self._updateWiredCommunication()
        # 8. Update the computation
        self._updateComputation()
        # 9. Update the storage (cache, memory, etc.)
        self._updateStorage()
        # 10. Update the task
        self._updateTask()
        # 11. Update the battery
        self._updateBattery()
        # 12. Update the blockchain
        self._updateBlockchain()
        return self.isDone()
    
    def _updateMission(self):
        """Update the mission for the entities.
        """
        pass

    def _updateAuthPrivacy(self):
        """Update the authentication and privacy for the entities.
        """
        pass

    def _updateAIModels(self):
        """Update the AI models for the moving entities. Not training the AI models, just updating the AI models when the entities enter a new region.
        """
        pass

    def _generateTask(self):
        """Generate the task for the entities.
        """
        pass

    def _updateWirelessCommunication(self):
        """Update the wireless communication for the fog computing nodes.
        """
        pass

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
