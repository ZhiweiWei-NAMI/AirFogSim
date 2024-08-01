import traci

class TrafficManager():
    """The traffic manager class. It manages both vehicle traffic and UAV traffic. It also manipulates the positions of the vehicles, UAVs, RSUs, and cloud servers.
    """

    def __init__(self, config_traffic):
        """Initialize the traffic manager.

        Args:
            config_traffic (dict): The traffic configuration part of the environment configuration.
        """
        self.config_traffic = config_traffic
        self.vehicles = {} # use vehicle_id -> {position, speed, routeId}
        self.UAVs = {}
        self.RSUs = {}
        self.cloudServers = {}
        self.routes = {} # each route is a series of edges in SUMO, routeId -> [edgeId1, edgeId2, ...]
        self.edges = {} # each edge is a series of lanes in SUMO, edgeId -> [laneId1, laneId2, ...]
        self.laneIds = [] # all lane ids in SUMO


    def getVehicleTrafficInfos(simulation_time):
        """Get the vehicle traffics at the given simulation time.

        Args:
            simulation_time (float): The simulation time in seconds.

        Returns:
            dict: The vehicle traffics, including the vehicle id, position, speed, angle, and current routeId.
        """
        return {}
    
    def getUAVTrafficInfos(simulation_time):
        """Get the UAV traffics at the given simulation time. The trajectory of the UAVs is controlled by their missions

        Args:
            simulation_time (float): The simulation time in seconds.

        Returns:
            dict: The UAV traffics, including the UAV id, position, speed, angle, and current routeId.
        """
        return {}
    
    def getRSUInfos():
        """Get the RSU information.

        Returns:
            dict: The RSU information, including the RSU id, position, and coverage information (e.g., covering type)
        """
        return {}