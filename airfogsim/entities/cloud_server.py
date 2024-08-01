from .abstract.fog_node import FogNode
from .abstract.task_node import TaskNode
from .abstract.network_node import NetworkNode

class CloudServer(FogNode, TaskNode, NetworkNode):
    """The class for cloud servers.
    """
    def __init__(self, id, position_x=0, position_y=0, position_z=0, speed=0, acceleration=0, task_profile=None, fog_profile=None, network_profile=None):
        """The constructor of the CloudServer class.

        Args:
            id (str): The unique ID of the cloud server.
            position_x (float): The x-coordinate of the cloud server.
            position_y (float): The y-coordinate of the cloud server.
            position_z (float): The z-coordinate of the cloud server.
            speed (float): The speed of the cloud server.
            acceleration (float): The acceleration of the cloud server.
            task_profile (dict): The task profile of the cloud server.
            fog_profile (dict): The fog profile of the cloud server.
            network_profile (dict): The network profile of the cloud server.
        """
        TaskNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, task_profile)
        FogNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, fog_profile)
        NetworkNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, network_profile)