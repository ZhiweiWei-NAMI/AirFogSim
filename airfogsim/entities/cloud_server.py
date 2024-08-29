from .abstract.fog_node import FogNode
from .abstract.task_node import TaskNode
from .abstract.network_node import NetworkNode

class CloudServer(FogNode, TaskNode, NetworkNode):
    """The class for cloud servers.
    """
    def __init__(self, id, position, task_profile=None, fog_profile=None, network_profile=None):
        """The constructor of the CloudServer class.

        Args:
            id (str): The unique ID of the cloud server.
            position (tuple): The position of the cloud server.
            task_profile (dict): The task profile of the cloud server.
            fog_profile (dict): The fog profile of the cloud server.
            network_profile (dict): The network profile of the cloud server.
        """
        position_x, position_y, position_z = position
        TaskNode.__init__(self, id, position_x, position_y, position_z, 0, 0, 0, task_profile)
        FogNode.__init__(self, id, position_x, position_y, position_z, 0, 0, 0, fog_profile)
        NetworkNode.__init__(self, id, position_x, position_y, position_z, 0, 0, 0, network_profile)