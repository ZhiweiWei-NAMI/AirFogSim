
from .abstract.network_node import NetworkNode
from .abstract.fog_node import FogNode
from .abstract.task_node import TaskNode

class RSU(TaskNode, FogNode, NetworkNode):
    """The class for Road Side Units (RSUs).
    """
    def __init__(self, id, position_x=0, position_y=0, position_z=0, speed=0, acceleration=0, task_profile=None, fog_profile=None, network_profile=None):
        """The constructor of the RSU class.

        Args:
            id (str): The unique ID of the RSU.
            position_x (float): The x-coordinate of the RSU.
            position_y (float): The y-coordinate of the RSU.
            position_z (float): The z-coordinate of the RSU.
            speed (float): The speed of the RSU.
            acceleration (float): The acceleration of the RSU.
            task_profile (dict): The task profile of the RSU.
            fog_profile (dict): The fog profile of the RSU.
            network_profile (dict): The network profile of the RSU.
        """
        TaskNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, task_profile)
        FogNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, fog_profile)
        NetworkNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, network_profile)