from .abstract.fog_node import FogNode
from .abstract.task_node import TaskNode


class UAV(TaskNode, FogNode):
    """The class for Unmanned Aerial Vehicles (UAVs).
    """
    def __init__(self, id, position_x=0, position_y=0, position_z=0, speed=0, acceleration=0, task_profile=None, fog_profile=None):
        """The constructor of the UAV class.

        Args:
            id (str): The unique ID of the UAV.
            position_x (float): The x-coordinate of the UAV.
            position_y (float): The y-coordinate of the UAV.
            position_z (float): The z-coordinate of the UAV.
            speed (float): The speed of the UAV.
            acceleration (float): The acceleration of the UAV.
            task_profile (dict): The task profile of the UAV.
            fog_profile (dict): The fog profile of the UAV.
        """
        TaskNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, task_profile)
        FogNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, fog_profile)