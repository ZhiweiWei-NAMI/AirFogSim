from .abstract.task_node import TaskNode
from .abstract.fog_node import FogNode
class Vehicle(TaskNode, FogNode):
    """The class for vehicles.
    """
    def __init__(self, id, position_x=0, position_y=0, position_z=0, speed=0, acceleration=0, task_profile=None, fog_profile=None):
        """The constructor of the Vehicle class.

        Args:
            id (str): The unique ID of the vehicle.
            position_x (float): The x-coordinate of the vehicle.
            position_y (float): The y-coordinate of the vehicle.
            position_z (float): The z-coordinate of the vehicle.
            speed (float): The speed of the vehicle.
            acceleration (float): The acceleration of the vehicle.
            task_profile (dict): The task profile of the vehicle.
            fog_profile (dict): The fog profile of the vehicle.
            ip_address (str): The IP address of the vehicle.
        """
        TaskNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, task_profile)
        FogNode.__init__(self, id, position_x, position_y, position_z, speed, acceleration, fog_profile)