from ..entities.fog_node import FogNode
from ..airfogsim_env import AirFogSimEnv
from .base_sched import BaseScheduler
class ComputationScheduler(BaseScheduler):
    """The computation scheduler for the fog nodes. Provide static methods to schedule the computation tasks for the fog nodes.
    """

    @staticmethod
    def setCPUByNodeName(env:AirFogSimEnv, node_name: str, resource_allocation: list):
        """Schedule the computation resources of the fog node (e.g., fog vehicle, edge server, cloud server) by the fog node name.

        Args:
            env (AirFogSimEnv): The environment.
            fog_node_name (str): The fog node name.
            resource_allocation (list): The resource allocation list. Each element is the resource allocation ratio for the computation tasks. Guarantee the sum of the resource allocation ratio is 1.

        Returns:
            bool: The flag to indicate whether the computation resources are scheduled successfully.
        """
        return True