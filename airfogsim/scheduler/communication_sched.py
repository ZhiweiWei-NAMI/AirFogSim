from ..airfogsim_env import AirFogSimEnv
from .base_sched import BaseScheduler
class CommunicationScheduler(BaseScheduler):
    """The communication scheduler for channels.
    """

    @staticmethod
    def scheduleBandwidthbyChannelType(env: AirFogSimEnv, channel_type: str, resource_allocation: list):
        """Schedule the bandwidth of the channel.

        Args:
            env (AirFogSimEnv): The environment.
            channel_type (str): The channel type.
            resource_allocation (list): The resource allocation list. Each element is the resource allocation ratio for the bandwidth. Guarantee the sum of the resource allocation ratio is 1.

        Returns:
            bool: The flag to indicate whether the bandwidth is scheduled successfully.
        """
        return True