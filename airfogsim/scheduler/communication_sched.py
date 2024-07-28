from ..airfogsim_env import AirFogSimEnv
from .base_sched import BaseScheduler
class CommunicationScheduler(BaseScheduler):
    """The communication scheduler for channels.
    """

    @staticmethod
    def setRBByChannelName(env: AirFogSimEnv, channel_type: str, n_RB: int):
        """Schedule the resource blocks of the channel by the channel name.

        Args:
            env (AirFogSimEnv): The environment.
            channel_type (str): The channel type.
            n_RB (int): The number of resource blocks.

        Returns:
            bool: The flag to indicate whether the bandwidth is scheduled successfully.
        """
        return True