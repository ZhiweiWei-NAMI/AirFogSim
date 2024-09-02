from .base_sched import BaseScheduler
class CommunicationScheduler(BaseScheduler):
    """The communication scheduler for channels.
    """

    
    @staticmethod
    def getNumberOfRB(env):
        """Get the number of resource blocks.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            int: The number of resource blocks.
        """
        return env.channel_manager.n_RB
    
    @staticmethod
    def setCommunicationWithRB(env, task_id: str, RB_nos: list):
        """Set the communication with the resource blocks.

        Args:
            env (AirFogSimEnv): The environment.
            task_id (str): The task id.
            RB_nos (list): The list of resource block numbers.
        """
        env.activated_offloading_tasks_with_RB_Nos[task_id] = RB_nos