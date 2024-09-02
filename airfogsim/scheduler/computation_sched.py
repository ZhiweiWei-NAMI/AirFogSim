from .base_sched import BaseScheduler
class ComputationScheduler(BaseScheduler):
    """The computation scheduler for the fog nodes. Provide static methods to schedule the computation tasks for the fog nodes.
    """

    @staticmethod
    def setComputingWithNodeCPU(env, task_id: str, allocated_cpu: float):
        """Set the computing with the node CPU.

        Args:
            env (AirFogSimEnv): The environment.
            task_id (str): The task id.
            allocated_cpu (float): The allocated CPU.
        """
        env.compute_tasks_with_cpu[task_id] = allocated_cpu