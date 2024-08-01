
from .base_sched import BaseScheduler
from ..airfogsim_env import AirFogSimEnv
class TaskScheduler(BaseScheduler):
    """Scheduler for task scheduling, setting the task generation model and executing the task offloading.
    """

    @staticmethod
    def setTaskGenerationModel(env:AirFogSimEnv, model, predictable_seconds = 10, **kwargs):
        """Set the task generation model for the environment. The task generation model will not affact the determined task infos, so better to set it before the simulation starts.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.
            model (str): The task generation model, including 'Poisson', 'Random', etc.
            predictable_seconds (float, optional): The maximum predictable seconds for the task generation. Defaults to 10.

        Examples:
            taskSched.setTaskGeneration(env, 'Poisson', max_predictable_task_num=10)
        """
        env.getTaskManager().setTaskGenerationModel(model, **kwargs)
        env.getTaskManager().setPredictableSeconds(predictable_seconds)