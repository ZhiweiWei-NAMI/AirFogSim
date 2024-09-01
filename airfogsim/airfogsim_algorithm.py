from .airfogsim_scheduler import AirFogSimScheduler
from .airfogsim_env import AirFogSimEnv
import numpy as np
class BaseAlgorithmModule:
    """Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    def __init__(self):
        self.compScheduler = AirFogSimScheduler.getComputationScheduler()
        self.commScheduler = AirFogSimScheduler.getCommunicationScheduler()
        self.entityScheduler = AirFogSimScheduler.getEntityScheduler()
        self.rewardScheduler = AirFogSimScheduler.getRewardScheduler()
        self.taskScheduler = AirFogSimScheduler.getTaskScheduler()
        self.blockchainScheduler = AirFogSimScheduler.getBlockchainScheduler()
        # self.missionScheduler = AirFogSimScheduler.getMissionScheduler()


    def initialize(self, env:AirFogSimEnv):
        """Initialize the algorithm with the environment. Should be implemented by the subclass. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.taskScheduler.setTaskGenerationModel(env, 'Poisson')
        self.taskScheduler.setTaskNodePossibility(env, node_types=['vehicle'], max_num=30, threshold_poss=0.5)
        self.rewardScheduler.setRewardModel(env, 'log(1+(task_deadline-task_delay))')

    def scheduleStep(self, env:AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleOffloading(env)
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        
    def scheduleOffloading(self, env:AirFogSimEnv):
        """The offloading scheduling logic. Should be implemented by the subclass. Default is to offload the task to the nearest node.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        all_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        all_node_infos = self.entityScheduler.getAllNodeInfos(env)
        all_node_infos_dict = {}
        for node_info in all_node_infos:
            all_node_infos_dict[node_info['id']] = node_info
        for task_dict in all_task_infos:
            task_node_id = task_dict['task_node_id']
            task_id = task_dict['task_id']
            task_node = all_node_infos_dict[task_node_id]
            neighbor_infos = self.entityScheduler.getNeighborNodeInfosById(env, task_node_id, sorted_by='distance')
            nearest_node_id = neighbor_infos[0]['id']
            self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, nearest_node_id)

    def scheduleCommunication(self, env:AirFogSimEnv):
        """The communication scheduling logic. Should be implemented by the subclass. Default is random.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        n_RB = self.commScheduler.getNumberOfRB(env)
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        for task_dict in all_offloading_task_infos:
            allocated_RB_nos = np.random.choice(n_RB, 3, replace=False)
            self.commScheduler.setCommunicationWithRB(env, task_dict['task_id'], allocated_RB_nos)

    def scheduleComputing(self, env:AirFogSimEnv):
        """The computing scheduling logic. Should be implemented by the subclass. Default is FIFS.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        for task_dict in all_offloading_task_infos:
            task_id = task_dict['task_id']
            task_node_id = task_dict['task_node_id']
            assigned_node_id = task_dict['assigned_to']
            assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
            self.compScheduler.setComputingWithNodeCPU(env, task_id, 0.3) # allocate cpu 0.3

    def getReward(self, env:AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done tasks in last time.
        
        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_task_infos = self.taskScheduler.getLastStepSuccTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos:
            reward += self.rewardScheduler.getRewardByTask(env, task_info)
        return reward