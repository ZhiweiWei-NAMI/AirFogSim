from airfogsim.airfogsim_env import AirFogSimEnv
from benchmarks.dqn_offloading.dqn_algorithm import DQNOffloadingAlgorithm as SimpleDQNOffloadingAlgorithm
from airfogsim.algorithm.TransDQN.dqn import DQN_Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from benchmarks.configs.parse_dqn_args import parseDQNArgs

class DQNOffloadingAlgorithm(SimpleDQNOffloadingAlgorithm):
    """
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleReturning: Relay (only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct, select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: 
        UAV: Fly to next position in route list and stay for a period of time.
    '''

    def __init__(self):
        super().__init__()
    
    def initialize(self, env, config):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        super().initialize(env, config)
        self.DQN_Agent = DQN_Agent(self.args)
            
    def scheduleStep(self, env: AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleReturning(env)
        # 这里使用DQN
        self.scheduleOffloading(env) 
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleTraffic(env)
        self.DQN_Agent.update(self.tensorboard_writer)

    def scheduleOffloading(self, env: AirFogSimEnv):
        # super().scheduleOffloading(env)
        all_tasks = self.taskScheduler.getAllToOffloadTasks(env, check_dependency=True)
        task_node = self.entityScheduler.getTaskNodeStates(env)
        task_data = self.entityScheduler.getTaskStates(env, all_tasks)
        # 需要维护一个list，存储每个task_node_np对应的task_node_id
        task_node_np, task_data_np, task_node_id_as_idx, task_id_as_idx, task_mask = self._reOrderTaskNodeAndTaskData(task_node, task_data, env.task_node_ids)
        # compute node
        compute_node = self.entityScheduler.getFogNodeStates(env)
        compute_node_np, compute_node_id_as_idx, compute_node_mask = self._reOrderComputeNode(compute_node)
        new_compute_node_mask = np.tile(compute_node_mask, (self.args.m1 * self.args.max_tasks, 1))
        action = self.DQN_Agent.select_action(task_node_np, task_data_np, compute_node_np, task_mask, new_compute_node_mask)
        action = action.reshape((self.args.m1, self.args.max_tasks))
        # 遍历task_mask，仅当其为1，才进行offloading
        for i in range(self.args.m1):
            if i >= len(task_node_id_as_idx):
                break
            task_node_id = task_node_id_as_idx[i]
            for j in range(self.args.max_tasks):
                if task_mask[i][j] == 1:
                    task_id = task_id_as_idx[i * self.args.max_tasks + j]
                    if action[i][j] == 0: # locally executed
                        target_node_id = task_node_id
                    elif action[i][j]-1 < len(compute_node_id_as_idx): # offloaded to fog node
                        target_node_id = compute_node_id_as_idx[action[i][j]-1]
                    else: # offloaded to self, as the unaccessible node
                        target_node_id = task_node_id
                    if task_id != -1:   
                        self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, target_node_id)
        
        # 如果self.state_dict不是None，那么可以获得上一个时隙的状态和reward，结合本时隙的状态，存储到replay buffer中
        if self.state_dict['task_node'] is not None:
            self.state_dict['reward'] = self.getRewardByTask(env)
            self.DQN_Agent.add_experience(self.state_dict['task_node'], 
                                          self.state_dict['task_data'], 
                                          self.state_dict['compute_node'], 
                                          self.state_dict['task_mask'], 
                                          self.state_dict['compute_node_mask'], 
                                          self.state_dict['action'], 
                                          self.state_dict['reward'], 
                                          task_node_np, task_data_np, compute_node_np, task_mask, new_compute_node_mask, 
                                          self.state_dict['done'])
        self.state_dict['task_node'] = task_node_np
        self.state_dict['task_data'] = task_data_np
        self.state_dict['compute_node'] = compute_node_np
        self.state_dict['task_mask'] = task_mask
        self.state_dict['compute_node_mask'] = new_compute_node_mask
        self.state_dict['action'] = action
        self.state_dict['done'] = env.simulation_time >= env.config['simulation']['max_simulation_time'] - env.traffic_interval
