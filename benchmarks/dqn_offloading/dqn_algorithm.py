from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
from airfogsim.algorithm.SimpleDQN.dqn import DQN_Agent
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from benchmarks.configs.parse_dqn_args import parseDQNArgs

class DQNOffloadingAlgorithm(BaseAlgorithmModule):

    def __init__(self):
        super().__init__()
    
    def saveModel(self):
        self.DQN_Agent.saveModel()

    def reset(self):
        self.state_dict = {
            'task_node': None,
            'task_data': None,
            'compute_node': None,
            'task_mask': None,
            'compute_node_mask': None,
            'action': None,
            'reward': None
        }

    def initialize(self, env, config):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.env = env
        self.min_position_x, self.max_position_x = 0, 2000
        self.min_position_y, self.max_position_y = 0, 2000
        self.min_position_z, self.max_position_z = 0, 200
        self.min_speed, self.max_speed = 0, 20
        self.task_min_cpu, self.task_max_cpu = config['task'].get('task_min_cpu', 0), config['task'].get('task_max_cpu', 1)
        self.task_min_size, self.task_max_size = config['task'].get('task_min_size', 0), config['task'].get('task_max_size', 1)
        self.task_min_deadline, self.task_max_deadline = config['task'].get('task_min_deadline', 0), config['task'].get('task_max_deadline', 1)
        self.task_min_priority, self.task_max_priority = config['task'].get('task_min_priority', 0), config['task'].get('task_max_priority', 1)
        self.rewardScheduler.setModel(env, 'REWARD', '-task_priority*task_delay')
        self.node_dim = 6
        self.task_dim = 7
        self.state_dict = {
            'task_node': None,
            'task_data': None,
            'compute_node': None,
            'task_mask': None,
            'compute_node_mask': None,
            'action': None,
            'reward': None
        }
        self.fog_type_dict = {
            'V': 0,
            'U': 1,
            'I': 2,
            'C': 3,
        }
        # # lifecycle includes to_generate, to_offload, to_return, computing, computed, returning, finished
        self.task_lifecycle_state_dict = {
            'to_generate': 0,
            'to_offload': 1,
            'to_return': 2,
            'computing': 3,
            'computed': 4,
            'returning': 5,
            'finished': 6
        }
        self.args = parseDQNArgs()
        self.DQN_Agent = DQN_Agent(self.args)
        self.tensorboard_writer = SummaryWriter()
            
    def _encode_node_state(self, node_state, node_type):
        # id, time, position_x, position_y, position_z, speed, fog_profile, node_type
        # ['UAV_9' 0 2655.3572089655477 1030.70353000605 100.0 0 {'lambda': 1} 'U']
        # 选取 position_x, position_y, position_z, speed, fog_profile, node_type, 6维
        # 注意，fog_profile要转为数字；node_type要转为encoding；position_x, position_y, position_z, speed要normalize
        # fog_profile: {'lambda': 1} -> 1
        if node_type == 'FN' or True:
            cpu = node_state[6].get('cpu', 0)
            require_cpu = self.compScheduler.getRequiredComputingResourceByNodeId(self.env, node_state[0])
            profile = cpu - require_cpu
        elif node_type == 'TN':
            profile = node_state[6].get('lambda', 0)
        fog_type = self.fog_type_dict.get(node_state[7], -1)
        position_x = (node_state[2] - self.min_position_x) / (self.max_position_x - self.min_position_x)
        position_y = (node_state[3] - self.min_position_y) / (self.max_position_y - self.min_position_y)
        position_z = (node_state[4] - self.min_position_z) / (self.max_position_z - self.min_position_z)
        speed = (node_state[5] - self.min_speed) / (self.max_speed - self.min_speed)
        state = [position_x, position_y, position_z, speed, profile, fog_type]
        return np.asarray(state)
    
    def _encode_task_state(self, task_state):
        # ['task_id', 'time', 'task_node_id', 'task_size', 'task_cpu', 'required_returned_size', 'task_deadline', 'task_priority', 'task_arrival_time', 'task_lifecycle_state']
        # 选取 task_size, task_cpu, required_returned_size, task_deadline, task_priority, task_arrival_time, 6维
        # 注意，task_size, task_cpu, required_returned_size要normalize
        task_size = (task_state[3] - self.task_min_size) / (self.task_max_size - self.task_min_size)
        task_cpu = (task_state[4] - self.task_min_cpu) / (self.task_max_cpu - self.task_min_cpu)
        required_returned_size = (task_state[5] - self.task_min_size) / (self.task_max_size - self.task_min_size)
        task_deadline = (task_state[6] - self.task_min_deadline) / (self.task_max_deadline - self.task_min_deadline)
        task_priority = (task_state[7] - self.task_min_priority) / (self.task_max_priority - self.task_min_priority)
        task_arrival_time = (task_state[1] - task_state[8]) / (self.task_max_deadline - self.task_min_deadline) # time - task_arrival_time, 即任务已经等待的时间
        task_lifecycle_state = self.task_lifecycle_state_dict.get(task_state[9], -1)
        state = [task_size, task_cpu, required_returned_size, task_deadline, task_priority, task_arrival_time, task_lifecycle_state]
        return np.asarray(state)

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

    def scheduleMission(self, env: AirFogSimEnv):
        return

    def scheduleReturning(self, env: AirFogSimEnv):
        # 直接返回给task node
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                return_route = [task.getTaskNodeId()]
                self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)

    def scheduleTraffic(self, env: AirFogSimEnv):
        """The UAV traffic scheduling logic. Should be implemented by the subclass. Default is move to the nearest
         mission sensing or task position. If there is no mission allocated to UAV, movement is random.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        UAVs_info = self.trafficScheduler.getUAVTrafficInfos(env)
        UAVs_mobile_pattern = {}
        for UAV_id, UAV_info in UAVs_info.items():
            current_position = UAV_info['position']
            target_position = self.trafficScheduler.getRandomTargetPositionForUAV(env, UAV_id)
            mobility_pattern = self.trafficScheduler.getDefaultUAVMobilityPattern(env, UAV_id, current_position, target_position)
            UAVs_mobile_pattern[UAV_id] = mobility_pattern
        self.trafficScheduler.setUAVMobilityPatterns(env, UAVs_mobile_pattern)

    def _reOrderTaskNodeAndTaskData(self, task_nodes, task_data, task_node_ids):
        # 可以根据规则对于task node进行排序或筛选，对应的task信息也就删掉了（只要修改task_node_id_as_idx即可）
        # 要按照task_node对应找task_data；如果超过max_tasks，就要新生成一个task_node存储多出来的task
        task_node_id_as_idx = task_node_ids.copy()
        task_node_id_as_idx = sorted(task_node_id_as_idx, key=lambda x: self.taskScheduler.getToOffloadTaskNumberByTaskNode(self.env, x), reverse=True) # reverse=True表示降序
        task_data_np = np.zeros((self.args.m1, self.args.max_tasks, self.args.d_task))
        task_mask = np.zeros((self.args.m1, self.args.max_tasks))
        task_node_id_as_idx = task_node_id_as_idx[:self.args.m1] # 只取前m1个task node
        task_id_as_idx = [-1] * self.args.m1 * self.args.max_tasks
        task_node_task_cnt = [0] * self.args.m1
        surplus_task_data = {} # task_node_id -> task_data
        task_node_dict = {task_node[0]: task_node for task_node in task_nodes} # task_node_id -> task_data
        for i, task in enumerate(task_data):
            task_node_id = task[2]
            if task_node_id not in task_node_id_as_idx:
                continue
            task_node_index = task_node_id_as_idx.index(task_node_id)
            if task_node_task_cnt[task_node_index] < self.args.max_tasks:
                task_data_np[task_node_index][task_node_task_cnt[task_node_index]] = self._encode_task_state(task)
                task_mask[task_node_index][task_node_task_cnt[task_node_index]] = 1
                task_id_as_idx[task_node_index * self.args.max_tasks + task_node_task_cnt[task_node_index]] = task[0]
                task_node_task_cnt[task_node_index] += 1
            else:
                surplus_task_data[task_node_id] = surplus_task_data.get(task_node_id, [])
                surplus_task_data[task_node_id].append(task)
        # 对于超出的task，重新分配到新的task node
        task_node_ptr = len(task_node_id_as_idx)
        for task_node_id, tasks in surplus_task_data.items():
            if task_node_ptr >= self.args.m1:
                break
            while len(tasks) > 0:
                if task_node_ptr >= self.args.m1:
                    break
                task_node_id_as_idx.append(task_node_id)
                en_num = min(len(tasks), self.args.max_tasks)
                for i in range(en_num):
                    task_data_np[task_node_ptr][i] = self._encode_task_state(tasks[i])
                    task_mask[task_node_ptr][i] = 1
                    task_id_as_idx[task_node_ptr * self.args.max_tasks + i] = tasks[i][0]
                task_node_ptr += 1
                tasks = tasks[en_num:]
        # 处理完task data后，按照task_node_id_as_idx，对应生成task_node_np
        task_node_np = np.zeros((self.args.m1, self.args.d_node))
        for i, task_node_id in enumerate(task_node_id_as_idx):
            task_node = task_node_dict[task_node_id]
            task_node_np[i] = self._encode_node_state(task_node, 'TN')
        return task_node_np, task_data_np, task_node_id_as_idx, task_id_as_idx, task_mask
    
    def _reOrderComputeNode(self, compute_nodes):
        # 可以根据规则对于compute node进行排序或筛选
        compute_node_np = np.zeros((self.args.m2, self.args.d_node))
        compute_node_mask = np.zeros((self.args.m2))
        compute_node_id_as_idx = []
        # compute_nodes按照cpu降序排列
        # compute_nodes = sorted(compute_nodes, key=lambda x: x[6].get('cpu', 0), reverse=True)
        for i, compute_node in enumerate(compute_nodes):
            if i >= self.args.m2:
                break
            compute_node_np[i] = self._encode_node_state(compute_node, 'FN')
            compute_node_mask[i] = 1
            compute_node_id_as_idx.append(compute_node[0])
        return compute_node_np, compute_node_id_as_idx, compute_node_mask
    
    def _extend_compute_node_mask_wrt_tasks(self, env, compute_node_mask, task_mask, compute_node_id_as_idx, task_node_id_as_idx, task_id_as_idx):
        # 把compute_node_mask从m2扩展到 [m1*max_tasks, m2]
        new_compute_node_mask = np.zeros((self.args.m1, self.args.max_tasks, self.args.m2))
        # 遍历compute_node_id和task_node_id，获取对应的rate
        task_compute_node_rate, wait_delay = self.commScheduler.getEstimatedRateBetweenNodeIds(env, task_node_id_as_idx, compute_node_id_as_idx)
        current_time = env.simulation_time
        for i in range(len(task_node_id_as_idx)):
            task_node_id = task_node_id_as_idx[i]
            for j in range(self.args.max_tasks):
                if task_mask[i][j] == 1:
                    new_compute_node_mask[i, j, :] = compute_node_mask
                    task_id = task_id_as_idx[i * self.args.max_tasks + j]
                    task_info = self.taskScheduler.getTaskInfoByTaskNodeAndTaskId(env, task_node_id, task_id)
                    cpu = task_info.get('task_cpu', 0)
                    data_size = task_info.get('task_size', 0)
                    deadline = task_info.get('task_deadline', 0)
                    arrival_time = task_info.get('task_arrival_time', 0)
                    remained_delay = deadline + arrival_time - current_time
                    if remained_delay <= 0:
                        new_compute_node_mask[i, j, :] = np.zeros((self.args.m2))
                        continue
                    for k in range(len(compute_node_id_as_idx)):
                        compute_node_id = compute_node_id_as_idx[k]
                        rate = task_compute_node_rate[i][k]
                        rate = max(rate, 0.1)
                        comp_delay = self.compScheduler.getComputeDelayByNodeId(env, compute_node_id, cpu)
                        trans_delay = data_size / rate
                        total_delay = comp_delay + trans_delay + wait_delay
                        if total_delay > remained_delay:
                            new_compute_node_mask[i, j, k] = 0
                else:
                    new_compute_node_mask[i, j, :] = np.zeros((self.args.m2))
        new_compute_node_mask = new_compute_node_mask.reshape((self.args.m1 * self.args.max_tasks, self.args.m2))
        return new_compute_node_mask

    
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
        # new_compute_node_mask = self._extend_compute_node_mask_wrt_tasks(env, compute_node_mask, task_mask, compute_node_id_as_idx, task_node_id_as_idx, task_id_as_idx)
        # 把compute_node_mask从[m2]扩展到 [m1*max_tasks, m2]
        new_compute_node_mask = np.tile(compute_node_mask, (self.args.m1 * self.args.max_tasks, 1))
        # [m1 * max_tasks, m2]
        # action: [m1 * max_tasks]
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

    def scheduleCommunication(self, env: AirFogSimEnv):
        super().scheduleCommunication(env)


    def scheduleComputing(self, env: AirFogSimEnv):
        # alloc_cpu_callback function, 用于分配CPU资源的回调函数,输入为_computing_tasks (dict), simulation_interval (float), current_time (float)
        def alloc_cpu_callback(computing_tasks, **kwargs):
            # _computing_tasks: {task_id: task_dict}
            # simulation_interval: float
            # current_time: float
            # 返回值是一个字典，key是task_id，value是分配的cpu
            # 本函数的目的是将所有的cpu分配给task
            appointed_fog_node = set()
            alloc_cpu_dict = {}
            task_list_list = computing_tasks.values()
            sorted_task_list = []
            for task_list in task_list_list:
                sorted_task = sorted(task_list, key=lambda x: x.to_dict().get('compute_order', 100), reverse=False)
                # 再根据先后顺序排序
                sorted_task = sorted(sorted_task, key=lambda x: x.to_dict().get('task_arrival_time', 0), reverse=False)
                sorted_task_list.append(sorted_task)
            for tasks in sorted_task_list:
                for task in tasks:
                    task_dict = task.to_dict()
                    assigned_node_id = task_dict['assigned_to']
                    if assigned_node_id in appointed_fog_node:
                        continue
                    appointed_fog_node.add(assigned_node_id)
                    assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
                    if assigned_node_info is None:
                        continue
                    alloc_cpu = assigned_node_info.get('fog_profile', {}).get('cpu', 0)
                    alloc_cpu_dict[task_dict['task_id']] = alloc_cpu
            return alloc_cpu_dict
        self.compScheduler.setComputingCallBack(env, alloc_cpu_callback) 

    def getRewardByTask(self, env: AirFogSimEnv):
        last_step_succ_task_infos = self.taskScheduler.getLastStepSuccTaskInfos(env)
        # last_step_fail_task_infos = self.taskScheduler.getLastStepFailTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos:
            reward += self.rewardScheduler.getRewardByTask(env, task_info)
            # if task_info['task_delay'] <= task_info['task_deadline']:
            #     reward += 1
        return reward

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)
