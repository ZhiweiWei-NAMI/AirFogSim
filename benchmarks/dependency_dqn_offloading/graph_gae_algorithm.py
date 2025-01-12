from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
import torch
from dqn_config import parseDQNArgs
import random
import networkx as nx
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import os
import shutil
class MyHeteroDataset(InMemoryDataset):
    def __init__(self, root, dag_dataset = None, transform=None, pre_transform=None):
        self.dag_dataset = dag_dataset
        super(MyHeteroDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # 列表形式返回原始文件

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # 如果你的数据已经处理好, 可以直接跳过process
        data_list = self.dag_dataset  # Assuming self.dag_dataset is your list of HeteroData objects

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def saveDataset(self, path = None):
        if path is None:
            path = self.processed_dir
        # 确保路径存在
        if not os.path.exists(path):
            os.makedirs(path)
        # 保存数据集
        data, slices = self.collate(self.dag_dataset)
        torch.save((data, slices), os.path.join(path, 'data.pt'))

class GraphEmbeddingAlgorithm(BaseAlgorithmModule):

    def __init__(self):
        super().__init__()

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
        self.dag_dataset = []
            
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
        # 使用预训练 GAE使得task_dependence，以及相邻车辆的状态，都能够被编码
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
        self.scheduleOffloading(env) 
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleTraffic(env)
        self.addGraphEmbeddingExperience(env)
        
    def updateEmbeddingModel(self):
        self.tensorboard_writer = SummaryWriter()
        # 通过self.dag_dataset来更新模型
        # 1. 通过self.dag_dataset构建DataLoader
        # 2. 通过DataLoader来更新GAE模型
        return
    
    def saveDataset(self):
        # 保存self.dag_dataset
        dataset = MyHeteroDataset(root='./data', dag_dataset=self.dag_dataset)
        dataset.dag_dataset = self.dag_dataset  # 假设 self.dag_dataset 已经创建好
        dataset.process() # 保存
        dataset.saveDataset()

    def loadDataset(self):
        # 加载self.dag_dataset
        dataset = MyHeteroDataset(root='./data')
        dataset = dataset[0]
        self.dag_dataset = dataset
    
    def addGraphEmbeddingExperience(self, env: AirFogSimEnv):
        # 添加到self.dag_dataset
        computing_task_infos = self.taskScheduler.getAllComputingTaskInfos(env)
        offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        to_offload_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        all_task_infos = computing_task_infos + offloading_task_infos + to_offload_task_infos
        # task_info包含的信息有：task_id, task_node_id, task_size, task_cpu, required_returned_size, task_deadline, task_priority, task_arrival_time, task_lifecycle_state
        task_states = self.entityScheduler.getTaskStatesByInfo(env, all_task_infos)  # 对于每一个task_states的元素，可以通过self._encode_task_state(task_state)来编码
        compute_nodes = self.entityScheduler.getFogNodeStates(env)  # 对于每一个compute_node的元素，可以通过self._encode_node_state(compute_node, 'FN')来编码
        task_nodes = self.entityScheduler.getTaskNodeStates(env)  # 对于每一个task_node的元素，可以通过self._encode_node_state(task_node, 'TN')来编码
        task_dags = self.taskScheduler.getAllTaskDAGs(env)
        # task_dags是一个字典，每个元素都是一个nx.DiGraph()，key是task_node_id，需要全部merge到一个图中
        task_dag_in_one = nx.DiGraph()
        for task_dag in task_dags.values():
            if task_dag is not None:
                task_dag_in_one = nx.union(task_dag_in_one, task_dag)

        # 1. 构建异构图
        graph = HeteroData()

        # 2. 初始化图，包含task和node的基础信息
        # Task Nodes
        task_ids = []
        task_embeddings = []
        for task_info, task_state in zip(all_task_infos, task_states):
            task_id = task_info['task_id']
            embedding = self._encode_task_state(task_state) # 假设你已经实现了 _encode_task_state 函数
            task_ids.append(task_id)
            task_embeddings.append(embedding)

        # Compute Nodes
        compute_node_embeddings = []
        compute_node_ids = []
        for node in compute_nodes:
            embedding = self._encode_node_state(node, 'FN') # 假设你已经实现了 _encode_node_state 函数
            compute_node_embeddings.append(embedding)
            compute_node_ids.append(node[0])
        graph['compute_node'].x = torch.tensor(compute_node_embeddings, dtype=torch.float)
        graph['compute_node'].node_id = compute_node_ids

        # Task Nodes (as a separate node type for clarity)
        task_node_embeddings = []
        task_node_ids = []
        for node in task_nodes:
            embedding = self._encode_node_state(node, 'TN')
            task_node_embeddings.append(embedding)
            task_node_ids.append(node[0])

        # mapping ids from string to int
        task_id_to_idx = {task_id: i for i, task_id in enumerate(task_ids)}
        compute_node_id_to_idx = {node_id: i for i, node_id in enumerate(compute_node_ids)}
        task_node_id_to_idx = {node_id: i for i, node_id in enumerate(task_node_ids)}

        all_node_id_to_idx = {}
        all_node_id_to_idx.update(task_node_id_to_idx)
        all_node_id_to_idx.update(compute_node_id_to_idx)

        graph['task_node'].x = torch.tensor(task_node_embeddings, dtype=torch.float)
        graph['task_node'].node_id = task_node_ids

        useful_task_ids = []
        useful_task_id_to_idx = {}
        # 3. 遍历task_dag_in_one，为每一个task添加task_node信息和task之间的依赖关系
        task_dependency_edges = []
        for task_id in task_ids:
            # task_node_id = task_id # 假设你的 task_id 就是 task_node_id
            if task_id in task_dag_in_one.nodes:
                for successor in task_dag_in_one.successors(task_id):
                    if successor in task_id_to_idx:
                        if task_id not in useful_task_ids:
                            useful_task_ids.append(task_id)
                            useful_task_id_to_idx[task_id] = useful_task_ids.index(task_id)
                        if successor not in useful_task_ids:
                            useful_task_ids.append(successor)
                            useful_task_id_to_idx[successor] = useful_task_ids.index(successor)
                            
                        task_dependency_edges.append((useful_task_id_to_idx[task_id], useful_task_id_to_idx[successor]))

        # 把graph中不需要的task删除
        task_ids = useful_task_ids
        task_embeddings = [task_embeddings[task_id_to_idx[task_id]] for task_id in task_ids]
        graph['task'].x = torch.tensor(task_embeddings, dtype=torch.float)
        graph['task'].task_id = task_ids
        all_task_infos = [task_info for task_info in all_task_infos if task_info['task_id'] in task_ids]

        # Convert to tensor, ensuring it's long type for edge indices
        task_dependency_edges = torch.tensor(task_dependency_edges, dtype=torch.long).t().contiguous()
        graph['task', 'depends_on', 'task'].edge_index = task_dependency_edges

        # Add association edges based on current_node_id in task_info
        task_cur_node_association_edges = []
        for task_info in all_task_infos:
            task_id = task_info['task_id']
            current_node_id = task_info['current_node_id']
            # check if current_node_id is one of compute node
            if current_node_id in compute_node_ids:
                task_cur_node_association_edges.append((useful_task_id_to_idx[task_id], all_node_id_to_idx[current_node_id]))

        task_task_node_association_edges = []
        # Add association edges based on task_node_id in task_info
        for task_info in all_task_infos:
          task_id = task_info['task_id']
          task_node_id = task_info['task_node_id']
          if task_node_id in task_node_ids:
            task_task_node_association_edges.append((useful_task_id_to_idx[task_id], all_node_id_to_idx[task_node_id]))
        
        # 4. 遍历task_nodes和compute_nodes，为每一个node添加node的相邻信息
        proximity_edges = {}
        all_node_ids = task_node_ids + compute_node_ids
        
        # Create a mapping from node_id to node_type
        node_id_to_type = {}
        for task_node_id in task_node_ids:
            node_id_to_type[task_node_id] = 'task_node'
        for compute_node_id in compute_node_ids:
            node_id_to_type[compute_node_id] = 'compute_node'

        for i in range(len(all_node_ids)):
            for j in range(i + 1, len(all_node_ids)):
                node1_id = all_node_ids[i]
                node2_id = all_node_ids[j]
                distance = self.entityScheduler.getDistanceBetweenNodes(env, node1_id, node2_id)
                if distance <= 150:
                    # Determine the types of the nodes
                    node1_type = node_id_to_type[node1_id]
                    node2_type = node_id_to_type[node2_id]
                    # 根据节点类型将边添加到对应的列表中
                    edge_type = (node1_type, 'near', node2_type)
                    if edge_type not in proximity_edges:
                        proximity_edges[edge_type] = []
                    proximity_edges[edge_type].append((all_node_id_to_idx[node1_id], all_node_id_to_idx[node2_id]))

                    # 添加反向边
                    reverse_edge_type = (node2_type, 'near', node1_type)
                    if reverse_edge_type not in proximity_edges:
                        proximity_edges[reverse_edge_type] = []
                    proximity_edges[reverse_edge_type].append((all_node_id_to_idx[node2_id], all_node_id_to_idx[node1_id]))

        if proximity_edges != {}:
            for edge_type, edges in proximity_edges.items():
                if edges:
                    edge_tensor = torch.tensor(edges, dtype=torch.long).t().contiguous()
                    graph[edge_type].edge_index = edge_tensor
                else:
                    # 添加一个占位符
                    dummy_index = torch.tensor([[0], [0]], dtype=torch.long)
                    graph[edge_type].edge_index = dummy_index
        else:
            # Add a placeholder that won't affect the model
            dummy_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph['task_node', 'near', 'task_node'].edge_index = dummy_index
            graph['compute_node', 'near', 'compute_node'].edge_index = dummy_index
            graph['task_node', 'near', 'compute_node'].edge_index = dummy_index
            graph['compute_node', 'near', 'task_node'].edge_index = dummy_index

        # Add task_node and compute_node association after adding proximity edges
        if task_cur_node_association_edges:
            task_cur_node_association_edges = torch.tensor(task_cur_node_association_edges, dtype=torch.long).t().contiguous()
            graph['task', 'associated_with', 'compute_node'].edge_index = task_cur_node_association_edges
        else:
            dummy_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph['task', 'associated_with', 'compute_node'].edge_index = dummy_index
        
        # Add task_node and task association after adding proximity edges
        if task_task_node_association_edges:
            task_task_node_association_edges = torch.tensor(task_task_node_association_edges, dtype=torch.long).t().contiguous()
            graph['task', 'associated_with', 'task_node'].edge_index = task_task_node_association_edges
        else:
            dummy_index = torch.tensor([[0], [0]], dtype=torch.long)
            graph['task', 'associated_with', 'task_node'].edge_index = dummy_index

        # 5. 把构建的异构图添加到self.dag_dataset中
        self.dag_dataset.append(graph)

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

    def scheduleOffloading(self, env: AirFogSimEnv):
        all_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        for task_dict in all_task_infos:
            task_node_id = task_dict['task_node_id']
            task_id = task_dict['task_id']
            neighbor_infos = self.entityScheduler.getNeighborNodeInfosById(env, task_node_id, sorted_by='distance', max_num=10)
            if len(neighbor_infos) > 0:
                # Randomly select the nearest node to offload the task
                random.shuffle(neighbor_infos)
                nearest_node_id = neighbor_infos[0]['id']
                self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, nearest_node_id)

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
            for tasks in computing_tasks.values():
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
        last_step_fail_task_infos = self.taskScheduler.getLastStepFailTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos+last_step_fail_task_infos:
            reward += self.rewardScheduler.getRewardByTask(env, task_info)
        return reward

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)
