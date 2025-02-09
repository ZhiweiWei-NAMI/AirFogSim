import numpy as np
from ..entities.task import Task
from ..enum_const import EnumerateConstants
from collections import deque
import itertools

import random
import networkx as nx

def generate_random_dag(node_ids, edge_probability):
    """
    Generates a random DAG using NetworkX.

    Args:
        node_ids: The node ids in the DAG.
        edge_probability: The probability of an edge existing between two nodes.

    Returns:
        A NetworkX DiGraph representing the DAG.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(node_ids)
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            if random.random() < edge_probability:
                dag.add_edge(node_ids[i], node_ids[j])

    # Ensure the graph is acyclic
    if not nx.is_directed_acyclic_graph(dag):
        return generate_random_dag(node_ids, edge_probability) # Regenerate if cyclic

    return dag

def merge_two_dags(dag1, dag2, edge_probability):
    """
    Merges two DAGs into a new DAG.

    Args:
        dag1: The first DAG.
        dag2: The second DAG.
        edge_probability: The probability of an edge existing between two nodes.

    Returns:
        A NetworkX DiGraph representing the merged DAG.
    """
    if dag1 is None:
        return dag2
    merged_dag = nx.union(dag1, dag2)
    # for node1 in dag1.nodes():
    #     for node2 in dag2.nodes():
    #         if random.random() < edge_probability:
    #             merged_dag.add_edge(node1, node2)
    # Ensure the graph is acyclic
    # if not nx.is_directed_acyclic_graph(dag1):
    #     return merge_two_dags(dag1, dag2, edge_probability)
    return merged_dag

class TaskManager:
    """ Task Manager is responsible for generating tasks and managing the task status. 
    """
    SUPPORTED_TASK_GENERATION_MODELS = ['Poisson', 'Uniform', 'Normal', 'Exponential', 'None']
    ATTRIBUTE_MODELS = ['Uniform', 'Normal']
    def __init__(self, config_task, predictable_seconds=2):
        task_generation_model = config_task.get('task_generation_model', 'Poisson')
        task_generation_kwargs = config_task.get('task_generation_kwargs', {})
        cpu_model = config_task.get('cpu_model', 'Uniform')
        cpu_kwargs = config_task.get('cpu_kwargs', {})
        size_model = config_task.get('size_model', 'Uniform')
        size_kwargs = config_task.get('size_kwargs', {})
        required_returned_size_model = config_task.get('required_returned_size_model', 'Uniform')
        required_returned_size_kwargs = config_task.get('required_returned_size_kwargs', {})
        deadline_model = config_task.get('deadline_model', 'Uniform')
        deadline_kwargs = config_task.get('deadline_kwargs', {})
        priority_model = config_task.get('priority_model', 'Uniform')
        priority_kwargs = config_task.get('priority_kwargs', {})
        self._config_task = config_task
        self._task_dependencies = {}

        self._task_generation_model = task_generation_model
        assert task_generation_model in TaskManager.SUPPORTED_TASK_GENERATION_MODELS, 'The task generation model is not supported. Only support {}'.format(TaskManager.SUPPORTED_TASK_GENERATION_MODELS)
        self._generated_task_history = {} # key: node id, value: list of task info
        self._to_generate_task_infos = {} # use Node Id as the key, and the value is the task info list
        self._waiting_to_offload_tasks = {} # after generating, waiting to offload
        self._offloading_tasks = {}
        self._computing_tasks = {}
        self._waiting_to_return_tasks = {} # after computing, waiting to set return route
        self._returning_tasks = {}
        self._done_tasks = {}
        self._out_of_ddl_tasks = {}
        self._removed_tasks = {}
        self._recently_done_100_tasks = deque(maxlen=100)
        self._recently_failed_100_tasks = deque(maxlen=100)
        self._task_id = 0
        self._predictable_seconds = predictable_seconds # the seconds that the task generation can be predicted
        self.setTaskGenerationModel(task_generation_model, **task_generation_kwargs)
        self.setTaskAttributeModel('CPU', cpu_model, **cpu_kwargs)
        self.setTaskAttributeModel('Size', size_model, **size_kwargs)
        self.setTaskAttributeModel('Deadline', deadline_model, **deadline_kwargs)
        self.setTaskAttributeModel('Priority', priority_model, **priority_kwargs)
        self.setTaskAttributeModel('ReturnSize', required_returned_size_model, **required_returned_size_kwargs)

    def reset(self):
        self._to_generate_task_infos = {} # use Node Id as the key, and the value is the task info list
        self._waiting_to_offload_tasks = {} # after generating, waiting to offload
        self._offloading_tasks = {}
        self._computing_tasks = {}
        self._generated_task_history = {}
        self._waiting_to_return_tasks = {} # after computing, waiting to set return route
        self._returning_tasks = {}
        self._done_tasks = {}
        self._out_of_ddl_tasks = {}
        self._removed_tasks = {}
        self._recently_done_100_tasks = deque(maxlen=100)
        self._recently_failed_100_tasks = deque(maxlen=100)
        self._task_id = 0

    def getDoneTasks(self):
        """Get the done task info list.

        Returns:
            list: The list of the done task info.
        """
        done_tasks = []
        for task_node_id, task_infos in self._done_tasks.items():
            for task_info in task_infos:
                done_tasks.append(task_info)
        return done_tasks
    
    def getOutOfDDLTasks(self):
        """Get the task info list that is out of deadline.

        Returns:
            list: The list of the out of deadline task info.
        """
        out_of_ddl_tasks = []
        for task_node_id, task_infos in self._out_of_ddl_tasks.items():
            for task_info in task_infos:
                out_of_ddl_tasks.append(task_info)
        return out_of_ddl_tasks

    def getTaskByTaskNodeAndTaskId(self, task_node_id, task_id):
        """Get the done task by the task node id and the task id.

        Args:
            task_node_id (str): The task node id.
            task_id (str): The task id.

        Returns:
            Task: The task.

        Examples:
            task_manager.getTaskByTaskNodeAndTaskId('vehicle1', 'Task_1')
        """
        all_task_by_task_node_id = self._generated_task_history.get(task_node_id, [])
        for task in all_task_by_task_node_id:
            if task.getTaskId() == task_id:
                return task
        return None

    def setPredictableSeconds(self, predictable_seconds):
        """Set the predictable seconds for the task generation.

        Args:
            predictable_seconds (float): The predictable seconds.

        Examples:
            task_manager.setPredictableSeconds(5)
        """
        self._predictable_seconds = predictable_seconds
    def addToComputeTask(self, task:Task, node_id, current_time):
        """Add the task to the to_compute_tasks.

        Args:
            task (Task): The task to add.
            node_id (str): The assigned node id (namely, the node id that computes the task).

        Examples:
            task_manager.addToComputeTask(task, 'vehicle1', 10.3)
        """
        to_compute_task_list = self._computing_tasks.get(node_id, [])
        to_compute_task_list.append(task)
        task.setAssignedTo(node_id) # 计算节点设为任务产生节点
        task.startToCompute(current_time)
        self._computing_tasks[node_id] = to_compute_task_list

    def getToComputeTasks(self, node_id):
        """Get the tasks to compute by the node id.

        Args:
            node_id (str): The node id.

        Returns:
            list: The list of the tasks to compute.

        Examples:
            task_manager.getToComputeTasks('vehicle1')
        """
        return self._computing_tasks.get(node_id, [])
    
    def setToComputeTasks(self, node_id, task_list):
        """Set the tasks to compute by the node id.

        Args:
            node_id (str): The node id.
            task_list (list): The list of the tasks to compute.

        Examples:
            task_manager.setToComputeTasks('vehicle1', [task1, task2])
        """
        self._computing_tasks[node_id] = task_list

    def finishOffloadingTask(self, task:Task, current_time):
        """Remove the offloading task by the task id, and then move the task to the to_compute_tasks.

        Args:
            task (Task): The task which is finished offloading.

        Returns:
            bool: True if the task is executed successfully, False otherwise.

        Examples:
            task_manager.removeOffloadingTaskByNodeIdAndTaskId('vehicle1', 'Task_1')
        """
        node_id = task.getAssignedTo() # node id is used for to_compute_tasks and to_return_tasks
        task_node_id = task.getTaskNodeId() # task node id is used for to_offload_tasks
        task_id = task.getTaskId() 
        if not task.isReturning(): # if the task is offloading
            for task_info in self._offloading_tasks[task_node_id]:
                if task_info.getTaskId() == task_id:
                    task_info.startToCompute(current_time)
                    self._offloading_tasks[task_node_id].remove(task_info)
                    to_compute_task_list = self._computing_tasks.get(node_id, [])
                    to_compute_task_list.append(task_info)
                    self._computing_tasks[node_id] = to_compute_task_list
                    return True
        else: # if the task is returning
            for task_info in self._returning_tasks[node_id]:
                if task_info.getTaskId() == task_id:
                    self._returning_tasks[node_id].remove(task_info)
                    if task_info.task_delay <= task_info.task_deadline:
                        self._done_tasks[task_node_id] = self._done_tasks.get(task_node_id, [])
                        self._done_tasks[task_node_id].append(task_info)
                        self._recently_done_100_tasks.append(task_info)
                    else:
                        self._out_of_ddl_tasks[task_node_id] = self._out_of_ddl_tasks.get(task_node_id, [])
                        self._out_of_ddl_tasks[task_node_id].append(task_info)
                        self._recently_failed_100_tasks.append(task_info)
                        task_info.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_DDL)
                    return True
        return False
    
    def failOffloadingTask(self, task:Task, failure = EnumerateConstants.TASK_FAIL_OUT_OF_NODE):
        """Remove the offloading task by the task id, and then move the task to the failed_tasks.

        Args:
            task (Task): The task which is failed offloading.

        Returns:
            bool: True if the task is executed successfully, False otherwise.
        """
        node_id = task.getAssignedTo()
        task_node_id = task.getTaskNodeId()
        task_id = task.getTaskId()
        task.setTaskFailueCode(failure)
        if not task.isReturning():
            for task_info in self._offloading_tasks.get(task_node_id, []):
                if task_info.getTaskId() == task_id:
                    self._offloading_tasks[task_node_id].remove(task_info)
                    failed_task_list = self._out_of_ddl_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._out_of_ddl_tasks[task_node_id] = failed_task_list
                    return True
        else:
            for task_info in self._returning_tasks.get(node_id, []):
                if task_info.getTaskId() == task_id:
                    self._returning_tasks[node_id].remove(task_info)
                    failed_task_list = self._out_of_ddl_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._out_of_ddl_tasks[task_node_id] = failed_task_list
                    return True
        return False

    def computeTasks(self, alloc_cpu_callback, simulation_interval, current_time):
        """Compute the tasks by the allocated CPU.

        Args:
            alloc_cpu_callback (function): The callback function to allocate the CPU.
            simulation_interval (float): The simulation interval.
            current_time (float): The current time.
        """
        allocated_cpus = alloc_cpu_callback(self._computing_tasks)
        for node_id, task_infos in self._computing_tasks.items():
            for task_info in task_infos.copy(): # task_info is Task
                task_id = task_info.getTaskId()
                allocated_cpu = allocated_cpus.get(task_id, 0)
                if allocated_cpu == 0:
                    continue
                task_info.compute(allocated_cpu, simulation_interval, current_time)
                if task_info.isComputed():
                    task_infos.remove(task_info)
                    self._waiting_to_return_tasks[node_id] = self._waiting_to_return_tasks.get(node_id, [])
                    self._waiting_to_return_tasks[node_id].append(task_info)
                    # task_info.startToReturn(current_time)
                    # self._returning_tasks[node_id] = self._returning_tasks.get(node_id, [])
                    # self._returning_tasks[node_id].append(task_info)

    def getRecentlyDoneTasks(self):
        """Get the recently done tasks (the maximum number is 100).

        Returns:
            list: The list of the recently done tasks.
        """
        return self._recently_done_100_tasks
    
    def getRecentlyFailedTasks(self):
        """Get the recently failed tasks (the maximum number is 100).

        Returns:
            list: The list of the recently failed tasks.
        """
        return self._recently_failed_100_tasks
    
    def getWaitingToOffloadTasks(self):
        """Get the tasks to offload.

        Returns:
            dict: The tasks to offload. The key is the node id, and the value is the task list.
        """
        return self._waiting_to_offload_tasks
    
    def getWaitingToOffloadTasksByNodeId(self, node_id):
        """Get the tasks to offload by the node id.

        Args:
            node_id (str): The node id.

        Returns:
            list: The list of the tasks to offload.

        Examples:
            task_manager.getWaitingToOffloadTasksByNodeId('vehicle1')
        """
        return self._waiting_to_offload_tasks.get(node_id, [])
    
    def setWaitingToOffloadTasksByNodeId(self, node_id, task_list):
        """Set the tasks to offload by the node id.

        Args:
            node_id (str): The node id.
            task_list (list): The list of the tasks to offload.

        Examples:
            task_manager.setWaitingToOffloadTasksByNodeId('vehicle1', [task1, task2])
        """
        self._waiting_to_offload_tasks[node_id] = task_list
    
    def getOffloadingTasks(self):
        offloading_tasks, num = self.getOffloadingTasksWithNumber()
        return offloading_tasks
    
    def getComputingTasks(self):
        """Get the tasks to compute.

        Returns:
            dict: The tasks to compute. The key is the node id, and the value is the task list.
        """
        return self._computing_tasks

    def getOffloadingTasksWithNumber(self):
        """Get the offloading tasks (transmission) with the total number.

        Returns:
            dict: The offloading tasks. The key is the node id, and the value is the task list.
            int: The total number of the offloading tasks.

        Examples:
            offloading_tasks, total_num = task_manager.getOffloadingTasksWithNumber()
        """
        # 遍历task，if isTransmitting() == True, 则加入到offloading_tasks中
        offloading_tasks = {} # key: transmitter node id, value: list of tasks
        total_num = 0
        for task_node_id, task_infos in self._offloading_tasks.items():
            offloading_tasks[task_node_id] = task_infos
            total_num += len(offloading_tasks[task_node_id])
        for node_id, task_infos in self._returning_tasks.items():
            to_offload_task = offloading_tasks.get(node_id, [])
            to_offload_task.extend(task_infos)
            offloading_tasks[node_id] = to_offload_task
            total_num += len(offloading_tasks[node_id])
        return offloading_tasks, total_num
    
    def getOffloadingTasksByNodeId(self, node_id):
        """Get the offloading tasks by the node id.

        Args:
            node_id (str): The node id.

        Returns:
            list: The list of the offloading tasks.

        Examples:
            task_manager.getOffloadingTasksByNodeId('vehicle1')
        """
        return self._offloading_tasks.get(node_id, [])
    
    def setOffloadingTasksByNodeId(self, node_id, task_list):
        """Set the offloading tasks by the node id.

        Args:
            node_id (str): The node id.
            task_list (list): The list of the offloading tasks.

        Examples:
            task_manager.setOffloadingTasksByNodeId('vehicle1', [task1, task2])
        """
        self._offloading_tasks[node_id] = task_list

    def removeTasksByNodeId(self, to_remove_node_id):
        """Remove the tasks by the node id.

        Args:
            to_remove_node_id (str): The node id.

        Examples:
            task_manager.removeTasksByNodeId('vehicle1')
        """
    
        # remove all related tasks in to_compute_tasks
        for node_id in list(self._computing_tasks.keys()):
            task_set = self._computing_tasks.get(node_id, [])
            if node_id == to_remove_node_id:
                for task_info in task_set.copy():
                    removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                    removed_task_set.append(task_info)
                    self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                    task_set.remove(task_info)
                    self._computing_tasks[node_id] = task_set
                del self._computing_tasks[node_id]
            else:
                for task_info in task_set.copy():
                    if task_info.isRelatedToNode(to_remove_node_id):
                        removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                        removed_task_set.append(task_info)
                        self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                        task_set.remove(task_info)
                        self._computing_tasks[node_id] = task_set
        # remove all related tasks in to_offload_tasks
        for task_node_id in list(self._offloading_tasks.keys()):
            task_set = self._offloading_tasks.get(task_node_id, [])
            if task_node_id == to_remove_node_id:
                for task_info in task_set.copy():
                    removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                    removed_task_set.append(task_info)
                    self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                    task_set.remove(task_info)
                    self._offloading_tasks[task_node_id] = task_set
                del self._offloading_tasks[task_node_id]
            else:
                for task_info in task_set.copy():
                    if task_info.isRelatedToNode(to_remove_node_id):
                        removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                        removed_task_set.append(task_info)
                        self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                        task_set.remove(task_info)
                        self._offloading_tasks[task_node_id] = task_set
        # remove all related tasks in to_return_tasks
        for node_id in list(self._returning_tasks.keys()):
            task_set = self._returning_tasks.get(node_id, [])
            if node_id == to_remove_node_id:
                for task_info in task_set.copy():
                    removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                    removed_task_set.append(task_info)
                    self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                    task_set.remove(task_info)
                    self._returning_tasks[node_id] = task_set
                del self._returning_tasks[node_id]
            else:
                for task_info in task_set.copy():
                    if task_info.isRelatedToNode(to_remove_node_id):
                        removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                        removed_task_set.append(task_info)
                        self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                        task_set.remove(task_info)
                        self._returning_tasks[node_id] = task_set
        # remove all related tasks in to_generate_tasks
        for task_node_id in list(self._to_generate_task_infos.keys()):
            task_set = self._to_generate_task_infos.get(task_node_id, [])
            if task_node_id == to_remove_node_id:
                for task_info in task_set.copy():
                    removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                    removed_task_set.append(task_info)
                    self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                    task_set.remove(task_info)
                    self._to_generate_task_infos[task_node_id] = task_set
                del self._to_generate_task_infos[task_node_id]
            else:
                for task_info in task_set.copy():
                    if task_info.isRelatedToNode(to_remove_node_id):
                        removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                        removed_task_set.append(task_info)
                        self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                        task_set.remove(task_info)
                        self._to_generate_task_infos[task_node_id] = task_set
        # remove all related tasks in waiting_to_return_tasks
        for task_node_id in list(self._waiting_to_return_tasks.keys()):
            task_set = self._waiting_to_return_tasks.get(task_node_id, [])
            if task_node_id == to_remove_node_id:
                for task_info in task_set.copy():
                    removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                    removed_task_set.append(task_info)
                    self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                    task_set.remove(task_info)
                    self._waiting_to_return_tasks[task_node_id] = task_set
                del self._waiting_to_return_tasks[task_node_id]
            else:
                for task_info in task_set.copy():
                    if task_info.isRelatedToNode(to_remove_node_id):
                        removed_task_set = self._removed_tasks.get(task_info.getTaskNodeId(), [])
                        removed_task_set.append(task_info)
                        self._removed_tasks[task_info.getTaskNodeId()] = removed_task_set
                        task_set.remove(task_info)
                        self._waiting_to_return_tasks[task_node_id] = task_set


    def setTaskAttributeModel(self, attribute, model, **kwargs):
        """Set the task attribute model. The given model should be in the ATTRIBUTE_MODELS. The provided kwargs should be the parameters for the task attribute model.

        Args:
            attribute (str): The task attribute.
            model (str): The task attribute model.
            **kwargs: The parameters for the task attribute model.

        Examples:
            task_manager.setTaskAttributeModel('CPU', 'Uniform', low=0, high=1)
            task_manager.setTaskAttributeModel('Size', 'Normal', mean=0, std=1)
            task_manager.setTaskAttributeModel('Deadline', 'Uniform', low=0, high=1)
            task_manager.setTaskAttributeModel('Priority', 'Normal', mean=0, std=1)
        """
        if attribute == 'CPU':
            self._task_cpu_model = model
            if model == 'Uniform':
                self._task_cpu_low = kwargs.get('low', 0)
                self._task_cpu_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_cpu_mean = kwargs.get('mean', 0)
                self._task_cpu_std = kwargs.get('std', 1)
        elif attribute == 'Size':
            self._task_size_model = model
            if model == 'Uniform':
                self._task_size_low = kwargs.get('low', 0)
                self._task_size_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_size_mean = kwargs.get('mean', 0)
                self._task_size_std = kwargs.get('std', 1)
        elif attribute == 'Deadline':
            self._task_deadline_model = model
            if model == 'Uniform':
                self._task_deadline_low = kwargs.get('low', 0)
                self._task_deadline_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_deadline_mean = kwargs.get('mean', 0)
                self._task_deadline_std = kwargs.get('std', 1)
        elif attribute == 'Priority':
            self._task_priority_model = model
            if model == 'Uniform':
                self._task_priority_low = kwargs.get('low', 0)
                self._task_priority_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_priority_mean = kwargs.get('mean', 0)
                self._task_priority_std = kwargs.get('std', 1)
        elif attribute == 'ReturnSize':
            self._task_return_size_model = model
            if model == 'Uniform':
                self._task_return_size_low = kwargs.get('low', 0)
                self._task_return_size_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_return_size_mean = kwargs.get('mean', 0)
                self._task_return_size_std = kwargs.get('std', 1)

    def setTaskGenerationModel(self, task_generation_model, **kwargs):
        """Set the task generation model. The given task generation model should be in the SUPPORTED_TASK_GENERATION_MODELS. The provided kwargs should be the parameters for the task generation model (per second).

        Args:
            task_generation_model (str): The task generation model.
            **kwargs: The parameters for the task generation model.

        Examples:
            task_manager.setTaskGenerationModel('Poisson', lambda=1)
            task_manager.setTaskGenerationModel('Uniform', low=0, high=1)
            task_manager.setTaskGenerationModel('Normal', mean=0, std=1)
            task_manager.setTaskGenerationModel('Exponential', beta=1)
        """
        if task_generation_model == 'Poisson':
            self._task_generation_lambda = kwargs.get('lambda', 1)
        elif task_generation_model == 'Uniform':
            self._task_generation_low = kwargs.get('low', 0)
            self._task_generation_high = kwargs.get('high', 1)
        elif task_generation_model == 'Normal':
            self._task_generation_mean = kwargs.get('mean', 0)
            self._task_generation_std = kwargs.get('std', 1)
        elif task_generation_model == 'Exponential':
            self._task_generation_beta = kwargs.get('beta', 1)

    def _checkAttribute(self, attribute, attribute_name):
        min_value = self._config_task.get(f'task_min_{attribute_name.lower()}', None)
        max_value = self._config_task.get(f'task_max_{attribute_name.lower()}', None)
        if min_value is not None and attribute < min_value:
            attribute = min_value
        if max_value is not None and attribute > max_value:
            attribute = max_value
        return attribute

    def _generateCPU(self):
        cpu = 0
        if self._task_cpu_model == 'Uniform':
            cpu = np.random.uniform(self._task_cpu_low, self._task_cpu_high)
        elif self._task_cpu_model == 'Normal':
            cpu = np.random.normal(self._task_cpu_mean, self._task_cpu_std)
        cpu = self._checkAttribute(cpu, 'cpu')
        return cpu
        
    def _generateSize(self, size_type='offload'):
        assert size_type in ['offload', 'return'], 'The size type should be either offload or return.'
        size = 0
        if self._task_size_model == 'Uniform':
            if size_type == 'offload':
                size = np.random.uniform(self._task_size_low, self._task_size_high)
            elif size_type == 'return':
                size = np.random.uniform(self._task_return_size_low, self._task_return_size_high)
        elif self._task_size_model == 'Normal':
            if size_type == 'offload':
                size = np.random.normal(self._task_size_mean, self._task_size_std)
            elif size_type == 'return':
                size = np.random.normal(self._task_return_size_mean, self._task_return_size_std)
        if size_type == 'offload':
            size = self._checkAttribute(size, 'size')
        elif size_type == 'return':
            size = self._checkAttribute(size, 'returned_size')
        return size
        
    def _generateDeadline(self):
        deadline = 0
        if self._task_deadline_model == 'Uniform':
            deadline = np.random.uniform(self._task_deadline_low, self._task_deadline_high)
        elif self._task_deadline_model == 'Normal':
            deadline = np.random.normal(self._task_deadline_mean, self._task_deadline_std)
        deadline = self._checkAttribute(deadline, 'deadline')
        return deadline
        
    def _generatePriority(self):
        priority = 0
        if self._task_priority_model == 'Uniform':
            priority = np.random.uniform(self._task_priority_low, self._task_priority_high)
        elif self._task_priority_model == 'Normal':
            priority = np.random.normal(self._task_priority_mean, self._task_priority_std)
        priority = self._checkAttribute(priority, 'priority')
        return priority

    def _generateTaskInfo(self, task_node_id, arrival_time):
        self._task_id += 1
        task = Task(task_id = f'Task_{self._task_id}', task_node_id = task_node_id, task_cpu = self._generateCPU(),
                    task_size = self._generateSize('offload'), task_deadline = self._generateDeadline(),
                    task_priority = self._generatePriority(), task_arrival_time = arrival_time, required_returned_size=self._generateSize('return'))
        self._generated_task_history[task_node_id] = self._generated_task_history.get(task_node_id, [])
        self._generated_task_history[task_node_id].append(task)
        return task

    def generateTaskInfoOfMission(self,task_node_id,task_deadline,arrival_time,return_size, to_return_node_id=None, return_lazy_set=False):
        """Generate the tasks for mission by the node id.

        Args:
            task_node_id (str): The task node id.
            task_deadline (int): The task deadline (timeslot,same as mission deadline).
            arrival_time (int): The task arrive timeslot.

        Returns:
            Task: The task.
        """
        self._task_id += 1
        task = Task(task_id=f'Task_{self._task_id}', task_node_id=task_node_id, task_cpu=0, to_return_node_id=to_return_node_id,
                    task_size=0, task_deadline=task_deadline,task_priority=self._generatePriority(), return_lazy_set=True,
                    task_arrival_time=arrival_time,required_returned_size = return_size)
        task.setGenerated()
        task.setStartToTransmitTime(arrival_time)
        return task

    def _generateTasks(self, task_node_ids_kwardsDict, cur_time, simulation_interval):
        # 1. Move the tasks from the to_generate_task_infos to the todo_tasks according to the current time
        for task_node_id, task_infos in self._to_generate_task_infos.items():
            for task_info in task_infos.copy():
                if task_info.getTaskArrivalTime() <= cur_time: # if the task is arrived
                    todo_task_list = self._waiting_to_offload_tasks.get(task_node_id, [])
                    todo_task_list.append(task_info)
                    self._waiting_to_offload_tasks[task_node_id] = todo_task_list
                    self._to_generate_task_infos[task_node_id].remove(task_info)
                    task_info.setGenerated()

        # 2. Generate new to_generate_task_infos, oblige to the task generation model, simulation_interval, and predictable_seconds
        todo_task_num = 0
        for task_node_id, kwargsDict in task_node_ids_kwardsDict.items():
            to_genernate_task_infos = self._to_generate_task_infos.get(task_node_id, [])
            todo_task_num += len(to_genernate_task_infos)
            last_generation_time = cur_time if len(to_genernate_task_infos) == 0 else to_genernate_task_infos[-1].getTaskArrivalTime()
            # 每prediction_seconds生成一次任务
            if last_generation_time > cur_time // self._predictable_seconds * self._predictable_seconds:
                continue
            last_generation_time = max(last_generation_time, cur_time)
            last_generation_time += simulation_interval
            while last_generation_time <= cur_time + self._predictable_seconds:
                if self._task_generation_model == 'Poisson':
                    kwlambda = kwargsDict.get('lambda', self._task_generation_lambda)
                    task_num = np.random.poisson(kwlambda * simulation_interval)
                        
                elif self._task_generation_model == 'Uniform':
                    kwlow = kwargsDict.get('low', self._task_generation_low)
                    kwhigh = kwargsDict.get('high', self._task_generation_high)
                    task_num = np.random.randint(kwlow * simulation_interval, kwhigh * simulation_interval+1)
                    assert int(kwlow * simulation_interval) < int(kwhigh * simulation_interval), 'There is no task to generate.'

                elif self._task_generation_model == 'Normal':
                    kwmean = kwargsDict.get('mean', self._task_generation_mean)
                    kwstd = kwargsDict.get('std', self._task_generation_std)
                    task_num = np.random.normal(kwmean * simulation_interval, kwstd * simulation_interval)
                    assert kwmean * simulation_interval > 0, 'There is no task to generate.'
                    task_num = int(task_num)
                    task_num = task_num if task_num > 0 else 0

                elif self._task_generation_model == 'Exponential':
                    kwbeta = kwargsDict.get('beta', self._task_generation_beta)
                    task_num = np.random.exponential(kwbeta * simulation_interval)
                    assert kwbeta * simulation_interval > 0, 'There is no task to generate.'
                    task_num = int(task_num)

                elif self._task_generation_model == 'None':# 不生成任务
                    break 

                else:
                    raise NotImplementedError('The task generation model is not implemented.')

                if task_num == 0 and abs(cur_time + self._predictable_seconds - last_generation_time) < 1e-3:
                    task_num = 1 # 保证最后一个时间点至少有一个任务
                for i in range(int(task_num)):
                    task_info = self._generateTaskInfo(task_node_id, last_generation_time)
                    to_genernate_task_infos.append(task_info)
                    todo_task_num += 1
                last_generation_time += simulation_interval
            self._to_generate_task_infos[task_node_id] = to_genernate_task_infos
        return todo_task_num

    def generateAndCheckTasks(self, task_node_ids_kwardsDict, cur_time, simulation_interval):
        """Generate tasks and check the task status. This function should be called at each time step. It also moves the tasks to the failed tasks if the deadline is missed.

        Args:
            task_node_ids_kwardsDict (dict): The task node ids and the corresponding task generation parameters. If the parameters are not provided, the default parameters will be used.
            cur_time (float): The current simulation time.
            simulation_interval (float): The interval between two task generation operations.

        Returns:
            int: The number of tasks to be generated.

        Examples:
            task_manager.generateAndCheckTasks(['vehicle1', 'vehicle2'], 10.3, 1.5)
        """
        # 对每个task node当前to_generate_task，对to_offload_task或to_compute_task生成
        self._generateTaskDAG(task_node_ids_kwardsDict)
        todo_task_number = self._generateTasks(task_node_ids_kwardsDict, cur_time, simulation_interval)
        self.checkTasks(cur_time)
        return todo_task_number

    def _generateTaskDAG(self, task_node_ids_kwardsDict):
        # 保持每个task_node 至少有一个DAG，从而保持to_generate_task_infos中至少有一个task_info
        for task_node_id, kwargsDict in task_node_ids_kwardsDict.items():
            # self._task_dependencies={}
            task_dag = self._task_dependencies.get(task_node_id, None)
            # 从to_generate_task_infos中获取待生成的task；如果没有出现在现在的dag中，则新建一个dag，并且整合到当前的dag中
            task_id_in_dag = []
            if task_dag is not None:
                task_id_in_dag = list(task_dag.nodes())
            task_id_in_to_gen = [task_info.getTaskId() for task_info in self._to_generate_task_infos.get(task_node_id, [])]
            task_id_not_in_dag = [task_id for task_id in task_id_in_to_gen if task_id not in task_id_in_dag]

            dag_edge_prob = kwargsDict.get('dag_edge_prob', 0.3)
            # nx.DiGraph(), 生成一个有向图
            new_task_dag = generate_random_dag(task_id_not_in_dag, dag_edge_prob)
            merged_task_dag = merge_two_dags(task_dag, new_task_dag, dag_edge_prob)
            self._task_dependencies[task_node_id] = merged_task_dag

    def checkTasks(self, cur_time):
        # 1. Check the todo tasks
        to_offload_items = itertools.chain(self._waiting_to_offload_tasks.items(), self._offloading_tasks.items())
        for task_node_id, task_infos in to_offload_items:
            for task_info in task_infos.copy():
                if task_info.isExecutedLocally():
                    # 直接进入到下一个阶段, compute
                    if task_info.getCurrentNodeId() != task_info.getTaskNodeId():
                        task_info.transmit_to_Node(task_info.getTaskNodeId(), 1, task_info.getLastOperationTime(), fast_return=True)
                    self.addToComputeTask(task_info, task_info.getTaskNodeId(), cur_time)
                    task_infos.remove(task_info)
        # 2. Check the return tasks
        to_return_items = itertools.chain(self._returning_tasks.items(), self._waiting_to_return_tasks.items())
        for node_id, task_infos in to_return_items:
            for task_info in task_infos.copy():
                if not task_info.requireReturn():
                    # 直接跳到下一个阶段
                    task_node_id = task_info.getTaskNodeId()
                    task_info.transmit_to_Node(task_info.getToReturnNodeId(), 1, cur_time, fast_return=True)
                    if task_info.task_delay <= task_info.task_deadline + 1e-5:
                        self._done_tasks[task_node_id] = self._done_tasks.get(task_node_id, [])
                        self._done_tasks[task_node_id].append(task_info)
                        self._recently_done_100_tasks.append(task_info)
                    else:
                        self._out_of_ddl_tasks[task_node_id] = self._out_of_ddl_tasks.get(task_node_id, [])
                        self._out_of_ddl_tasks[task_node_id].append(task_info)
                        self._recently_failed_100_tasks.append(task_info)
                    task_infos.remove(task_info)
        # 3. Check the offloading or returning tasks, if the transmission time is out of TTI, then move the task to the failed tasks
        transmitting_tasks = itertools.chain(self._offloading_tasks.items(), self._returning_tasks.items())
        for node_id, task_infos in transmitting_tasks:
            for task_info in task_infos.copy():
                last_transmission_time = task_info.getLastTransmissionTime()
                if cur_time - last_transmission_time > self._config_task.get('tti_threshold', 0.5):
                    self.failOffloadingTask(task_info, failure = EnumerateConstants.TASK_FAIL_OUT_OF_TTI)

        # 4. Out of Hard DDL (self._config_task.get('hard_ddl', 2))
        all_tasks = itertools.chain(self._waiting_to_offload_tasks.items(), self._offloading_tasks.items(), self._computing_tasks.items(), self._waiting_to_return_tasks.items(), self._returning_tasks.items(), self._to_generate_task_infos.items())
        for node_id, task_infos in all_tasks:
            for task_info in task_infos.copy():
                task_node_id = task_info.getTaskNodeId()
                task_id = task_info.getTaskId()
                if cur_time - task_info.getTaskArrivalTime() > task_info.getTaskDeadline():
                # self._config_task.get('hard_ddl', 2):
                    if task_info in self._to_generate_task_infos.get(task_info.getTaskNodeId(), []):
                        pass
                    task_info.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_DDL)
                    task_infos.remove(task_info)
                    self._out_of_ddl_tasks[task_node_id] = self._out_of_ddl_tasks.get(task_node_id, [])
                    self._out_of_ddl_tasks[task_node_id].append(task_info)
                    self._recently_failed_100_tasks.append(task_info)
                else:
                    flag = self.checkTaskDependency(task_node_id, task_id)
                    if flag is not None:
                        continue
                    failed_task_list = self._out_of_ddl_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._out_of_ddl_tasks[task_node_id] = failed_task_list
                    task_infos.remove(task_info)
                    task_info.setTaskFailueCode(EnumerateConstants.TASK_FAIL_PARENT_FAILED)

        # 5. Check the task DAG
        for task_node_id, task_dag in self._task_dependencies.items():
            # 如果对于task_dag中后继为0的task，已经完成/失败了，则将其以及其前序task移除
            not_finished_tasks = self._getNotFinishedTasksByNode(task_node_id)
            remove_flag = True
            removed_nodes = []
            while remove_flag:
                remove_flag = False
                for task_id in list(task_dag.nodes()).copy():
                    if task_dag.out_degree(task_id) == 0 and task_id not in not_finished_tasks:
                        removed_nodes.append({task_id: task_dag.predecessors(task_id)})
                        task_dag.remove_node(task_id)
                        remove_flag = True
            if len(removed_nodes) > 1:
                a=3 #debug
                
    def _getNotFinishedTasksByNode(self, task_node_id):
        not_finished_tasks = []
        for task_info in self._to_generate_task_infos.get(task_node_id, []):
            not_finished_tasks.append(task_info.getTaskId())
        for task_info in self._waiting_to_offload_tasks.get(task_node_id, []):
            not_finished_tasks.append(task_info.getTaskId())
        for task_info in self._offloading_tasks.get(task_node_id, []):
            not_finished_tasks.append(task_info.getTaskId())

        iter_chain = itertools.chain(self._computing_tasks.items(), self._waiting_to_return_tasks.items(), self._returning_tasks.items())
        for node_id, task_infos in iter_chain:
            for task_info in task_infos:
                if task_info.getTaskNodeId() == task_node_id:
                    not_finished_tasks.append(task_info.getTaskId())
        return not_finished_tasks

    def offloadTask(self, task_node_id, task_id, target_node_id, current_time, route = None):
        """Offload the task by the task id and the target node id.

        Args:
            task_node_id (str): The task node id.
            task_id (str): The task id.
            target_node_id (str): The target node id.
            current_time (float): The current simulation time
            route (list, optional): The route for the task offloading. Default [target_node_id]

        Returns:
            bool: True if the task is offloaded successfully, False otherwise.

        Examples:
            task_manager.offloadTask('vehicle1', 'Task_1', 'fog1')
        """
        if route is None:
            route = [target_node_id]
        assert route[-1] == target_node_id, 'The last node of the route should be the target node id.'
        if task_node_id in self._waiting_to_offload_tasks:
            for task_info in self._waiting_to_offload_tasks[task_node_id].copy():
                flag = self.checkTaskDependency(task_node_id, task_id)
                if task_info.getTaskId() == task_id and flag == True:
                    task_info.offloadTo(target_node_id, route, current_time)
                    self._offloading_tasks[task_node_id] = self._offloading_tasks.get(task_node_id, [])
                    self._offloading_tasks[task_node_id].append(task_info)
                    self._waiting_to_offload_tasks[task_node_id].remove(task_info)
                    assert len(task_info.getToOffloadRoute())>0
                    # if execute locally
                    if target_node_id == task_node_id and task_info.getCurrentNodeId() == task_node_id:
                        self.addToComputeTask(task_info, task_node_id, current_time)
                        self._offloading_tasks[task_node_id].remove(task_info)
                    return True
                elif flag is None: # None表示有parent task失败
                    failed_task_list = self._out_of_ddl_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._out_of_ddl_tasks[task_node_id] = failed_task_list
                    self._waiting_to_offload_tasks[task_node_id].remove(task_info)
                    task_info.setTaskFailueCode(EnumerateConstants.TASK_FAIL_PARENT_FAILED)
        return False
    
    def checkTaskDependency(self, task_node_id, task_id):
        """Check the task dependency by the task node id and the task id.

        Args:
            task_node_id (str): The task node id.
            task_id (str): The task id.

        Returns:
            bool: True if the task is generated, False if the task is not generated, None if the parent tasks are failed.

        """
        done_task_id_for_node = [task_info.getTaskId() for task_info in self._done_tasks.get(task_node_id, [])]
        failed_task_id_for_node = [task_info.getTaskId() for task_info in self._out_of_ddl_tasks.get(task_node_id, [])]
        task_dag = self._task_dependencies[task_node_id] # nx.DiGraph
        parents = []
        if task_dag is not None and task_id in task_dag:
            # if the parent tasks are done, then the task is generated
            parents = list(task_dag.predecessors(task_id))
        if all([task_id in done_task_id_for_node for task_id in parents]) or len(parents) == 0:
            return True
        # if the parent tasks failed, then the task is failed
        elif any([task_id in failed_task_id_for_node for task_id in parents]) and len(parents) > 0:
            return None
        return False
    
    def getToOffloadTasks(self, task_node_id):
        """Get the tasks to offload by the task node id.

        Args:
            task_node_id (str): The task node id.

        Returns:
            list: The list of the tasks to offload.

        Examples:
            task_manager.getToOffloadTasks('vehicle1')
        """
        return self._waiting_to_offload_tasks.get(task_node_id, [])

    def getWaitingToReturnTaskInfos(self):
        """Get waiting to return task infos.

        Args:

        Returns:
            dict: node_id -> {task:Task,...}

        Examples:
            task_manager.getWaitingToReturnTaskInfos()
        """
        return self._waiting_to_return_tasks

    def setTaskReturnRouteAndStartReturn(self,task_id,route,current_time):
        to_remove_tasks={}
        for node_id,task_infos in self._waiting_to_return_tasks.items():
            for task_info in task_infos:
                if task_info.getTaskId()==task_id:
                    task_info.setToReturnRoute(route)
                    self._returning_tasks[node_id]=self._returning_tasks.get(node_id,[])
                    self._returning_tasks[node_id].append(task_info)
                    to_remove_tasks[node_id]=to_remove_tasks.get(node_id,[])
                    to_remove_tasks[node_id].append(task_info)
                    task_info.startToReturn(current_time)

        for node_id,task_infos in to_remove_tasks.items():
            for task_info in task_infos:
                self._waiting_to_return_tasks[node_id].remove(task_info)

    def getAllTasks(self):
        """Get all tasks.

        Args:

        Returns:
            list: The list of all tasks.

        Examples:
            task_manager.getAllTasks()
        """
        all_tasks = []
        for task_node_id, task_infos in self._waiting_to_offload_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._offloading_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._computing_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._waiting_to_return_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._returning_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._done_tasks.items():
            all_tasks.extend(task_infos)
        for task_node_id, task_infos in self._out_of_ddl_tasks.items():
            all_tasks.extend(task_infos)
        return all_tasks
    
    