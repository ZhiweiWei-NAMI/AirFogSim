import numpy as np
from ..entities.task import Task
from ..enum_const import EnumerateConstants
class TaskManager:
    """ Task Manager is responsible for generating tasks and managing the task status. 
    """
    SUPPORTED_TASK_GENERATION_MODELS = ['Poisson', 'Uniform', 'Normal', 'Exponential']
    ATTRIBUTE_MODELS = ['Uniform', 'Normal']
    def __init__(self, task_generation_model = 'Poisson', predictable_seconds = 5, cpu_model = 'Uniform', size_model = 'Uniform', deadline_model = 'Uniform', priority_model = 'Uniform', **kwargs):
        self._task_generation_model = task_generation_model
        assert task_generation_model in TaskManager.SUPPORTED_TASK_GENERATION_MODELS, 'The task generation model is not supported. Only support {}'.format(TaskManager.SUPPORTED_TASK_GENERATION_MODELS)
        self._to_generate_task_infos = {} # use Node Id as the key, and the value is the task info list
        self._to_offload_tasks = {}
        self._to_compute_tasks = {}
        self._to_return_tasks = {}
        self._done_tasks = {}
        self._failed_tasks = {}
        self._removed_tasks = {}
        self._task_id = 0
        self._predictable_seconds = predictable_seconds # the seconds that the task generation can be predicted
        self.setTaskGenerationModel(task_generation_model, **kwargs)
        self.setTaskAttributeModel('CPU', cpu_model, **kwargs)
        self.setTaskAttributeModel('Size', size_model, **kwargs)
        self.setTaskAttributeModel('Deadline', deadline_model, **kwargs)
        self.setTaskAttributeModel('Priority', priority_model, **kwargs)

    def addToComputeTask(self, task:Task, node_id, current_time):
        """Add the task to the to_compute_tasks.

        Args:
            task (Task): The task to add.
            node_id (str): The assigned node id (namely, the node id that computes the task).

        Examples:
            task_manager.addToComputeTask(task, 'vehicle1', 10.3)
        """
        to_compute_task_list = self._to_compute_tasks.get(node_id, [])
        to_compute_task_list.append(task)
        task.startToCompute(current_time)
        self._to_compute_tasks[node_id] = to_compute_task_list

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
        if not task_info.isReturning(): # if the task is offloading
            for task_info in self._to_offload_tasks[task_node_id]:
                if task_info.getTaskId() == task_id:
                    task_info.startToCompute(current_time)
                    self._to_offload_tasks[task_node_id].remove(task_info)
                    to_compute_task_list = self._to_compute_tasks.get(node_id, [])
                    to_compute_task_list.append(task_info)
                    self._to_compute_tasks[node_id] = to_compute_task_list
                    return True
        elif task_info.isReturning(): # if the task is returning
            for task_info in self._to_return_tasks[node_id]:
                if task_info.getTaskId() == task_id:
                    self._to_return_tasks[node_id].remove(task_info)
                    self._done_tasks[task_node_id] = self._done_tasks.get(task_node_id, [])
                    self._done_tasks[task_node_id].append(task_info)
                    return True
        return False
    
    def failOffloadingTask(self, task:Task):
        """Remove the offloading task by the task id, and then move the task to the failed_tasks.

        Args:
            task (Task): The task which is failed offloading.

        Returns:
            bool: True if the task is executed successfully, False otherwise.
        """
        node_id = task.getAssignedTo()
        task_node_id = task.getTaskNodeId()
        task_id = task.getTaskId()
        if not task.isReturning():
            for task_info in self._to_offload_tasks[task_node_id]:
                if task_info.getTaskId() == task_id:
                    self._to_offload_tasks[task_node_id].remove(task_info)
                    failed_task_list = self._failed_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._failed_tasks[task_node_id] = failed_task_list
                    return True
        elif task.isReturning():
            for task_info in self._to_return_tasks[node_id]:
                if task_info.getTaskId() == task_id:
                    self._to_return_tasks[node_id].remove(task_info)
                    failed_task_list = self._failed_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._failed_tasks[task_node_id] = failed_task_list
                    return True
        return False

    def computeTasks(self, allocated_cpu_by_taskId, simulation_interval, current_time):
        """Compute the tasks by the allocated CPU.

        Args:
            allocated_cpu_by_taskId (dict): The allocated CPU by the task id.
            simulation_interval (float): The simulation interval.
            current_time (float): The current time.
            _getNodeById (function): The function to get the node by ID.
        """
        for node_id, task_infos in self._to_compute_tasks.items():
            for task_info in task_infos.copy(): # task_info is Task
                task_id = task_info.getTaskId()
                allocated_cpu = allocated_cpu_by_taskId.get(task_id, 0)
                task_info.compute(allocated_cpu, simulation_interval, current_time)
                if task_info.isComputed():
                    self._to_compute_tasks[node_id].remove(task_info)
                    task_info.startToReturn(current_time)
                    
                    if task_info.requireReturn():
                        self._to_return_tasks[node_id] = self._to_return_tasks.get(node_id, [])
                        self._to_return_tasks[node_id].append(task_info)
                    else:
                        task_node_id = task_info.getTaskNodeId()
                        self._done_tasks[task_node_id] = self._done_tasks.get(task_node_id, [])
                        self._done_tasks[task_node_id].append(task_info)


    def getOffloadingTasks(self):
        """Get the offloading tasks (transmission).

        Returns:
            dict: The offloading tasks. The key is the node id, and the value is the task list.
            int: The total number of offloading tasks.

        Examples:
            offloading_tasks, total_num = task_manager.getOffloadingTasks()
        """
        # 遍历task，if isTransmitting() == True, 则加入到offloading_tasks中
        offloading_tasks = {} # key: transmitter node id, value: list of tasks
        total_num = 0
        for task_node_id, task_infos in self._to_offload_tasks.items():
            offloading_tasks[task_node_id] = [task_info for task_info in task_infos if task_info.isTransmitting()]
            total_num += len(offloading_tasks[task_node_id])
        for node_id, task_infos in self._to_return_tasks.items():
            to_offload_task = offloading_tasks.get(node_id, [])
            to_offload_task.extend([task_info for task_info in task_infos if task_info.isTransmitting()])
            offloading_tasks[node_id] = to_offload_task
            total_num += len(offloading_tasks[node_id])
        return offloading_tasks, total_num

    def removeTasksByNodeId(self, node_id):
        """Remove the tasks by the node id.

        Args:
            node_id (str): The node id.

        Examples:
            task_manager.removeTasksByNodeId('vehicle1')
        """
        # remove as fog node
        if node_id in self._to_compute_tasks:
            to_compute_task_list = self._to_compute_tasks[node_id]
            for task_info in to_compute_task_list:
                self._removed_tasks[task_info.getTaskNodeId()] = task_info
            del self._to_compute_tasks[node_id]
        # remove as task node
        if node_id in self._to_offload_tasks:
            to_offload_task_list = self._to_offload_tasks[node_id]
            for task_info in to_offload_task_list:
                self._removed_tasks[task_info.getTaskNodeId()] = task_info
            del self._to_offload_tasks[node_id]
        # remove as fog node
        if node_id in self._to_return_tasks:
            to_return_task_list = self._to_return_tasks[node_id]
            for task_info in to_return_task_list:
                self._removed_tasks[task_info.getTaskNodeId()] = task_info
            del self._to_return_tasks[node_id]

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
            if model == 'Uniform':
                self._task_cpu_low = kwargs.get('low', 0)
                self._task_cpu_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_cpu_mean = kwargs.get('mean', 0)
                self._task_cpu_std = kwargs.get('std', 1)
        elif attribute == 'Size':
            if model == 'Uniform':
                self._task_size_low = kwargs.get('low', 0)
                self._task_size_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_size_mean = kwargs.get('mean', 0)
                self._task_size_std = kwargs.get('std', 1)
        elif attribute == 'Deadline':
            if model == 'Uniform':
                self._task_deadline_low = kwargs.get('low', 0)
                self._task_deadline_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_deadline_mean = kwargs.get('mean', 0)
                self._task_deadline_std = kwargs.get('std', 1)
        elif attribute == 'Priority':
            if model == 'Uniform':
                self._task_priority_low = kwargs.get('low', 0)
                self._task_priority_high = kwargs.get('high', 1)
            elif model == 'Normal':
                self._task_priority_mean = kwargs.get('mean', 0)
                self._task_priority_std = kwargs.get('std', 1)

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


    def _generateCPU(self):
        if self._task_cpu_model == 'Uniform':
            return np.random.uniform(self._task_cpu_low, self._task_cpu_high)
        elif self._task_cpu_model == 'Normal':
            return np.random.normal(self._task_cpu_mean, self._task_cpu_std)
        
    def _generateSize(self):
        if self._task_size_model == 'Uniform':
            return np.random.uniform(self._task_size_low, self._task_size_high)
        elif self._task_size_model == 'Normal':
            return np.random.normal(self._task_size_mean, self._task_size_std)
        
    def _generateDeadline(self):
        if self._task_deadline_model == 'Uniform':
            return np.random.uniform(self._task_deadline_low, self._task_deadline_high)
        elif self._task_deadline_model == 'Normal':
            return np.random.normal(self._task_deadline_mean, self._task_deadline_std)
        
    def _generatePriority(self):
        if self._task_priority_model == 'Uniform':
            return np.random.uniform(self._task_priority_low, self._task_priority_high)
        elif self._task_priority_model == 'Normal':
            return np.random.normal(self._task_priority_mean, self._task_priority_std)

    def _generateTaskInfo(self, task_node_id, arrival_time):
        self._task_id += 1
        return Task(task_id = 'Task_'+self._task_id, task_node_id = task_node_id, task_cpu = self._generateCPU(), task_size = self._generateSize(), task_deadline = self._generateDeadline(), task_priority = self._generatePriority(), task_arrival_time = arrival_time)
    
    def _generateTasks(self, task_node_ids_kwardsDict, cur_time, simulation_interval):
        # 1. Move the tasks from the to_generate_task_infos to the todo_tasks according to the current time
        for task_node_id, task_infos in self._to_generate_task_infos.items():
            for task_info in task_infos.copy():
                if task_info.getTaskArrivalTime() >= cur_time: # if the task is arrived, i.e., the task arrival time is greater than the current time
                    todo_task_list = self._to_offload_tasks.get(task_node_id, [])
                    todo_task_list.append(task_info)
                    self._to_offload_tasks[task_node_id] = todo_task_list
                    self._to_generate_task_infos[task_node_id].remove(task_info)

        # 2. Generate new to_generate_task_infos, oblige to the task generation model, simulation_interval, and predictable_seconds
        todo_task_num = 0
        for task_node_id, kwargsDict in task_node_ids_kwardsDict.items():
            to_genernate_task_infos = self._to_generate_task_infos.get(task_node_id, [])
            todo_task_num += len(to_genernate_task_infos)
            last_generation_time = cur_time if len(to_genernate_task_infos) == 0 else to_genernate_task_infos[-1].getTaskArrivalTime()
            last_generation_time += simulation_interval
            while last_generation_time <= cur_time + self._predictable_seconds:
                if self._task_generation_model == 'Poisson':
                    kwlambda = kwargsDict.get('lambda', self._task_generation_lambda)
                    task_num = np.random.poisson(kwlambda * simulation_interval)
                    for i in range(task_num):
                        task_info = self._generateTaskInfo(task_node_id, last_generation_time) # Task()
                        to_genernate_task_infos.append(task_info)
                        todo_task_num += 1
                elif self._task_generation_model == 'Uniform':
                    kwlow = kwargsDict.get('low', self._task_generation_low)
                    kwhigh = kwargsDict.get('high', self._task_generation_high)
                    task_num = np.random.randint(kwlow * simulation_interval, kwhigh * simulation_interval+1)
                    assert int(kwlow * simulation_interval) < int(kwhigh * simulation_interval), 'There is no task to generate.'
                    for i in range(task_num):
                        task_info = self._generateTaskInfo(task_node_id, last_generation_time)
                        to_genernate_task_infos.append(task_info)
                        todo_task_num += 1

                elif self._task_generation_model == 'Normal':
                    kwmean = kwargsDict.get('mean', self._task_generation_mean)
                    kwstd = kwargsDict.get('std', self._task_generation_std)
                    task_num = np.random.normal(kwmean * simulation_interval, kwstd * simulation_interval)
                    assert kwmean * simulation_interval > 0, 'There is no task to generate.'
                    task_num = int(task_num)
                    task_num = task_num if task_num > 0 else 0
                    for i in range(int(task_num)):
                        task_info = self._generateTaskInfo(task_node_id, last_generation_time)
                        to_genernate_task_infos.append(task_info)
                        todo_task_num += 1
                elif self._task_generation_model == 'Exponential':
                    kwbeta = kwargsDict.get('beta', self._task_generation_beta)
                    task_num = np.random.exponential(kwbeta * simulation_interval)
                    assert kwbeta * simulation_interval > 0, 'There is no task to generate.'
                    task_num = int(task_num)
                    for i in range(int(task_num)):
                        task_info = self._generateTaskInfo(task_node_id, last_generation_time)
                        to_genernate_task_infos.append(task_info)
                        todo_task_num += 1
                else:
                    raise NotImplementedError('The task generation model is not implemented.')
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
        todo_task_number = self._generateTasks(task_node_ids_kwardsDict, cur_time, simulation_interval)
        self._checkTasks(cur_time)
        return todo_task_number

    def _checkTasks(self, cur_time):
        # 1. Check the todo tasks
        for task_node_id, task_infos in self._to_offload_tasks.items():
            for task_info in task_infos.copy():
                if task_info.getTaskDeadline() + task_info.getTaskArrivalTime() <= cur_time:
                    task_info.setTaskFailueCode(EnumerateConstants.TASK_FAIL_OUT_OF_DDL)
                    failed_task_list = self._failed_tasks.get(task_node_id, [])
                    failed_task_list.append(task_info)
                    self._failed_tasks[task_node_id] = failed_task_list
                    self._to_offload_tasks[task_node_id].remove(task_info)