class Task:
    """ Task is the class that represents the task. 
    """
    def __init__(self, task_id, task_node_id, task_cpu, task_size, task_deadline, task_priority, task_arrival_time):
        """The constructor of the Task class.

        Args:
            task_id (str): The unique ID of the task.
            task_node_id (str): The ID of the task node.
            task_cpu (float): The required CPU of the task.
            task_size (float): The size of the task.
            task_deadline (float): The deadline of the task.
            task_priority (float): The value of the task.
            task_arrival_time (float): The start time of the task.
        """
        self._task_id = task_id
        self._task_node_id = task_node_id
        self._task_cpu = task_cpu
        self._task_size = task_size
        self._task_deadline = task_deadline
        self._task_priority = task_priority
        self._task_arrival_time = task_arrival_time
        self._executed_locally = True
        self._assigned_to = None
        self._routes = [task_node_id]
        self._routed_time = [task_arrival_time] # the time that the task is routed to the node
        self._start_to_transmit_time = 0
        self._last_transmission_time = 0
        self._transmitted_size = 0
        self._start_to_compute_time = 0
        self._computed_size = 0
        self._last_compute_time = 0
        self._failure_reason = None

    def setTaskFailueReason(self, reason):
        """Set the task failure reason.

        Args:
            reason (str): The reason of the task failure.
        """
        self._failure_reason = reason

    def getTaskFailureReason(self):
        """Get the task failure reason.

        Returns:
            str: The reason of the task failure.
        """
        return self._failure_reason
    
    def isFinished(self):
        """Check if the task is finished.

        Returns:
            bool: True if the task is finished, False otherwise.
        """
        return self._computed_size >= self._task_size
    
    def getTransmittedSize(self):
        """Get the transmitted size.

        Returns:
            float: The transmitted size.
        """
        return self._transmitted_size
    
    def getComputedSize(self):
        """Get the computed size.

        Returns:
            float: The computed size.
        """
        return self._computed_size
    
    def getComputedRatio(self):
        """Get the computed ratio.

        Returns:
            float: The computed ratio.
        """
        return self._computed_size / self._task_cpu
    
    def getTransmittedRatio(self):
        """Get the transmitted ratio.

        Returns:
            float: The transmitted ratio.
        """
        return self._transmitted_size / self._task_size
    
    def getRoutedNodeIdsWithTime(self):
        """Get the routed node IDs with time.

        Returns:
            list: The routed node IDs with time.
        """
        return self._routes, self._routed_time
    
    def getTaskCPU(self):
        """Get the required task CPU.

        Returns:
            float: The task CPU.
        """
        return self._task_cpu
    def getLastTransmissionTime(self):
        """Get the last transmission time if the task is transmitted.
        """
        return self._last_transmission_time
    
    def getLastComputeTime(self):
        """Get the last compute time.
        """
        return self._last_compute_time
        
    def getTaskArrivalTime(self):
        """Get the task arrival time.

        Returns:
            float: The task arrival time.
        """
        return self._task_arrival_time
    
    def getTaskPriority(self):
        """Get the task priority.

        Returns:
            float: The task priority.
        """
        return self._task_priority

    def getTaskId(self):
        """Get the task ID.

        Returns:
            str: The task ID.
        """
        return self._task_id

    def getTaskNodeId(self):
        """Get the task node ID.

        Returns:
            str: The task node ID.
        """
        return self._task_node_id

    def getTaskSize(self):
        """Get the task size.

        Returns:
            float: The task size.
        """
        return self._task_size

    def getTaskDeadline(self):
        """Get the task deadline.

        Returns:
            float: The task deadline.
        """
        return self._task_deadline