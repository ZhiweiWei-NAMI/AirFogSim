from ..enum_const import EnumerateConstants
from .mission import Mission
class Task:
    """ Task is the class that represents the task. 
    """
    def __init__(self, task_id, task_node_id, task_cpu, task_size, task_deadline, task_priority, task_arrival_time, farther_mission:Mission = None, required_returned_size = 0):
        """The constructor of the Task class.

        Args:
            task_id (str): The unique ID of the task.
            task_node_id (str): The ID of the task node.
            task_cpu (float): The required CPU of the task.
            task_size (float): The size of the task.
            task_deadline (float): The deadline of the task.
            task_priority (float): The value of the task.
            task_arrival_time (float): The start time of the task.
            farther_mission (Mission): The farther mission of the task.
            required_returned_size (float): The required returned size of the task.
        """
        self._task_id = task_id
        self._task_node_id = task_node_id
        self._task_cpu = task_cpu
        self._task_size = task_size
        self._required_returned_size = required_returned_size
        self._task_deadline = task_deadline
        self._task_priority = task_priority
        self._task_arrival_time = task_arrival_time
        self._executed_locally = False
        self._assigned_to = None
        self._decided_route = [] # the decided route for offloading
        self._routes = [task_node_id] # arrived node list
        self._to_offload_route = [] # the route to offload the task. If task is computed, i.e., _compute_size >= _task_cpu, the list denotes the route the returned data should be transmitted.
        self._decided_offload_time = [] # the decided offload time
        self._routed_time = [task_arrival_time] # the time that the task is routed to the node
        self._start_to_transmit_time = 0
        self._last_transmission_time = 0
        self._transmitted_size = 0
        self._start_to_compute_time = 0
        self._start_to_return_time = 0
        self._last_return_time = 0
        self._computed_size = 0
        self._last_compute_time = 0
        self._failure_reason_code = -1
        self._farther_mission = farther_mission

    def to_dict(self):
        """Convert the task to dictionary.

        Returns:
            dict: The dictionary of the task.
        """
        # 遍历所有属性，将其转化为字典
        task_dict = {}
        for key, value in self.__dict__.items():
            key = key[1:]
            if key == "farther_mission":
                task_dict[key] = value.to_dict()
            else:
                task_dict[key] = value
        return task_dict

    def startToCompute(self, time):
        """Set the start time to compute the task and set to_offload_route as to_return_route.

        Args:
            time (float): The start time to compute the task.
        """
        self._start_to_compute_time = time
        self._last_compute_time = time

    def startToReturn(self, current_time):
        """Set the start time to return the task.

        Args:
            current_time (float): The current time.
        """
        self._start_to_return_time = current_time
        self._last_return_time = current_time
        if self._required_returned_size > 0:
            self._to_offload_route = [self._task_node_id] # the route to offload the task. If task is computed, i.e., _compute_size >= _task_cpu, the list denotes the route the returned data should be transmitted.
        else:
            self._to_offload_route = []

    def requireReturn(self):
        """Check if the task requires return.

        Returns:
            bool: True if the task requires return, False otherwise.
        """
        assert self.isComputed(), "The task should be computed before returning."
        if self._assigned_to != self._task_node_id and self._required_returned_size > 0:
            assert len(self._to_offload_route) > 0, "The route to offload the task should be set."
            return True
        return False

    def compute(self, allocated_cpu, simulation_interval, current_time):
        """Compute the task.

        Args:
            allocated_cpu (float): The allocated CPU.
            simulation_interval (float): The simulation interval.
            current_time (float): The current time.
        """
        assert current_time >= self._start_to_compute_time, "The current time should be greater than the start time to compute the task."
        assert self._assigned_to == self.getCurrentNodeId(), "The task should be computed at the assigned node."
        self._computed_size += allocated_cpu * simulation_interval
        self._last_compute_time = current_time
        if self._computed_size >= self._task_cpu:
            self._computed_size = self._task_cpu

    def getReturnedSize(self):
        """Get the returned size.

        Returns:
            float: The required returned size.
        """
        return self._required_returned_size
    
    def isComputed(self):
        """Check if the task is computed.

        Returns:
            bool: True if the task is computed, False otherwise.
        """
        return self._computed_size >= self._task_cpu

    def setFartherMission(self, farther_mission:Mission):
        """Set the farther mission.

        Args:
            farther_mission (Mission): The farther mission.
        """
        self._farther_mission = farther_mission

    def wait_to_ddl(self, current_time):
        """Check if the task is out of deadline.

        Args:
            current_time (float): The current time.

        Returns:
            bool: True if the task is out of deadline, False otherwise.
        """
        return self._task_arrival_time + self._task_deadline <= current_time
    
    def isReturning(self):
        """Check if the task is returning or offloading.

        Returns:
            bool: True if the task is returning, False otherwise.
        """
        return self.isComputed() and self.isTransmitting()
    
    def transmit_to_Node(self, node_id, trans_data, current_time):
        """Transmit the data to the node. Possible to return or offload the task. 

        Args:
            node_id (str): The ID of the node.

        Returns:
            bool: True if the task is transmitted, False if requires more transmission.
        """
        assert self.isTransmitting(), "The task should be transmitting."
        isReturning = self.isReturning()
        self._transmitted_size += trans_data
        if isReturning:
            self._last_return_time = current_time
            require_transmit_size = self._required_returned_size
        else:
            self._last_transmission_time = current_time
            require_transmit_size = self._task_size
        if self._transmitted_size >= require_transmit_size:
            self._transmitted_size = 0
            self._routes.append(node_id)
            self._routed_time.append(current_time)
            del self._to_offload_route[0] # remove the first element
            return True
        return False

    def offloadTo(self, node_id, route, time):
        """Offload the task to the node. If node_id is the same as the task node ID, the task is executed locally.

        Args:
            node_id (str): The ID of the node.
            route (list): The route to the node. Each element is the node ID.
            time (float): The time to offload the task.

        Examples:
            task.offloadTo('node1', ['node2', 'node1'], 10)
        """
        self._decided_route = [self._task_node_id] + route # the decided route for offloading, starting from the task node
        self._to_offload_route = route
        assert route[-1] == node_id, "The last node in the route should be the node ID."
        self._assigned_to = node_id
        self._decided_offload_time = time
        self._executed_locally = node_id == self._task_node_id
        self._start_to_transmit_time = time
        self._last_transmission_time = time

    def isExecutedLocally(self):
        """Check if the task is executed locally.

        Returns:
            bool: True if the task is executed locally, False otherwise.
        """
        return self._executed_locally
    
    def getToOffloadRoute(self):
        """Get the route to offload the task.

        Returns:
            list: The route to offload the task.
        """
        return self._to_offload_route
    
    def getDecidedOffloadTime(self):
        """Get the decided offload time.

        Returns:
            float: The decided offload time.
        """
        return self._decided_offload_time
    
    def getAssignedTo(self):
        """Get the assigned node ID.

        Returns:
            str: The assigned node ID.
        """
        return self._assigned_to
    
    def setAssignedTo(self, node_id):
        """Set the assigned node ID.

        Args:
            node_id (str): The node ID.
        """
        self._assigned_to = node_id

    def isTransmitting(self):
        """Check if the task is transmitting.

        Returns:
            bool: True if the task is transmitting, False otherwise.
        """
        require_transmit_size = self._task_size if not self.isComputed() else self._required_returned_size
        return self._transmitted_size < require_transmit_size and not self._executed_locally
    
    def isTransmittedToAssignedNode(self):
        """Check if the task is transmitted to the assigned node.

        Returns:
            bool: True if the task is transmitted, False otherwise.
        """
        return self.getCurrentNodeId() == self._assigned_to

    def setTaskFailueCode(self, code):
        """Set the task failure code

        Args:
            code (int): The reason of the task failure. The code is in EnumerateConstants class.
        """
        self._failure_reason_code = code

    def getTaskFailureReason(self):
        """Get the task failure reason.

        Returns:
            str: The reason of the task failure.
        """
        return EnumerateConstants.getDescByCode(self._failure_reason_code)
    
    def isFinished(self):
        """Check if the task is finished.

        Returns:
            bool: True if the task is finished, False otherwise.
        """
        return self.isComputed() and len(self._to_offload_route) == 0
    
    def getDelay(self):
        """Get the delay of the task.

        Returns:
            float: The delay of the task.
        """
        return self._last_return_time - self._task_arrival_time

    def getLastOperationTime(self):
        """Get the last operation time.

        Returns:
            float: The last operation time.
        """
        last_time = max(self._last_transmission_time, self._last_compute_time, self._last_return_time)
        return last_time
    
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
    
    def getCurrentNodeId(self):
        """Get the current node ID.

        Returns:
            str: The current node ID.
        """
        return self._routes[-1]
    
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