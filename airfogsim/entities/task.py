from ..enum_const import EnumerateConstants
from .mission import Mission
class Task:
    """ Task is the class that represents the task. 
    """
    def __init__(self, task_id, task_node_id, task_cpu, task_size, task_deadline, task_priority, task_arrival_time, farther_mission:Mission = None):
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
        self._decided_route = None # the decided route for offloading
        self._routes = [task_node_id] # arrived node list
        self._to_offload_route = None # the route to offload the task
        self._decided_offload_time = None # the decided offload time
        self._routed_time = [task_arrival_time] # the time that the task is routed to the node
        self._start_to_transmit_time = 0
        self._last_transmission_time = 0
        self._transmitted_size = 0
        self._start_to_compute_time = 0
        self._computed_size = 0
        self._last_compute_time = 0
        self._failure_reason_code = -1
        self._farther_mission = farther_mission

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
    
    def transmit_to_Node(self, node_id, trans_data, current_time):
        """Transmit the data to the node.

        Args:
            node_id (str): The ID of the node.

        Returns:
            bool: True if the task is transmitted, False if requires more transmission.
        """
        self._transmitted_size += trans_data
        self._last_transmission_time = current_time
        if self._transmitted_size >= self._task_size:
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

    def isTransmitting(self):
        """Check if the task is transmitting.

        Returns:
            bool: True if the task is transmitting, False otherwise.
        """
        return self._transmitted_size < self._task_size and not self._executed_locally
    
    def isTransmittedToAssignedNode(self):
        """Check if the task is transmitted to the assigned node.

        Returns:
            bool: True if the task is transmitted, False otherwise.
        """
        return self._transmitted_size >= self._task_size and not self._executed_locally and len(self._to_offload_route) == 0

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