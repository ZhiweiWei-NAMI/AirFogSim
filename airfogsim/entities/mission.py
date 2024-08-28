import numpy as np
class Mission:
    """Mission class to represent a mission in the simulation. The mission is a series of waypoints that the UAV/vehicle needs to visit and stay for a while. It also contains a set of tasks that need to be executed at each waypoint.
    """
    def __init__(self, mission_id, appointed_node_id, mission_routes, mission_duration, mission_start_time, mission_deadline, mission_task_sets, distance_threshold=100):
        """The constructor of the Mission class.

        Args:
            mission_id (str): The unique ID of the mission.
            appointed_node_id (str): The appointed node ID.
            mission_routes (list): The list of routes.
            mission_duration (float): The duration of the mission.
            mission_start_time (float): The start time of the mission.
            mission_deadline (float): The deadline of the mission.
            mission_task_sets (list): The list of tasks.
            distance_threshold (float): The distance threshold for the location comparison.
        """
        self._mission_id = mission_id
        self._appointed_node_id = appointed_node_id
        self._mission_routes = mission_routes
        self._mission_duration = mission_duration
        self._mission_stayed_time = np.zeros(len(mission_routes))
        self._last_stayed_time = -np.ones(len(mission_routes))
        self._mission_start_time = mission_start_time
        self._mission_deadline = mission_deadline
        self._mission_task_sets = mission_task_sets
        self._distance_threshold = distance_threshold
        assert len(mission_routes) == len(mission_task_sets) == len(mission_duration), "The length of mission_routes, mission_task_sets, and mission_duration should be the same."
        for taskset in mission_task_sets:
            for task in taskset:
                task.setFartherMission(self)

    def outOfDeadline(self, current_time):
        """Check if the mission is out of deadline.

        Args:
            current_time (float): The current time.

        Returns:
            bool: True if the mission is out of deadline, False otherwise.
        """
        return self._mission_start_time + self._mission_deadline <= current_time
    
    def updateMission(self, time_step, current_time, node):
        """Check the position of each waypoint and update the mission duration.

        Args:
            time_step (float): The time step.
            current_time (float): The current time.
            node (Node): The node object.
        """
        xyz = node.getPosition()
        for i in range(len(self._mission_routes)):
            if self._mission_stayed_time[i] < self._mission_duration[i]:
                if np.linalg.norm(np.array(xyz) - np.array(self._mission_routes[i])) < self._distance_threshold:
                    self._mission_stayed_time[i] += time_step
                    self._last_stayed_time[i] = current_time
                    break
    def isFinished(self):
        """Check if the mission is finished.

        Returns:
            bool: True if the mission is finished, False otherwise.
        """
        task_flag = True
        for taskset in self._mission_task_sets:
            for task in taskset:
                if not task.isFinished():
                    task_flag = False
                    break
        return all(self._mission_stayed_time >= self._mission_duration) and task_flag
        