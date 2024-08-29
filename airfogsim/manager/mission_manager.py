class MissionManager:
    """MissionManager class is responsible for managing the missions for each node according to its ids.
    """

    def __init__(self):
        """The constructor of the MissionManager class.
        """
        self._missions = {} # key: node_id, value: list of missions
        self._success_missions = {} # key: node_id, value: list of success missions
        self._failed_missions = {} # key: node_id, value: list of failed missions

    def addMission(self, node_id, mission, task_manager, current_time):
        """Add the mission to the node.

        Args:
            node_id (str): The ID of the node.
            mission (Mission): The mission to add.
            task_manager (TaskManager): The task manager.
            current_time (float): The current time.
        """
        if node_id not in self._missions:
            self._missions[node_id] = []
        self._missions[node_id].append(mission)
        for taskset in mission._mission_task_sets:
            # The tasks in mission is released by the cloud server
            for task in taskset:
                task_manager.addToComputeTask(task, node_id, current_time)

    def updateMissions(self, time_step, current_time, _getNodeById):
        """Update the missions at the current time.

        Args:
            current_time (float): The current time.
            _getNodeById (function): The function to get the node by ID.
        """
        for node_id in self._missions:
            to_remove = []
            for mission in self._missions[node_id]:
                if mission.outOfDeadline(current_time):
                    self._failed_missions[node_id].append(mission)
                    to_remove.append(mission)
                    continue
                node = _getNodeById(node_id)
                mission.updateMission(time_step, current_time, node)
                if mission.isFinished():
                    self._success_missions[node_id].append(mission)
                    to_remove.append(mission)
                    continue
            for mission in to_remove:
                self._missions[node_id].remove(mission)

    def getMissionCompletionRatio(self):
        """Get the mission completion ratio.

        Returns:
            float: The mission completion ratio.
            int: The total count of missions.
        """
        success_count = 0
        total_count = 0
        for node_id in self._success_missions:
            success_count += len(self._success_missions[node_id])
            total_count += len(self._success_missions[node_id]) + len(self._failed_missions.get(node_id, []))
        ratio = success_count / total_count if total_count > 0 else 0.0
        return ratio, total_count
    