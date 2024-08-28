class MissionManager:
    """MissionManager class is responsible for managing the missions for each node according to its ids.
    """

    def __init__(self):
        """The constructor of the MissionManager class.
        """
        self._missions = {} # key: node_id, value: list of missions
        self._success_missions = {} # key: node_id, value: list of success missions
        self._failed_missions = {} # key: node_id, value: list of failed missions

    def addMission(self, node_id, mission):
        """Add the mission to the node.

        Args:
            node_id (str): The ID of the node.
            mission (Mission): The mission to add.
        """
        if node_id not in self._missions:
            self._missions[node_id] = []
        self._missions[node_id].append(mission)

    def updateMissions(self, time_step, current_time, _getNodeById):
        """Update the missions at the current time.

        Args:
            current_time (float): The current time.
            _getNodeById (function): The function to get the node by ID.
        """
        for node_id in self._missions:
            for mission in self._missions[node_id]:
                if mission.outOfDeadline(current_time):
                    self._failed_missions[node_id].append(mission)
                    continue
                node = _getNodeById(node_id)
                mission.updateMission(time_step, current_time, node)
                if mission.isFinished():
                    self._success_missions[node_id].append(mission)
                    continue