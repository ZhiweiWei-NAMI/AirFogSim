import random
from collections import deque
from ..entities.mission import Mission

import numpy as np


class MissionManager:
    """MissionManager class is responsible for managing the missions for each node according to its ids.
    """
    SUPPORTED_TASK_GENERATION_MODELS = ['Poisson', 'Uniform', 'Normal', 'Exponential']

    def __init__(self, config_mission, config_sensing):
        """The constructor of the MissionManager class.
        """
        self._to_generate_missions_profile = []  # list: item is mission_profile dicts
        self._executing_missions = {}  # key: node_id, value: list of missions
        self._success_missions = {}  # key: node_id, value: list of success missions
        self._failed_missions = {}  # key: node_id, value: list of failed missions
        self._early_failed_missions = []  # list: item is early failed missions(missions failed before assigned)
        self._recently_done_100_missions = deque(maxlen=100)
        self._recently_fail_100_missions = deque(maxlen=100)
        self._mission_id_counter = 0

        self._config_mission = config_mission
        self._x_range = self._config_mission.get("x_range", [0, 1000])
        self._y_range = self._config_mission.get("y_range", [0, 1000])
        self._UAV_height = self._config_mission.get('UAV_height', 100)
        self._TTL_range=self._config_mission.get('TTL_range', [100,200])
        self._duration_range = self._config_mission.get('duration_range', [5, 10])
        self._mission_size_range = self._config_mission.get('mission_size_range', [10, 20])
        self._sensor_accuracy_range = self._config_mission.get('sensor_accuracy_range', [0, 1])
        self._predictable_seconds = self._config_mission.get('predictable_seconds', 5)
        self._mission_generation_model = self._config_mission['mission_generation_model']
        self._generation_model_args = self._config_mission['generation_model_args']
        self._distance_threshold=self._config_mission['distance_threshold']

        self._sensor_type_num = config_sensing['sensor_type_num']

    def reset(self):
        self._to_generate_missions_profile = []  # list: item is mission_profile dicts
        self._executing_missions = {}  # key: node_id, value: list of missions
        self._success_missions = {}  # key: node_id, value: list of success missions
        self._failed_missions = {}  # key: node_id, value: list of failed missions
        self._early_failed_missions = []  # list: item is early failed missions(missions failed before assigned)
        self._recently_done_100_missions = deque(maxlen=100)
        self._recently_fail_100_missions = deque(maxlen=100)
        self._mission_id_counter = 0

    def __getNewMissionId(self):
        self._mission_id_counter += 1
        return self._mission_id_counter

    def _generateBasicMissionProfile(self):
        new_mission_profile = {}
        new_mission_profile['mission_id'] = f'Mission_{self.__getNewMissionId()}'
        x = random.uniform(self._x_range[0], self._x_range[1])
        y = random.uniform(self._y_range[0], self._y_range[1])
        new_mission_profile['appointed_node_id'] = None
        new_mission_profile['appointed_sensor_id'] = None
        new_mission_profile['appointed_sensor_accuracy'] = None
        new_mission_profile['mission_routes'] = [(x, y, self._UAV_height)]
        start, end = self._duration_range
        new_mission_profile['mission_duration'] = [random.randint(start, end)]
        size_min, size_max = self._mission_size_range
        new_mission_profile['mission_size'] = random.randint(size_min, size_max)
        new_mission_profile['mission_sensor_type'] = 'Sensor_type_' + str(random.randint(1, self._sensor_type_num))
        new_mission_profile['mission_accuracy'] = random.random()  # 随机生成0-1之间的精度
        new_mission_profile['mission_start_time'] = None
        new_mission_profile['mission_deadline'] = random.randint(self._TTL_range[0],self._TTL_range[1]) # TTL
        new_mission_profile[
            'mission_task_sets'] = []  # Each point to sensing have a task set, each set have several tasks
        new_mission_profile['mission_arrival_time'] = None
        new_mission_profile['distance_threshold'] = self._distance_threshold
        return new_mission_profile

    def generateMissionsProfile(self, cur_time, simulation_interval):
        """Generate new mission profiles.

        Returns:
            int: New missions num.
        """
        # Generate new _to_generate_missions_profile, oblige to the mission generation model, simulation_interval, and predictable_seconds
        todo_mission_num = 0
        todo_mission_num += len(self._to_generate_missions_profile)
        last_generation_time = cur_time if len(self._to_generate_missions_profile) == 0 else \
        self._to_generate_missions_profile[-1].get('mission_arrival_time')
        last_generation_time += simulation_interval
        while last_generation_time <= cur_time + self._predictable_seconds:
            if self._mission_generation_model == 'Poisson':
                kwlambda = self._generation_model_args['Poisson']['lambda']
                mission_num = np.random.poisson(kwlambda * simulation_interval)

            elif self._mission_generation_model == 'Uniform':
                kwlow = self._generation_model_args['Uniform']['low']
                kwhigh = self._generation_model_args['Uniform']['high']
                mission_num = np.random.randint(kwlow * simulation_interval, kwhigh * simulation_interval + 1)
                assert int(kwlow * simulation_interval) < int(
                    kwhigh * simulation_interval), 'There is no task to generate.'

            elif self._mission_generation_model == 'Normal':
                kwmean = self._generation_model_args['Normal']['mean']
                kwstd = self._generation_model_args['Normal']['std']
                mission_num = np.random.normal(kwmean * simulation_interval, kwstd * simulation_interval)
                assert kwmean * simulation_interval > 0, 'There is no task to generate.'
                mission_num = int(mission_num)
                mission_num = mission_num if mission_num > 0 else 0

            elif self._mission_generation_model == 'Exponential':
                kwbeta = self._generation_model_args['Exponential']['beta']
                mission_num = np.random.exponential(kwbeta * simulation_interval)
                assert kwbeta * simulation_interval > 0, 'There is no task to generate.'
                mission_num = int(mission_num)

            else:
                raise NotImplementedError('The mission generation model is not implemented.')

            # now_mission_num = 0
            # for node_id, missions in self._executing_missions.items():
            #     now_mission_num += len(missions)
            # now_mission_num += len(self._to_generate_missions_profile)
            # mission_num = min(max(0, 20 - now_mission_num), mission_num)

            for i in range(mission_num):
                new_mission_profile = self._generateBasicMissionProfile()
                new_mission_profile['mission_arrival_time'] = last_generation_time
                self._to_generate_missions_profile.append(new_mission_profile)
                todo_mission_num += 1
            last_generation_time += simulation_interval

        return todo_mission_num

    def generateMission(self, mission_profile):
        """Generate new mission according to mission_profile.

        Returns:
            Mission: A new mission object.
        """
        return Mission(mission_profile)

    def addMission(self, mission, sensor_manager):
        """Add the mission to the node.

        Args:
            node_id (str): The ID of the node.
            mission (Mission): The mission to add.
            task_manager (TaskManager): The task manager.
            current_time (float): The current time.
        """
        node_id = mission.getAppointedNodeId()
        if node_id not in self._executing_missions:
            self._executing_missions[node_id] = []
        self._executing_missions[node_id].append(mission)
        sensor_manager.startUseById(mission.getAppointedSensorId())

    def updateMissions(self, time_step, current_time, _getNodeById, sensor_manager, task_manager):
        """Update the missions at the current time.

        Args:
            current_time (float): The current time.
            _getNodeById (function): The function to get the node by ID.
        """
        self._checkMissions(current_time, sensor_manager)

        for node_id in self._executing_missions:
            to_remove = []
            for mission in self._executing_missions[node_id]:
                sensor_id = mission.getAppointedSensorId()
                node = _getNodeById(node_id)
                mission.updateMission(time_step, current_time, node)

                # Task start only when sensing is completed
                routes_length = mission.getRoutesLength()
                for index in range(routes_length):
                    if mission.isSensingFinished(index):
                        tasks = mission.getMissionTasks(index)
                        for task in tasks:
                            if not task.isStarted():
                                task_manager.addToComputeTask(task, node_id, current_time)

                if mission.isFinished():
                    mission.setMissionFinishTime(current_time)  # set finish time (after returning all data)
                    success_missions_on_node = self._success_missions.get(node_id, [])
                    success_missions_on_node.append(mission)
                    self._success_missions[node_id] = success_missions_on_node
                    self._recently_done_100_missions.append(mission)
                    to_remove.append(mission)
                    sensor_manager.endUseById(sensor_id)
                    continue

            for mission in to_remove:
                self._executing_missions[node_id].remove(mission)

    def _checkMissions(self, current_time, sensor_manager):
        # 1.Check missions in _executing_missions
        for node_id in self._executing_missions:
            to_remove = []
            for mission in self._executing_missions[node_id]:
                sensor_id = mission.getAppointedSensorId()
                if mission.outOfDeadline(current_time):
                    mission.setMissionFinishTime(current_time)  # set finish time (when failed)
                    failed_missions_on_node = self._failed_missions.get(node_id, [])
                    failed_missions_on_node.append(mission)
                    self._failed_missions[node_id] = failed_missions_on_node
                    self._recently_fail_100_missions.append(mission)
                    to_remove.append(mission)
                    sensor_manager.endUseById(sensor_id)
            for mission in to_remove:
                self._executing_missions[node_id].remove(mission)
        # 2.Check to be generated mission profiles in _to_generate_missions_profile
        to_remove = []
        for mission_profile in self._to_generate_missions_profile:
            if mission_profile['mission_deadline']+mission_profile['mission_arrival_time'] < current_time:
                dead_mission = Mission(mission_profile)
                dead_mission.setMissionFinishTime(current_time)
                self._early_failed_missions.append(dead_mission)
                self._recently_fail_100_missions.append(dead_mission)
                to_remove.append(dead_mission.getMissionId())
        self.deleteMissionsProfile(to_remove)

    def getRecentlyDoneMissions(self):
        """Get the recently done missions (the maximum number is 100).

        Returns:
            list: The list of the recently done missions.
        """
        return self._recently_done_100_missions

    def getRecentlyFailMissions(self):
        """Get the recently failed missions (the maximum number is 100).

        Returns:
            list: The list of the recently done missions.
        """
        return self._recently_fail_100_missions

    def getDoneMissionByMissionNodeAndMissionId(self, appointed_node_id, mission_id):
        """Get the done missions by the appointed node id and the mission id.

        Args:
            appointed_node_id (str): The appointed node id.
            mission_id (str): The mission id.

        Returns:
            Mission: The mission.

        Examples:
            mission_manager.getDoneMissionByMissionNodeAndMissionId('vehicle1', 'Mission_1')
        """
        if appointed_node_id in self._success_missions:
            for mission in self._success_missions[appointed_node_id]:
                if mission.getMissionId() == mission_id:
                    return mission
        return None

    def getFailMissionByMissionNodeAndMissionId(self, appointed_node_id, mission_id):
        """Get the failed missions by the appointed node id and the mission id.

        Args:
            appointed_node_id (str): The appointed node id.
            mission_id (str): The mission id.

        Returns:
            Mission: The mission.

        Examples:
            mission_manager.getFailMissionByMissionNodeAndMissionId('vehicle1', 'Mission_1')
        """
        assert appointed_node_id in self._failed_missions
        for mission in self._failed_missions[appointed_node_id]:
            if mission.getMissionId() == mission_id:
                return mission
        return None

    def getEarlyFailMissionByMissionId(self,mission_id):
        """Get the early failed missions by the mission id.

         Args:
             mission_id (str): The mission id.

         Returns:
             Mission: The mission.

         Examples:
             mission_manager.getEarlyFailMissionByMissionId('Mission_1')
         """
        for mission in self._early_failed_missions:
            if mission.getMissionId() == mission_id:
                return mission
        return None

    def getToGenerateMissionNum(self):
        """Get the to generate missions total number.

        Returns:
            int: The total count of to generate missions.
        """
        return len(self._to_generate_missions_profile)

    def getExecutingMissionNum(self):
        """Get the executing missions total number.

        Returns:
            int: The total count of executing missions.
        """
        executing_count = 0
        for node_id in self._executing_missions:
            executing_count += len(self._executing_missions[node_id])
        return executing_count

    def getSuccessMissionNum(self):
        """Get the success missions total number.

        Returns:
            int: The total count of success missions.
        """
        success_count = 0
        for node_id in self._success_missions:
            success_count += len(self._success_missions[node_id])
        return success_count

    def getFailedMissionNum(self):
        """Get the failed missions total number.

        Returns:
            int: The total count of failed missions.
        """
        failed_count = 0
        for node_id in self._failed_missions:
            failed_count += len(self._failed_missions[node_id])
        return failed_count

    def getEarlyFailedMissionNum(self):
        """Get the early failed missions total number.

        Returns:
            int: The total count of early failed missions.
        """
        return len(self._early_failed_missions)

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
        total_count+= len(self._early_failed_missions)
        ratio = success_count / total_count if total_count > 0 else 0.0
        return ratio, total_count

    def getArrivedMissionsProfile(self, cur_time):
        """Get arrived mission profile.

        Args:
            cur_time (float): The current time.

        Returns:
            dict: Misssion profiles.
        """
        arrived_missions_profile = []
        for mission_profile in self._to_generate_missions_profile:
            generation_time = mission_profile['mission_arrival_time']
            if generation_time <= cur_time:
                arrived_missions_profile.append(mission_profile)
        return arrived_missions_profile

    def deleteMissionsProfile(self, mission_profile_ids):
        """Delete mission profile after building mission by mission profile.

        Args:
            mission_profile_ids (list): The assigned mission ids.

        Returns:

        """
        # 使用列表推导，删除已分配的profile
        self._to_generate_missions_profile = [profile for profile in self._to_generate_missions_profile if
                                              profile['mission_id'] not in mission_profile_ids]

    def failExecutingMissionsByNodeId(self, to_fail_node_id,current_time):
        """Set fail missions

        Args:
            mission (mission): The mission object.

        Examples:
            mission_manager.failMissions(mission)
        """

        # remove all related missions in self._executing_missions
        for node_id in list(self._executing_missions.keys()):
            mission_set = self._executing_missions.get(node_id)
            if node_id == to_fail_node_id:
                for mission in mission_set.copy():
                    mission.setMissionFinishTime(current_time)
                    failed_mission_set = self._failed_missions.get(mission.getAppointedNodeId(), [])
                    failed_mission_set.append(mission)
                    self._failed_missions[mission.getAppointedNodeId()] = failed_mission_set
                    mission_set.remove(mission)
                    self._executing_missions[node_id] = mission_set
                del self._executing_missions[node_id]
            else:
                for mission in mission_set.copy():
                    if mission.isRelatedToNode(to_fail_node_id):
                        mission.setMissionFinishTime(current_time)
                        failed_mission_set = self._failed_missions.get(mission.getAppointedNodeId(), [])
                        failed_mission_set.append(mission)
                        self._failed_missions[mission.getAppointedNodeId()] = failed_mission_set
                        mission_set.remove(mission)
                        self._executing_missions[node_id] = mission_set
    def failNewMission(self,mission,current_time):
        mission.setMissionFinishTime(current_time)
        self._failed_missions[mission.getAppointedNodeId()] = self._failed_missions.get(mission.getAppointedNodeId(), [])
        self._failed_missions[mission.getAppointedNodeId()].append(mission)

    def getExecutingMissions(self):
        """Get executing missions

        Args:

        Returns:
            list: Executing missions list.

        Examples:
            mission_manager.getExecutingMissions()
        """
        return self._executing_missions

    def getConfig(self,name):
        return self._config_mission.get(name,None)
