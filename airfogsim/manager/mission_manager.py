import random
from collections import deque
from ..entities.mission import Mission

import numpy as np


class MissionManager:
    """MissionManager class is responsible for managing the missions for each node according to its ids.
    """
    SUPPORTED_TASK_GENERATION_MODELS = ['Poisson', 'Uniform', 'Normal', 'Exponential']

    def __init__(self,config_mission):
        """The constructor of the MissionManager class.
        """
        self._missions = {} # key: node_id, value: list of missions
        self._success_missions = {} # key: node_id, value: list of success missions
        self._failed_missions = {} # key: node_id, value: list of failed missions
        self._to_generate_missions_profile=[]
        self._recently_done_100_missions= deque(maxlen=100)
        self._mission_id_counter=0
        self._config_mission=config_mission

    def __getNewMissionId(self):
        self._mission_id_counter+=1
        return self._mission_id_counter

    def _generateBasicMissionProfile(self):
        new_mission_profile={}
        new_mission_profile['mission_id']=self.__getNewMissionId()
        x=random.randint(0,100)
        y=random.randint(0,100)
        new_mission_profile['appointed_node_id']=None
        new_mission_profile['appointed_sensor_id'] = None
        new_mission_profile['appointed_sensor_accuracy'] = None
        new_mission_profile['mission_routes']=[(x,y, self._config_mission['UAV_height'])]
        start, end = self._config_mission['duration_range']
        new_mission_profile['mission_duration']=[random.randint(start, end)]
        size_min,size_max=self._config_mission['task_size_range']
        new_mission_profile['mission_size']=random.randint(size_min, size_max)
        new_mission_profile['mission_sensor_type']='Sensor_type_' + str(random.randint(1, self._config_mission['sensor_type_num']))
        new_mission_profile['mission_accuracy']=random.random()  # 随机生成0-1之间的精度
        new_mission_profile['mission_start_time'] = None
        new_mission_profile['mission_deadline']=random.randint(new_mission_profile['mission_duration']+10, new_mission_profile['mission_duration']+200)
        new_mission_profile['mission_task_sets']=[]
        new_mission_profile['mission_arrival_time'] = None
        new_mission_profile['distance_threshold'] = 100
        return new_mission_profile

    def _generateMissionsProfile(self, cur_time, simulation_interval):
        # Generate new _to_generate_missions_profile, oblige to the mission generation model, simulation_interval, and predictable_seconds
        todo_mission_num = 0
        todo_mission_num += len(self._to_generate_missions_profile)
        last_generation_time = cur_time if len(self._to_generate_missions_profile) == 0 else self._to_generate_missions_profile[-1].get('taskArrivalTime')
        last_generation_time += simulation_interval
        while last_generation_time <= cur_time + self._config_mission.get('predictable_seconds'):
            if self._config_mission.get('mission_generation_model')== 'Poisson':
                kwlambda = self._config_mission['generation_model_args']['Poisson']['lambda']
                mission_num = np.random.poisson(kwlambda * simulation_interval)

            elif self._config_mission.get('mission_generation_model') == 'Uniform':
                kwlow = self._config_mission['generation_model_args']['Uniform']['low']
                kwhigh = self._config_mission['generation_model_args']['Uniform']['high']
                mission_num = np.random.randint(kwlow * simulation_interval, kwhigh * simulation_interval+1)
                assert int(kwlow * simulation_interval) < int(kwhigh * simulation_interval), 'There is no task to generate.'

            elif self._config_mission.get('mission_generation_model') == 'Normal':
                kwmean =  self._config_mission['generation_model_args']['Normal']['mean']
                kwstd =  self._config_mission['generation_model_args']['Normal']['std']
                mission_num = np.random.normal(kwmean * simulation_interval, kwstd * simulation_interval)
                assert kwmean * simulation_interval > 0, 'There is no task to generate.'
                mission_num = int(mission_num)
                mission_num = mission_num if mission_num > 0 else 0

            elif self._config_mission.get('mission_generation_model') == 'Exponential':
                kwbeta = self._config_mission['generation_model_args']['Exponential']['beta']
                mission_num = np.random.exponential(kwbeta * simulation_interval)
                assert kwbeta * simulation_interval > 0, 'There is no task to generate.'
                mission_num = int(mission_num)

            else:
                raise NotImplementedError('The mission generation model is not implemented.')

            for i in range(mission_num):
                new_mission_profile = self._generateBasicMissionProfile()
                new_mission_profile['mission_arrival_time'] = last_generation_time
                self._to_generate_missions_profile.append(new_mission_profile)
                todo_mission_num += 1
            last_generation_time += simulation_interval

        return todo_mission_num

    def generateMission(self,mission_profile):
        """Generate new mission according to mission_profile.

        Returns:
            Mission: A new mission object.
        """
        return Mission(mission_profile)

    def addMission(self, mission,sensor_manager):
        """Add the mission to the node.

        Args:
            node_id (str): The ID of the node.
            mission (Mission): The mission to add.
            task_manager (TaskManager): The task manager.
            current_time (float): The current time.
        """
        node_id=mission.getAppointedNodeId()
        if node_id not in self._missions:
            self._missions[node_id] = []
        self._missions[node_id].append(mission)
        sensor_manager.startUseById(mission.getAppointedSensorId())
        # for taskset in mission.getMissionTasks():
        #     # The tasks in mission is released by the cloud server
        #     for task in taskset:
        #         task_manager.addToComputeTask(task, node_id, current_time)

    def updateMissions(self, time_step, current_time, _getNodeById,sensor_manager,task_manager):
        """Update the missions at the current time.

        Args:
            current_time (float): The current time.
            _getNodeById (function): The function to get the node by ID.
        """
        for node_id in self._missions:
            to_remove = []
            for mission in self._missions[node_id]:
                sensor_id=mission.getAppointedSensorId()
                if mission.outOfDeadline(current_time):
                    self._failed_missions[node_id].append(mission)
                    to_remove.append(mission)
                    sensor_manager.endUseById(sensor_id)
                    continue
                node = _getNodeById(node_id)
                sensor_usable=sensor_manager.getUsableById(sensor_id)
                mission.updateMission(time_step, current_time, node,sensor_usable)

                # Task start only when sensing is completed
                routes_length=mission.getRoutesLength()
                for index, route in enumerate(routes_length):
                    if mission.isSensingFinished(index):
                        tasks=mission.getMissionTasks(index)
                        for task in tasks:
                            if not task.isStarted():
                                task_manager.addToComputeTask(task, node_id, current_time)

                if mission.isFinished():
                    mission.setMissionFinishTime(current_time) # set finish time (after returning all data)
                    self._success_missions[node_id].append(mission)
                    self._recently_done_100_missions.append(mission)
                    to_remove.append(mission)
                    sensor_manager.endUseById(sensor_id)
                    continue

            for mission in to_remove:
                self._missions[node_id].remove(mission)

    def getRecentlyDoneMissions(self):
        """Get the recently done missions (the maximum number is 100).

        Returns:
            list: The list of the recently done missions.
        """
        return self._recently_done_100_missions

    def getDoneMissionByMissionNodeAndMissionId(self,appointed_node_id,mission_id):
        """Get the done task by the task node id and the task id.

        Args:
            task_node_id (str): The task node id.
            task_id (str): The task id.

        Returns:
            Task: The task.

        Examples:
            task_manager.getDoneTaskByTaskNodeAndTaskId('vehicle1', 'Task_1')
        """
        if appointed_node_id in self._success_missions:
            for mission in self._success_missions[appointed_node_id]:
                if mission.getMissionId() == mission_id:
                    return mission
        return None

    def getFailedMissionNum(self):
        failed_count=0
        for node_id in self._success_missions:
            failed_count += len(self._failed_missions[node_id])

        return failed_count

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


    def getArrivedMissionsProfile(self,cur_time):
        """Get arrived mission profile.

        Args:
            cur_time (float): The current time.

        Returns:
            dict: Misssion profiles.
        """
        arrived_missions_profile=[]
        for mission_profile in self._to_generate_missions_profile:
            generation_time=mission_profile['mission_arrival_time']
            if generation_time<=cur_time:
                arrived_missions_profile.append(mission_profile)
        return arrived_missions_profile

    def deleteMissionsProfile(self,missions_profile):
        """Delete mission profile after building mission by mission profile.

        Args:
            missions_profile (dict): The used missions profile.

        Returns:

        """
        # 使用列表推导，从A中删除B中的元素
        self._to_generate_missions_profile = [item for item in self._to_generate_missions_profile if item not in missions_profile]





    