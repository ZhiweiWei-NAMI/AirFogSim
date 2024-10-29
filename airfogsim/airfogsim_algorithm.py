from .airfogsim_scheduler import AirFogSimScheduler
from .airfogsim_env import AirFogSimEnv
import numpy as np
class BaseAlgorithmModule:
    """Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    def __init__(self):
        self.compScheduler = AirFogSimScheduler.getComputationScheduler()
        self.commScheduler = AirFogSimScheduler.getCommunicationScheduler()
        self.entityScheduler = AirFogSimScheduler.getEntityScheduler()
        self.rewardScheduler = AirFogSimScheduler.getRewardScheduler()
        self.taskScheduler = AirFogSimScheduler.getTaskScheduler()
        self.blockchainScheduler = AirFogSimScheduler.getBlockchainScheduler()
        self.missionScheduler = AirFogSimScheduler.getMissionScheduler()
        self.sensorScheduler=AirFogSimScheduler.getSensorScheduler()
        self.trafficScheduler=AirFogSimScheduler.getTrafficScheduler()


    def initialize(self, env:AirFogSimEnv):
        """Initialize the algorithm with the environment. Should be implemented by the subclass. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.taskScheduler.setTaskGenerationModel(env, 'Poisson')
        self.taskScheduler.setTaskNodePossibility(env, node_types=['vehicle','UAV'], max_num=30, threshold_poss=0.5)
        # self.rewardScheduler.setRewardModel(env, 'log(1+(task_deadline-task_delay))')
        self.rewardScheduler.setRewardModel(env, '"5 * log(10, 1 + _mission_deadline) * (1 / (1 + exp(-_mission_deadline / (_mission_finish_time - _mission_start_time))) - 1 / (1 + exp(-1)))"')

    def scheduleStep(self, env:AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleOffloading(env)
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleMission(env)

    def scheduleMission(self, env:AirFogSimEnv):
        """The mission scheduling logic. Should be implemented by the subclass. Default is selecting the idle sensor with highest accuracy.
        
        Args:
            env (AirFogSimEnv): The environment object.

        """
        new_missions_profile=self.missionScheduler.getToBeAssignedMissionsProfile(env)
        delete_missions_profile=[]
        for mission_profile in new_missions_profile:
            mission_sensor_type=mission_profile['mission_sensor_type']
            mission_accuracy=mission_profile['mission_accuracy']
            appointed_node_id,appointed_sensor_id,appointed_sensor_accuracy=self.sensorScheduler.getAppointedSensor(env,mission_sensor_type,mission_accuracy)

            if appointed_node_id!=None and appointed_sensor_id!=None:
                mission_profile['appointed_node_id'] = appointed_node_id
                mission_profile['appointed_sensor_id'] = appointed_sensor_id
                mission_profile['appointed_sensor_accuracy'] = appointed_sensor_accuracy
                mission_profile['mission_start_time'] = self.trafficScheduler.getCurrentTime(env)
                for _ in mission_profile['mission_routes']:
                    task_set=[]
                    mission_task_profile={
                        'task_node_id':appointed_node_id,
                        'task_deadline':mission_profile['mission_deadline'],
                        'arrival_time':mission_profile['mission_arrival_time']
                    }
                    new_task=self.taskScheduler.generateTaskOfMission(env,mission_task_profile)
                    task_set.append(new_task)
                    mission_profile['mission_task_sets'].append(task_set)
                self.missionScheduler.generateAndAddMission(env,mission_profile)

                delete_missions_profile.append(mission_profile)
        self.missionScheduler.deleteBeAssignedMissionsProfile(delete_missions_profile)


        # 1. generate mission (according to Poisson)
        # missionScheduler = AirFogSimScheduler.getMissionScheduler()
        # mission_profiles = [{
        #     'id':'Mission-1',
        #     'position': (100,230),
        #     'duration': 100,
        #     'task_profiles': [{
        #         'id':'Task-1',
        #         'task_type':'classification',
        #         'task_node_id':'Node-1',
        #         'task_deadline': 100,
        #         'task_data_size': 100,
        #         'task_computation': 100,
        #         'task_offloading': 100
        #     }]
        #     } for _ in range(3)]
        # for mission_profile in mission_profiles:
        #     missionScheduler.generateMission(env, mission_profile)
        
    def scheduleOffloading(self, env:AirFogSimEnv):
        """The offloading scheduling logic. Should be implemented by the subclass. Default is to offload the task to the nearest node.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        all_task_infos = self.taskScheduler.getAllToOffloadTaskInfos(env)
        all_node_infos = self.entityScheduler.getAllNodeInfos(env)
        all_node_infos_dict = {}
        for node_info in all_node_infos:
            all_node_infos_dict[node_info['id']] = node_info
        for task_dict in all_task_infos:
            task_node_id = task_dict['task_node_id']
            task_id = task_dict['task_id']
            task_node = all_node_infos_dict[task_node_id]
            neighbor_infos = self.entityScheduler.getNeighborNodeInfosById(env, task_node_id, sorted_by='distance', max_num=5)
            nearest_node_id = neighbor_infos[0]['id']
            self.taskScheduler.setTaskOffloading(env, task_node_id, task_id, nearest_node_id)

    def scheduleCommunication(self, env:AirFogSimEnv):
        """The communication scheduling logic. Should be implemented by the subclass. Default is random.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        n_RB = self.commScheduler.getNumberOfRB(env)
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        for task_dict in all_offloading_task_infos:
            allocated_RB_nos = np.random.choice(n_RB, 3, replace=False)
            self.commScheduler.setCommunicationWithRB(env, task_dict['task_id'], allocated_RB_nos)

    def scheduleComputing(self, env:AirFogSimEnv):
        """The computing scheduling logic. Should be implemented by the subclass. Default is FIFS.
        
        Args:
            env (AirFogSimEnv): The environment object.
        """
        all_offloading_task_infos = self.taskScheduler.getAllOffloadingTaskInfos(env)
        for task_dict in all_offloading_task_infos:
            task_id = task_dict['task_id']
            task_node_id = task_dict['task_node_id']
            assigned_node_id = task_dict['assigned_to']
            assigned_node_info = self.entityScheduler.getNodeInfoById(env, assigned_node_id)
            self.compScheduler.setComputingWithNodeCPU(env, task_id, 0.3) # allocate cpu 0.3

    def getRewardByTask(self, env:AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done tasks in last time.
        
        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_task_infos = self.taskScheduler.getLastStepSuccTaskInfos(env)
        reward = 0
        for task_info in last_step_succ_task_infos:
            reward += self.rewardScheduler.getRewardByTask(env, task_info)
        return reward

    def getRewardByMission(self, env: AirFogSimEnv):
        """The reward calculation logic. Should be implemented by the subclass. Default is calculating reward of done missions in last time.

        Args:
            env (AirFogSimEnv): The environment object.

        Returns:
            float: The reward value.
        """
        last_step_succ_task_infos = self.missionScheduler.getLastStepSuccMissionInfos(env)
        reward = 0
        for mission_info in last_step_succ_task_infos:
            reward += self.rewardScheduler.getRewardByMission(env, mission_info)
        return reward