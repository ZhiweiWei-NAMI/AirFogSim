from base_sched import BaseScheduler

class MissionScheduler(BaseScheduler):
    @staticmethod
    def getToBeAssignedMissionsProfile(env):
        return env.mission_manager.getArrivedMissionsProfile()

    @staticmethod
    def deleteBeAssignedMissionsProfile(env,missions_profile):
        env.mission_manager.deleteMissionsProfile(missions_profile)


    @staticmethod
    def generateAndAddMission(env,mission_profile):
        mission=env.mission_manager.generateMission(mission_profile)
        env.new_missions.append(mission)

    @staticmethod
    def getLastStepSuccMissionInfos(env):
        """Get the success mission infos for last timeslot.

        Args:
            env (AirFogSimEnv): The AirFogSim environment.

        Returns:
            list: The list of mission infos (mission: object).
        """
        recently_done_100_missions = env.mission_manager.getRecentlyDoneMissions()
        last_step = env.simulation_time - env.traffic_interval
        mission_info_list = []
        for mission in recently_done_100_missions:
            if mission.isFinished() and mission.getMissionFinishTime() >= last_step:
                mission_info_list.append(mission.to_dict())
        return mission_info_list
