import math


# from airfogsim_env import AirFogSimEnv
# from airfogsim_algorithm import BaseAlgorithmModule,NVHAUAlgorithmModule


class AirFogSimEvaluation:
    # 仿真时间
    simulation_time=0

    # 过程监控
    to_generate_missions_num = 0
    executing_missions_num = 0
    success_missions_num = 0
    failed_missions_num = 0
    early_failed_missions_num = 0

    # 信道速率指标
    avg_trans_rate = 0
    avg_V2U_trans_rate = 0
    avg_V2I_trans_rate = 0
    avg_U2I_trans_rate = 0

    # reward
    sum_reward = 0
    sum_success_reward = 0
    avg_reward = 0
    avg_success_reward = 0

    # ratio
    finish_mission_num = 0
    completion_ratio = 0

    sum_weight = 0
    sum_weighted_acceleration_ratio = 0
    weighted_acceleration_ratio = 0

    sum_acceleration_ratio = 0
    acceleration_ratio = 0

    # optimization objectives
    sum_executing_time = 0
    sum_sensing_time = 0
    avg_floating_time = 0
    avg_executing_time = 0

    @staticmethod
    def updateEvaluationIndicators(env, algorithm_module):
        step_reward, step_punish, step_sum_reward = algorithm_module.getRewardByMission(
            env)  # success reward, fail punish, sum of reward and punish
        to_generate_missions_num = env.mission_manager.getToGenerateMissionNum()
        executing_missions_num = env.mission_manager.getExecutingMissionNum()
        success_missions_num = env.mission_manager.getSuccessMissionNum()
        failed_missions_num = env.mission_manager.getFailedMissionNum()
        early_failed_missions_num = env.mission_manager.getEarlyFailedMissionNum()
        sum_over_missions = success_missions_num + failed_missions_num + early_failed_missions_num

        # 仿真时间
        AirFogSimEvaluation.simulation_time=env.simulation_time

        # reward
        AirFogSimEvaluation.sum_reward += step_sum_reward
        AirFogSimEvaluation.sum_success_reward += step_reward
        AirFogSimEvaluation.avg_reward = AirFogSimEvaluation.sum_reward / sum_over_missions if sum_over_missions != 0 else 0
        AirFogSimEvaluation.avg_success_reward = AirFogSimEvaluation.sum_success_reward / success_missions_num if success_missions_num != 0 else 0

        # 过程监控
        AirFogSimEvaluation.to_generate_missions_num = to_generate_missions_num
        AirFogSimEvaluation.executing_missions_num = executing_missions_num
        AirFogSimEvaluation.success_missions_num = success_missions_num
        AirFogSimEvaluation.early_failed_missions_num = early_failed_missions_num
        AirFogSimEvaluation.failed_missions_num = failed_missions_num

        # 信道速率指标
        AirFogSimEvaluation.avg_trans_rate = env.getChannelAvgRate()
        AirFogSimEvaluation.avg_V2U_trans_rate = env.getChannelAvgRate(channel_type='V2U')
        AirFogSimEvaluation.avg_V2I_trans_rate = env.getChannelAvgRate(channel_type='V2I')
        AirFogSimEvaluation.avg_U2I_trans_rate = env.getChannelAvgRate(channel_type='U2I')

        # ratio
        last_step_succ_mission_infos = algorithm_module.missionScheduler.getLastStepSuccMissionInfos(env)
        last_step_fail_mission_infos = algorithm_module.missionScheduler.getLastStepFailMissionInfos(env)
        AirFogSimEvaluation.completion_ratio = env.mission_manager.getMissionCompletionRatio()
        AirFogSimEvaluation.finish_mission_num += len(last_step_succ_mission_infos)
        for mission_info in last_step_succ_mission_infos:
            TTL = mission_info['mission_deadline']
            duration = mission_info['mission_duration_sum']
            finish_time = mission_info['mission_finish_time']
            arrival_time = mission_info['mission_arrival_time']

            weight = math.log(10, 1 + TTL - duration)
            ratio = (TTL - duration) / (finish_time - arrival_time - duration)

            # print('TTL',TTL-duration)
            # print('executing',finish_time-arrival_time-duration)
            # print('weight',weight )
            # print('ratio',ratio )
            # print('size', mission_info['mission_size'])


            AirFogSimEvaluation.sum_weight += weight
            AirFogSimEvaluation.sum_weighted_acceleration_ratio += weight * ratio
            AirFogSimEvaluation.sum_acceleration_ratio += ratio

        AirFogSimEvaluation.weighted_acceleration_ratio = AirFogSimEvaluation.sum_weighted_acceleration_ratio / AirFogSimEvaluation.sum_weight if AirFogSimEvaluation.sum_weight > 0.0 else 0
        AirFogSimEvaluation.acceleration_ratio = AirFogSimEvaluation.sum_acceleration_ratio / AirFogSimEvaluation.finish_mission_num if AirFogSimEvaluation.finish_mission_num > 0 else 0

        # optimization objectives
        for mission_info in last_step_succ_mission_infos:
            duration = mission_info['mission_duration_sum']
            finish_time = mission_info['mission_finish_time']
            arrival_time = mission_info['mission_arrival_time']
            AirFogSimEvaluation.sum_executing_time += finish_time - arrival_time
            AirFogSimEvaluation.sum_sensing_time += duration

        AirFogSimEvaluation.avg_floating_time = (AirFogSimEvaluation.sum_executing_time - AirFogSimEvaluation.sum_sensing_time) / AirFogSimEvaluation.finish_mission_num if AirFogSimEvaluation.finish_mission_num > 0 else 0
        AirFogSimEvaluation.avg_executing_time = AirFogSimEvaluation.sum_executing_time / AirFogSimEvaluation.finish_mission_num if AirFogSimEvaluation.finish_mission_num > 0 else 0

    @staticmethod
    def printEvaluation():
        # 仿真时间
        print('simulation_time: ',AirFogSimEvaluation.simulation_time)

        # 过程监控
        print('过程监控')
        print('to_generate_missions_num: ', AirFogSimEvaluation.to_generate_missions_num)
        print('executing_missions_num: ', AirFogSimEvaluation.executing_missions_num)
        print('success_missions_num: ', AirFogSimEvaluation.success_missions_num)
        print('failed_missions_num: ', AirFogSimEvaluation.failed_missions_num)
        print('early_failed_missions_num: ', AirFogSimEvaluation.early_failed_missions_num)

        # 信道速率指标
        print('信道速率')
        print('avg_trans_rate: ', AirFogSimEvaluation.avg_trans_rate)
        print('avg_V2U_trans_rate: ', AirFogSimEvaluation.avg_V2U_trans_rate)
        print('avg_V2I_trans_rate: ', AirFogSimEvaluation.avg_V2I_trans_rate)
        print('avg_U2I_trans_rate: ', AirFogSimEvaluation.avg_U2I_trans_rate)

        # reward
        print('奖励')
        print('sum_reward: ', AirFogSimEvaluation.sum_reward)
        print('sum_success_reward: ', AirFogSimEvaluation.sum_success_reward)
        print('avg_reward: ', AirFogSimEvaluation.avg_reward)
        print('avg_success_reward: ', AirFogSimEvaluation.avg_success_reward)

        # ratio
        print('比率')
        print('finish_mission_num: ', AirFogSimEvaluation.finish_mission_num)
        print('completion_ratio: ', AirFogSimEvaluation.completion_ratio)

        print('sum_weight: ', AirFogSimEvaluation.sum_weight)
        print('sum_weighted_acceleration_ratio: ', AirFogSimEvaluation.sum_weighted_acceleration_ratio)
        print('weighted_acceleration_ratio: ', AirFogSimEvaluation.weighted_acceleration_ratio)

        print('sum_acceleration_ratio: ', AirFogSimEvaluation.sum_acceleration_ratio)
        print('acceleration_ratio: ', AirFogSimEvaluation.acceleration_ratio)

        # optimization objectives
        print('优化目标')
        print('sum_executing_time: ', AirFogSimEvaluation.sum_executing_time)
        print('sum_sensing_time: ', AirFogSimEvaluation.sum_sensing_time)
        print('avg_floating_time: ', AirFogSimEvaluation.avg_floating_time)
        print('avg_executing_time: ', AirFogSimEvaluation.avg_executing_time)

        print('\n')
