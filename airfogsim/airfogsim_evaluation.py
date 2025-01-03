import math

# from airfogsim_env import AirFogSimEnv
# from airfogsim_algorithm import BaseAlgorithmModule,NVHAUAlgorithmModule
import os
import sys
import matplotlib.pyplot as plt


class AirFogSimEvaluation:
    def __init__(self):
        self.initOrResetStepIndicators()
        self.initOrResetStepRecords()
        self.initOrResetEpisodeRecords()
        # Path to save image
        self.base_path = "../evaluation/"
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

    def initOrResetStepIndicators(self):
        # 1.仿真时间
        self.simulation_time = 0
        self.step_num = 0

        # 2.过程监控
        self.to_generate_missions_num = 0
        self.executing_missions_num = 0
        self.success_missions_num = 0
        self.failed_missions_num = 0
        self.early_failed_missions_num = 0

        self.traffic_step_generate_num = 0
        self.traffic_step_allocate_num = 0
        self.traffic_step_success_num = 0
        self.traffic_step_fail_num = 0
        self.traffic_step_early_fail_num = 0

        # 3.信道速率指标
        self.avg_trans_rate = 0
        self.avg_V2U_trans_rate = 0
        self.avg_V2I_trans_rate = 0
        self.avg_U2I_trans_rate = 0
        self.V2U_data_trans_list= []
        self.V2I_data_trans_list = []
        self.U2I_data_trans_list = []
        self.V2U_link_using = []
        self.V2I_link_using = []
        self.U2I_link_using = []


        # 4.reward
        self.sum_reward = 0
        self.sum_success_reward = 0
        self.avg_reward = 0
        self.avg_success_reward = 0

        # 5.ratio
        # 5.1 completion ratio
        self.finish_mission_num = 0
        self.completion_ratio = 0

        self.finish_mission_num_on_vehicle=0
        self.success_mission_num_on_vehicle=0
        self.completion_ratio_on_vehicle=0

        self.finish_mission_num_on_UAV=0
        self.success_mission_num_on_UAV = 0
        self.completion_ratio_on_UAV = 0

        # 5.2 weighted acceleration ratio
        self.sum_weight = 0
        self.sum_weighted_acceleration_ratio = 0
        self.weighted_acceleration_ratio = 0

        # 5.3 acceleration ratio
        self.sum_acceleration_ratio = 0
        self.acceleration_ratio = 0

        # 6.optimization objectives
        self.sum_executing_time = 0
        self.sum_sensing_time = 0
        self.avg_floating_time = 0
        self.avg_executing_time = 0

    def initOrResetStepRecords(self):
        self.step_num=0
        # 2.过程监控
        self.step_to_generate_missions_num = []
        self.step_executing_missions_num = []
        self.step_success_missions_num = []
        self.step_failed_missions_num = []
        self.step_early_failed_missions_num = []

        self.step_generate_num = []
        self.step_allocate_num = []
        self.step_success_num = []
        self.step_fail_num = []
        self.step_early_fail_num = []

        # 3.信道速率指标
        self.step_avg_trans_rate = []
        self.step_avg_V2U_trans_rate = []
        self.step_avg_V2I_trans_rate = []
        self.step_avg_U2I_trans_rate = []

        # 4.reward
        self.step_avg_reward = []
        self.step_avg_success_reward = []

        # 5.ratio
        # 5.1 completion ratio0
        self.step_completion_ratio = []
        self.step_completion_ratio_on_vehicle = []
        self.step_completion_ratio_on_UAV = []

        # 5.2 weighted acceleration ratio
        self.step_weighted_acceleration_ratio = []

        # 5.3 acceleration ratio
        self.step_acceleration_ratio = []

        # 6.optimization objectives
        self.step_avg_floating_time = []
        self.step_avg_executing_time = []

    def initOrResetEpisodeRecords(self):
        self.episode_num=0
        # 2.过程监控
        self.episode_to_generate_missions_num = []
        self.episode_executing_missions_num = []
        self.episode_success_missions_num = []
        self.episode_failed_missions_num = []
        self.episode_early_failed_missions_num = []

        # 3.信道速率指标
        self.episode_avg_trans_rate = []
        self.episode_avg_V2U_trans_rate = []
        self.episode_avg_V2I_trans_rate = []
        self.episode_avg_U2I_trans_rate = []

        # 4.reward
        self.episode_avg_reward = []
        self.episode_avg_success_reward = []

        # 5.ratio
        # 5.1 completion ratio
        self.episode_completion_ratio = []
        self.episode_completion_ratio_on_vehicle = []
        self.episode_completion_ratio_on_UAV = []

        # 5.2 weighted acceleration ratio
        self.episode_weighted_acceleration_ratio = []

        # 5.3 acceleration ratio
        self.episode_acceleration_ratio = []

        # 6.optimization objectives
        self.episode_avg_floating_time = []
        self.episode_avg_executing_time = []

    def updateEvaluationIndicators(self, env, algorithm_module):
        step_reward, step_punish, step_sum_reward = algorithm_module.getRewardByMission(
            env)  # success reward, fail punish, sum of reward and punish
        to_generate_missions_num = env.mission_manager.getToGenerateMissionNum()
        executing_missions_num = env.mission_manager.getExecutingMissionNum()
        success_missions_num = env.mission_manager.getSuccessMissionNum()
        failed_missions_num = env.mission_manager.getFailedMissionNum()
        early_failed_missions_num = env.mission_manager.getEarlyFailedMissionNum()
        sum_over_missions = success_missions_num + failed_missions_num + early_failed_missions_num

        last_step_succ_mission_infos = algorithm_module.missionScheduler.getLastStepSuccMissionInfos(env)
        last_step_fail_mission_infos = algorithm_module.missionScheduler.getLastStepFailMissionInfos(env)
        last_step_early_fail_mission_infos = algorithm_module.missionScheduler.getLastStepEarlyFailMissionInfos(env)

        # 1.仿真时间
        self.simulation_time = env.simulation_time

        # 2.过程监控
        self.to_generate_missions_num = to_generate_missions_num
        self.executing_missions_num = executing_missions_num
        self.success_missions_num = success_missions_num
        self.early_failed_missions_num = early_failed_missions_num
        self.failed_missions_num = failed_missions_num

        self.finish_mission_num = sum_over_missions


        self.traffic_step_generate_num,self.traffic_step_allocate_num = env.getMissionEvaluationIndicators()
        self.traffic_step_success_num = len(last_step_succ_mission_infos)
        self.traffic_step_fail_num = len(last_step_fail_mission_infos)
        self.traffic_step_early_fail_num = len(last_step_early_fail_mission_infos)

        # 3.信道速率指标
        self.avg_trans_rate = env.getChannelAvgRate()
        self.avg_V2U_trans_rate = env.getChannelAvgRate('V2U')
        self.avg_V2I_trans_rate = env.getChannelAvgRate('V2I')
        self.avg_U2I_trans_rate = env.getChannelAvgRate('U2I')
        self.V2U_data_trans_list = env.getChannelTransDataHistory('V2U')
        self.V2I_data_trans_list = env.getChannelTransDataHistory('V2I')
        self.U2I_data_trans_list = env.getChannelTransDataHistory('U2I')
        self.V2U_link_using = env.getLinkUsingHistory('V2U')
        self.V2I_link_using = env.getLinkUsingHistory('V2I')
        self.U2I_link_using = env.getLinkUsingHistory('U2I')

        # 4.reward
        self.sum_reward += step_sum_reward
        self.sum_success_reward += step_reward
        self.avg_reward = self.sum_reward / self.finish_mission_num if self.finish_mission_num != 0 else 0
        self.avg_success_reward = self.sum_success_reward / self.success_missions_num if self.success_missions_num != 0 else 0


        # 5.ratio
        # 5.1 completion ratio (all)
        self.completion_ratio = env.mission_manager.getMissionCompletionRatio()[0]  # return: (ratio, mission_num)
        for mission_info in last_step_succ_mission_infos:
            TTL = mission_info['mission_deadline']
            duration = mission_info['mission_duration_sum']
            finish_time = mission_info['mission_finish_time']
            arrival_time = mission_info['mission_arrival_time']

            weight = math.log(10, 1 + TTL - duration)
            ratio = (TTL - duration) / (finish_time - arrival_time - duration)

            self.sum_weight += weight
            self.sum_weighted_acceleration_ratio += weight * ratio
            self.sum_acceleration_ratio += ratio

        self.weighted_acceleration_ratio = self.sum_weighted_acceleration_ratio / self.sum_weight if self.sum_weight > 0.0 else 0
        self.acceleration_ratio = self.sum_acceleration_ratio / self.success_missions_num if self.success_missions_num > 0 else 0

        # 5.2 completion ratio (Classified by type, not consider early fail)
        for mission_info in last_step_succ_mission_infos:
            appointed_node_id=mission_info['appointed_node_id']
            node_type=self.__getNodeTypeById(appointed_node_id)
            if node_type=='V':
                self.finish_mission_num_on_vehicle+=1
                self.success_mission_num_on_vehicle+=1
            elif node_type=='U':
                self.finish_mission_num_on_UAV += 1
                self.success_mission_num_on_UAV += 1
            else:
                raise TypeError('Node type is invalid')

        for mission_info in last_step_fail_mission_infos:
            appointed_node_id=mission_info['appointed_node_id']
            node_type=self.__getNodeTypeById(appointed_node_id)
            if node_type=='V':
                self.finish_mission_num_on_vehicle+=1
            elif node_type=='U':
                self.finish_mission_num_on_UAV += 1
            else:
                raise TypeError('Node type is invalid')

        self.completion_ratio_on_vehicle=self.success_mission_num_on_vehicle / self.finish_mission_num_on_vehicle if self.finish_mission_num_on_vehicle > 0 else 0
        self.completion_ratio_on_UAV=self.success_mission_num_on_UAV / self.finish_mission_num_on_UAV if self.finish_mission_num_on_UAV > 0 else 0

        # optimization objectives
        for mission_info in last_step_succ_mission_infos:
            duration = mission_info['mission_duration_sum']
            finish_time = mission_info['mission_finish_time']
            arrival_time = mission_info['mission_arrival_time']
            self.sum_executing_time += finish_time - arrival_time
            self.sum_sensing_time += duration

        self.avg_floating_time = (self.sum_executing_time - self.sum_sensing_time) / self.success_missions_num if self.success_missions_num > 0 else 0
        self.avg_executing_time = self.sum_executing_time / self.success_missions_num if self.success_missions_num > 0 else 0


    def printEvaluation(self):
        # 仿真时间
        print('simulation_time: ', self.simulation_time)

        # 过程监控
        print('过程监控')
        print('to_generate_missions_num: ', self.to_generate_missions_num)
        print('executing_missions_num: ', self.executing_missions_num)
        print('success_missions_num: ', self.success_missions_num)
        print('failed_missions_num: ', self.failed_missions_num)
        print('early_failed_missions_num: ', self.early_failed_missions_num)

        print('traffic_step_generate_num: ', self.traffic_step_generate_num)
        print('traffic_step_allocate_num: ', self.traffic_step_allocate_num)
        print('traffic_step_success_num: ', self.traffic_step_success_num)
        print('traffic_step_fail_num: ', self.traffic_step_fail_num)
        print('traffic_step_early_fail_num: ', self.traffic_step_early_fail_num)

        # 信道速率指标
        print('信道速率')
        print('avg_trans_rate: ', self.avg_trans_rate)
        print('avg_V2U_trans_rate: ', self.avg_V2U_trans_rate)
        print('avg_V2I_trans_rate: ', self.avg_V2I_trans_rate)
        print('avg_U2I_trans_rate: ', self.avg_U2I_trans_rate)


        # reward
        print('奖励')
        print('sum_reward: ', self.sum_reward)
        print('sum_success_reward: ', self.sum_success_reward)
        print('avg_reward: ', self.avg_reward)
        print('avg_success_reward: ', self.avg_success_reward)

        # ratio
        print('比率')
        print('finish_mission_num: ', self.finish_mission_num)
        print('completion_ratio: ', self.completion_ratio)

        print('finish_mission_num_on_vehicle: ', self.finish_mission_num_on_vehicle)
        print('success_mission_num_on_vehicle: ', self.success_mission_num_on_vehicle)
        print('completion_ratio_on_vehicle: ', self.completion_ratio_on_vehicle)

        print('finish_mission_num_on_UAV: ', self.finish_mission_num_on_UAV)
        print('success_mission_num_on_UAV: ', self.success_mission_num_on_UAV)
        print('completion_ratio_on_UAV: ', self.completion_ratio_on_UAV)


        print('sum_weight: ', self.sum_weight)
        print('sum_weighted_acceleration_ratio: ', self.sum_weighted_acceleration_ratio)
        print('weighted_acceleration_ratio: ', self.weighted_acceleration_ratio)

        print('sum_acceleration_ratio: ', self.sum_acceleration_ratio)
        print('acceleration_ratio: ', self.acceleration_ratio)

        # optimization objectives
        print('优化目标')
        print('sum_executing_time: ', self.sum_executing_time)
        print('sum_sensing_time: ', self.sum_sensing_time)
        print('avg_floating_time: ', self.avg_floating_time)
        print('avg_executing_time: ', self.avg_executing_time)

        print('\n')

    def toFile(self, episode):
        file_dir = self.base_path+"evaluation.log"
        # 将print输出重定向到文件
        sys.stdout = open(file_dir, 'a')
        # episode
        print('episode: ', episode)
        # 过程监控
        print('过程监控')
        print('to_generate_missions_num: ', self.to_generate_missions_num)
        print('executing_missions_num: ', self.executing_missions_num)
        print('success_missions_num: ', self.success_missions_num)
        print('failed_missions_num: ', self.failed_missions_num)
        print('early_failed_missions_num: ', self.early_failed_missions_num)

        print('traffic_step_generate_num: ', self.traffic_step_generate_num)
        print('traffic_step_allocate_num: ', self.traffic_step_allocate_num)
        print('traffic_step_success_num: ', self.traffic_step_success_num)
        print('traffic_step_fail_num: ', self.traffic_step_fail_num)
        print('traffic_step_early_fail_num: ', self.traffic_step_early_fail_num)

        # 信道速率指标
        print('信道速率')
        print('avg_trans_rate: {:.2f}'.format(self.avg_trans_rate))
        print('avg_V2U_trans_rate: {:.2f}'.format(self.avg_V2U_trans_rate))
        print('avg_V2I_trans_rate: {:.2f}'.format(self.avg_V2I_trans_rate))
        print('avg_U2I_trans_rate: {:.2f}'.format(self.avg_U2I_trans_rate))

        # 奖励
        print('奖励')
        print('avg_reward: {:.2f}'.format(self.avg_reward))
        print('avg_success_reward: {:.2f}'.format(self.avg_success_reward))

        # 比率
        print('比率')
        print('completion_ratio: {:.2f}'.format(self.completion_ratio))
        print('completion_ratio_on_vehicle: ', self.completion_ratio_on_vehicle)
        print('completion_ratio_on_UAV: ', self.completion_ratio_on_UAV)

        print('weighted_acceleration_ratio: {:.2f}'.format(self.weighted_acceleration_ratio))
        print('acceleration_ratio: {:.2f}'.format(self.acceleration_ratio))


        # 优化目标
        print('优化目标')
        print('avg_floating_time: {:.2f}s'.format(self.avg_floating_time))
        print('avg_executing_time: {:.2f}s'.format(self.avg_executing_time))

        print('\n')
        # 恢复标准输出
        sys.stdout = sys.__stdout__

    def addToStepRecord(self):
        self.step_num += 1
        # 过程监控
        self.step_to_generate_missions_num.append(self.to_generate_missions_num)
        self.step_executing_missions_num.append(self.executing_missions_num)
        self.step_success_missions_num.append(self.success_missions_num)
        self.step_failed_missions_num.append(self.failed_missions_num)
        self.step_early_failed_missions_num.append(self.early_failed_missions_num)

        self.step_generate_num.append(self.traffic_step_generate_num)
        self.step_allocate_num.append(self.traffic_step_allocate_num)
        self.step_success_num.append(self.traffic_step_success_num)
        self.step_fail_num.append(self.traffic_step_fail_num)
        self.step_early_fail_num.append(self.traffic_step_early_fail_num)

        # 信道速率指标
        self.step_avg_trans_rate.append(self.avg_trans_rate)
        self.step_avg_V2U_trans_rate.append(self.avg_V2U_trans_rate)
        self.step_avg_V2I_trans_rate.append(self.avg_V2I_trans_rate)
        self.step_avg_U2I_trans_rate.append(self.avg_U2I_trans_rate)

        # 4.reward
        self.step_avg_reward.append(self.avg_reward)
        self.step_avg_success_reward.append(self.avg_success_reward)

        # 5.ratio
        self.step_completion_ratio.append(self.completion_ratio)
        self.step_completion_ratio_on_vehicle.append(self.completion_ratio_on_vehicle)
        self.step_completion_ratio_on_UAV.append(self.completion_ratio_on_UAV)

        self.step_weighted_acceleration_ratio.append(self.weighted_acceleration_ratio)
        self.step_acceleration_ratio.append(self.acceleration_ratio)

        # 6.optimization objectives
        self.step_avg_floating_time.append(self.avg_floating_time)
        self.step_avg_executing_time.append(self.avg_executing_time)

    def drawAndResetStepRecord(self,episode):
        path_to_save = self.base_path + f'episode_{episode}/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        # 过程监控
        # 累计任务
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_to_generate_missions_num,label='To generate', color='gray')
        plt.plot(x_indices, self.step_executing_missions_num,label='Executing', color='blueviolet')
        plt.plot(x_indices, self.step_success_missions_num,label='Success', color='limegreen')
        plt.plot(x_indices, self.step_failed_missions_num,label='Fail', color='red')
        plt.plot(x_indices, self.step_early_failed_missions_num,label='Not Allocate', color='orange')
        plt.title('Accumulate Process Monitoring')
        plt.xlabel('Step')
        plt.ylabel('The Number of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save+'AccumulateProcessMonitoring.png')
        plt.close()

        # 时隙任务
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 两行三列
        # 子图1: Generate
        axes[0, 0].plot(x_indices, self.step_generate_num, label='Generate', color='gray')
        axes[0, 0].set_title('Generate')
        axes[0, 0].set_ylabel('The Number')
        axes[0, 0].grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        axes[0, 0].legend()
        # 子图2: Allocate
        axes[0, 1].plot(x_indices, self.step_allocate_num, label='Allocate', color='blueviolet')
        axes[0, 1].set_title('Allocate')
        axes[0, 1].grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        axes[0, 1].legend()
        # 子图3: Success
        axes[0, 2].plot(x_indices, self.step_success_num, label='Success', color='limegreen')
        axes[0, 2].set_title('Success')
        axes[0, 2].grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        axes[0, 2].legend()
        # 子图4: Fail
        axes[1, 0].plot(x_indices, self.step_fail_num, label='Fail', color='red')
        axes[1, 0].set_title('Fail')
        axes[1, 0].set_ylabel('The Number')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        axes[1, 0].legend()
        # 子图5: Not Allocate
        axes[1, 1].plot(x_indices, self.step_early_fail_num, label='Not Allocate', color='orange')
        axes[1, 1].set_title('Not Allocate')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        axes[1, 1].legend()
        # 子图6: 留空或补充说明
        axes[1, 2].axis('off')  # 禁用第六个子图
        # axes[1, 2].text(0.5, 0.5, "Additional Info", ha='center', va='center', fontsize=12, color='gray')
        # 设置全局标题
        fig.suptitle('Timeslot Process Monitoring', fontsize=16)
        # 保存图像
        plt.savefig(path_to_save + 'TimeslotProcessMonitoring.png')
        plt.close()


        # 信道速率指标
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_avg_trans_rate, label='Avg', color='blueviolet')
        plt.plot(x_indices, self.step_avg_V2U_trans_rate, label='V2U', color='limegreen')
        plt.plot(x_indices, self.step_avg_V2I_trans_rate, label='V2I', color='red')
        plt.plot(x_indices, self.step_avg_U2I_trans_rate, label='U2I', color='orange')
        plt.title('Channel Rate')
        plt.xlabel('Step')
        plt.ylabel('The Average Rate of Channel')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'ChannelRate.png')
        plt.close()

        # 信道使用情况
        x_indices = list(range(1, len(self.V2U_link_using) + 1))
        plt.figure()
        plt.plot(x_indices, self.V2U_link_using, label='V2U', color='limegreen')
        plt.plot(x_indices, self.V2I_link_using, label='V2I', color='red')
        plt.plot(x_indices, self.U2I_link_using, label='U2I', color='orange')
        plt.title('Link Using')
        plt.xlabel('SimulationStep')
        plt.ylabel('Link Num in Use')
        plt.grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'LinkUsing.png')
        plt.close()

        # data trans
        # V2U
        x_indices = list(range(1, len(self.V2U_data_trans_list) + 1))
        plt.figure()
        plt.plot(x_indices, self.V2U_data_trans_list, color='orange')
        plt.title('V2U Data Trans')
        plt.xlabel('Trans Step')
        plt.ylabel('Data Size')
        plt.grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        plt.savefig(path_to_save + 'V2U_data_trans.png')
        plt.close()
        # V2I
        x_indices = list(range(1, len(self.V2I_data_trans_list) + 1))
        plt.figure()
        plt.plot(x_indices, self.V2I_data_trans_list, color='orange')
        plt.title('V2I Data Trans')
        plt.xlabel('Trans Step')
        plt.ylabel('Data Size')
        plt.grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        plt.savefig(path_to_save + 'V2I_data_trans.png')
        plt.close()
        # U2I
        x_indices = list(range(1, len(self.U2I_data_trans_list) + 1))
        plt.figure()
        plt.plot(x_indices, self.U2I_data_trans_list, color='orange')
        plt.title('U2I Data Trans')
        plt.xlabel('Trans Step')
        plt.ylabel('Data Size')
        plt.grid(True, which='both', linestyle='--', linewidth=0.1, alpha=0.7)
        plt.savefig(path_to_save + 'U2I_data_trans.png')
        plt.close()

        # reward
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_avg_reward, label='Avg', color='orange')
        plt.plot(x_indices, self.step_avg_success_reward, label='Suc', color='limegreen')
        plt.title('Reward')
        plt.xlabel('Step')
        plt.ylabel('The Average Reward of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'Reward.png')
        plt.close()

        # completion ratio
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_completion_ratio,label='Sum', color='blueviolet')
        plt.plot(x_indices, self.step_completion_ratio_on_vehicle,label='Vehicle', color='limegreen')
        plt.plot(x_indices, self.step_completion_ratio_on_UAV,label='UAV', color='orange')
        plt.title('Completion Ratio')
        plt.xlabel('Step')
        plt.ylabel('The Completion Ratio of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'CompletionRatio.png')
        plt.close()

        # acceleration ratio
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_weighted_acceleration_ratio,label='Weighted', color='orange')
        plt.plot(x_indices, self.step_acceleration_ratio,label='Not Weighted', color='limegreen')
        plt.title('Acceleration Ratio')
        plt.xlabel('Step')
        plt.ylabel('The Acceleration Ratio of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'AccelerationRatio.png')
        plt.close()

        # optimization objectives
        x_indices = list(range(1, self.step_num + 1))
        plt.figure()
        plt.plot(x_indices, self.step_avg_floating_time,label='Floating', color='orange')
        plt.plot(x_indices, self.step_avg_executing_time,label='Executing', color='limegreen')
        plt.title('Time')
        plt.xlabel('Step')
        plt.ylabel('The Related Time of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'Time.png')
        plt.close()


        self.initOrResetStepIndicators()
        self.initOrResetStepRecords()

    def addToEpisodeRecord(self):
        self.episode_num+=1
        # 过程监控
        self.episode_to_generate_missions_num.append(self.to_generate_missions_num)
        self.episode_executing_missions_num.append(self.executing_missions_num)
        self.episode_success_missions_num.append(self.success_missions_num)
        self.episode_failed_missions_num.append(self.failed_missions_num)
        self.episode_early_failed_missions_num.append(self.early_failed_missions_num)

        # 信道速率指标
        self.episode_avg_trans_rate.append(self.avg_trans_rate)
        self.episode_avg_V2U_trans_rate.append(self.avg_V2U_trans_rate)
        self.episode_avg_V2I_trans_rate.append(self.avg_V2I_trans_rate)
        self.episode_avg_U2I_trans_rate.append(self.avg_U2I_trans_rate)

        # reward
        self.episode_avg_reward.append(self.avg_reward)
        self.episode_avg_success_reward.append(self.avg_success_reward)

        # ratio
        self.episode_completion_ratio.append(self.completion_ratio)
        self.episode_completion_ratio_on_vehicle.append(self.completion_ratio_on_vehicle)
        self.episode_completion_ratio_on_UAV.append(self.completion_ratio_on_UAV)

        self.episode_weighted_acceleration_ratio.append(self.weighted_acceleration_ratio)
        self.episode_acceleration_ratio.append(self.acceleration_ratio)

        # optimization objectives
        self.episode_avg_floating_time.append(self.avg_floating_time)
        self.episode_avg_executing_time.append(self.avg_executing_time)

    def drawAndResetEpisodeRecord(self):
        path_to_save = self.base_path + 'final/'
        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)

        # 横轴为步数
        x_indices = list(range(1,self.episode_num+1))

        # 过程监控
        plt.figure()
        plt.plot(x_indices, self.episode_to_generate_missions_num, label='To generate', color='gray')
        plt.plot(x_indices, self.episode_executing_missions_num, label='Executing', color='blueviolet')
        plt.plot(x_indices, self.episode_success_missions_num, label='Success', color='limegreen')
        plt.plot(x_indices, self.episode_failed_missions_num, label='Fail', color='red')
        plt.plot(x_indices, self.episode_early_failed_missions_num, label='Not Allocate', color='orange')
        plt.title('Process Monitoring')
        plt.xlabel('Episode')
        plt.ylabel('The Number of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'ProcessMonitoring.png')
        plt.close()

        # 信道速率指标
        plt.figure()
        plt.plot(x_indices, self.episode_avg_trans_rate, label='Avg', color='blueviolet')
        plt.plot(x_indices, self.episode_avg_V2U_trans_rate, label='V2U', color='limegreen')
        plt.plot(x_indices, self.episode_avg_V2I_trans_rate, label='V2I', color='red')
        plt.plot(x_indices, self.episode_avg_U2I_trans_rate, label='U2I', color='orange')
        plt.title('Channel Rate')
        plt.xlabel('Episode')
        plt.ylabel('The Average Rate of Channel')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'ChannelRate.png')
        plt.close()

        # reward
        plt.figure()
        plt.plot(x_indices, self.episode_avg_reward, label='Avg', color='orange')
        plt.plot(x_indices, self.episode_avg_success_reward, label='Suc', color='limegreen')
        plt.title('Reward')
        plt.xlabel('Episode')
        plt.ylabel('The Average Reward of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'Reward.png')
        plt.close()

        # completion ratio
        plt.figure()
        plt.plot(x_indices, self.episode_completion_ratio, label='Sum', color='blueviolet')
        plt.plot(x_indices, self.episode_completion_ratio_on_vehicle, label='Vehicle', color='limegreen')
        plt.plot(x_indices, self.episode_completion_ratio_on_UAV, label='UAV', color='orange')
        plt.title('Completion Ratio')
        plt.xlabel('Episode')
        plt.ylabel('The Completion Ratio of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'CompletionRatio.png')
        plt.close()

        # acceleration ratio
        plt.figure()
        plt.plot(x_indices, self.episode_weighted_acceleration_ratio, label='Weighted Acc', color='orange')
        plt.plot(x_indices, self.episode_acceleration_ratio, label='Not Weighted Acc', color='limegreen')
        plt.title('Acceleration Ratio')
        plt.xlabel('Episode')
        plt.ylabel('The Acceleration Ratio of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'AccelerationRatio.png')
        plt.close()

        # optimization objectives
        plt.figure()
        plt.plot(x_indices, self.episode_avg_floating_time, label='Floating', color='orange')
        plt.plot(x_indices, self.episode_avg_executing_time, label='Executing', color='limegreen')
        plt.title('Time')
        plt.xlabel('Episode')
        plt.ylabel('The Related Time of Missions')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend()
        plt.savefig(path_to_save + 'Time.png')
        plt.close()

        self.initOrResetStepIndicators()
        self.initOrResetStepRecords()
        self.initOrResetEpisodeRecords()

    def __getNodeTypeById(self,node_id):
        node_id=node_id.capitalize()
        assert node_id[0] in ['V','R','U','C'],f'Invalid node type of {node_id}'
        if node_id[0] == 'V':
            return 'V'
        elif node_id[0] == 'R':
            return 'R'
        elif node_id[0] == 'U':
            return 'U'
        elif node_id[0] == 'C':
            return 'C'
        else:
            return None

