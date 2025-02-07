import argparse
import random

import torch
from pprint import pprint

from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.algorithm.crowdsensing.DDQN.DDQN_env import DDQN_Env
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
from .ReplayBuffer import BaseReplayBuffer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
cuda_num = torch.cuda.device_count()
# device = "cpu"
base_dir="/home/chenjiarui/data/project/crowdsensing"

if cuda_num > 0:
    cuda_list = list(range(cuda_num))
    DDQN_device = f"cuda:{cuda_list[cuda_num-1]}"
else:
    DDQN_device = "cpu"

print('torch_version: ',torch.__version__)
print('gpu_num: ',cuda_num)
print('device: ',DDQN_device)



def parseDDQNTrainArgs():
    parser = argparse.ArgumentParser(description='DDQN train arguments')
    parser.add_argument('--buffer_size', type=int, default=1000)  # 经验池容量
    parser.add_argument('--learning_rate', type=float, default=5e-4)  # 学习率
    parser.add_argument('--gamma', type=float, default=0.96)  # 折扣因子
    parser.add_argument('--epsilon', type=float, default=0.5)  # 探索系数
    parser.add_argument('--eps_end', type=float, default=0.01)  # 最低探索系数
    parser.add_argument('--eps_dec', type=float, default=2e-3)  # 探索系数衰减率
    parser.add_argument('--target_update', type=int, default=200)  # 目标网络的参数的更新频率
    parser.add_argument('--batch_size', type=int, default=16)  # 每次训练选取的经验数量32
    parser.add_argument('--train_min_size', type=int, default=200)  # 经验池超过200后再训练(train_min_size>batch_size)
    parser.add_argument('--tau', type=float, default=0.01)  # 目标网络软更新平滑因子(训练策略网络权重)
    parser.add_argument('--smooth_factor', type=float, default=0.99)  # 最大q值平滑因子（旧值权重）
    parser.add_argument('--device', type=str, default=DDQN_device)  # 训练设备(GPU/CPU)
    parser.add_argument('--model_base_dir', type=str, default=f"./models")  # 模型文件路径

    args = parser.parse_args()
    return args


def parseDDQNDimArgs():
    # [id, type, is_mission_node, is_schedulable, x, y, z]
    dim_node = 7
    # [sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]
    dim_mission = 10
    # [node_id,id,type, accuracy]
    dim_sensor = 4
    max_sensors = 6
    m_u = 15
    m_v = 100
    m_r = 5
    dim_states = dim_node * (m_u + m_v + m_r) + dim_mission + dim_sensor * max_sensors * (m_u + m_v)
    dim_actions = max_sensors * (m_v + m_u)

    parser = argparse.ArgumentParser(description='D3QN dimension arguments')
    parser.add_argument('--dim_node', type=int, default=dim_node)  # Dimension of nodes (UAV/Veh/RSU)
    parser.add_argument('--dim_mission', type=int, default=dim_mission)  # Dimension of mission
    parser.add_argument('--dim_sensor', type=int, default=dim_sensor)  # Dimension of sensor
    parser.add_argument('--max_sensors', type=int, default=max_sensors)  # Sensors on each mission node
    parser.add_argument('--m_u', type=int, default=m_u)  # Maximum UAVs
    parser.add_argument('--m_v', type=int, default=m_v)  # Maximum vehicles
    parser.add_argument('--m_r', type=int, default=m_r)  # Maximum RSUs

    parser.add_argument('--dim_states', type=int, default=dim_states)  # Dimension of states
    parser.add_argument('--dim_hiddens', type=float, default=512)  # Dimension of hidden layer
    parser.add_argument('--dim_value', type=float, default=128)  # Dimension of value sub layer
    parser.add_argument('--dim_advantages', type=float, default=128)  # Dimension of advantages sub layer
    parser.add_argument('--dim_actions', type=int, default=dim_actions)  # Dimension of states
    args = parser.parse_args()
    return args


class DDQN_Train_AlgorithmModule(BaseAlgorithmModule):
    """
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleMission: 
        Mission: Missions assigned to both vehicles and UAVs, decided by DDQN model.
        Sensor: Decided by DDQN model.
    scheduleReturning: Relay(only for task assigned to vehicle): select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct: select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: 
        UAV: Fly to next position in route list and stay for a period of time.
    '''

    class TaskAllocationReplayBuffer(BaseReplayBuffer):
        def __init__(self):
            # 创建一个字典，长度不限
            super().__init__()

        def __expToFlattenArray(self, exp):
            state = exp['state']
            mask = exp['mask']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            next_mask = exp['next_mask']
            done = exp['done']
            return np.array(state), np.array(mask), np.array(action), np.array(reward), np.array(
                next_state), np.array(next_mask), np.array(done)

        def add(self, exp_id, state, mask, action, reward=None, next_state=None, next_mask=None,
                done=None):
            self.buffer[exp_id] = {'state': state, 'mask': mask, 'action': action, 'reward': reward,
                                   'next_state': next_state, 'next_mask': next_mask, 'done': done}

        def setNextState(self, exp_id, next_state, next_mask, done):
            # assert exp_id in self.buffer, "State_id is invalid."
            if exp_id not in self.buffer:
                return

            self.buffer[exp_id]['next_state'] = next_state
            self.buffer[exp_id]['next_mask'] = next_mask
            self.buffer[exp_id]['done'] = done

        def completeAndPopExperience(self, exp_id, reward):
            assert exp_id in self.buffer, "exp_id is invalid."
            if self.buffer[exp_id]['next_state'] is None:
                return None

            self.buffer[exp_id]['reward'] = reward
            packed_exp = self.__expToFlattenArray(self.buffer[exp_id])
            del self.buffer[exp_id]
            return packed_exp

        def size(self):
            return super().size()

        def clear(self):
            super().clear()


    def __init__(self):
        super().__init__()
        self.algorithm_module_tag = "DDQN_Train"
        print('algorithm: ', self.algorithm_module_tag)

    def initialize(self, env: AirFogSimEnv, config={}, last_episode=None,final=False):
        """Initialize the algorithm with the environment. Including setting the task generation model, setting the reward model, etc.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.rewardScheduler.setModel(env, 'REWARD',
                                      '5 * log(10, 1 + (_mission_deadline-_mission_duration_sum)) * (1 / (1 + exp(-(_mission_deadline-_mission_duration_sum) / (_mission_finish_time - _mission_arrival_time-_mission_duration_sum))) - 1 / (1 + exp(-1)))')
        self.rewardScheduler.setModel(env, 'PUNISH', '-1')

        self.max_simulation_time = env.max_simulation_time
        self.min_position_x, self.max_position_x = self.trafficScheduler.getMapRange(env, 'X')
        self.min_position_y, self.max_position_y = self.trafficScheduler.getMapRange(env, 'Y')
        self.min_position_z, self.max_position_z = self.trafficScheduler.getMapRange(env, 'Z')
        self.min_UAV_speed, self.max_UAV_speed = self.trafficScheduler.getConfig(env, 'UAV_speed_range')
        self.max_n_vehicles = self.trafficScheduler.getConfig(env, 'max_n_vehicles')
        self.max_n_UAVs = self.trafficScheduler.getConfig(env, 'max_n_UAVs')
        self.max_n_RSUs = self.trafficScheduler.getConfig(env, 'max_n_RSUs')
        self.max_mission_size = self.missionScheduler.getConfig(env, 'mission_size_range')[1]
        self.max_energy = env.energy_manager.getConfig('initial_energy_range')[0]
        self.TA_distance_Veh=self.missionScheduler.getConfig(env, 'TA_distance_Veh')
        self.TA_distance_UAV = self.missionScheduler.getConfig(env, 'TA_distance_UAV')
        self.node_type_dict = {
            'U': 1,
            'V': 2,
            'I': 3,
            'C': 4,
        }
        self.node_priority = {
            'U': 1,
            'V': 2,
            'I': 3,
        }

        self.DDQN_dim_args = parseDDQNDimArgs()
        self.DDQN_train_args = parseDDQNTrainArgs()
        self.DDQN_env = DDQN_Env(self.DDQN_dim_args, self.DDQN_train_args)
        if last_episode is not None and last_episode > 0:
            self.DDQN_env.loadModel(last_episode,final)


        self.last_mission_id = None  # Last allocated mission id,used in next state update
        self.ta_buffer = self.TaskAllocationReplayBuffer()

    def reset(self, env: AirFogSimEnv):
        self.last_mission_id = None  # Last allocated mission id,used in next state update
        self.ta_buffer.clear()

    def _encode_node_states(self, node_states, max_node_num, dim_state):
        # UAVs,Vehicles,RSUs
        # [id, type, is_mission_node, is_schedulable, x, y, z]
        # [1, 'U', True, True, 105.23, 568.15. 225.65]
        # 选取[id, type, is_mission_node, is_schedulable, x, y, z]

        encode_states = []

        # 删除可能超出最大节点数的节点（一般是因为车辆超出）
        if len(node_states) > max_node_num:
            to_delete_num = max_node_num - len(node_states)
            for i in range(len(node_states) - 1, -1, -1):
                if node_states[i][1] == 'V':
                    del node_states[i]
                if to_delete_num == 0:
                    break

        for node_state in node_states:
            id=node_state[0]
            node_type = self.node_type_dict.get(node_state[1], -1)
            is_mission_node = int(node_state[2])
            is_schedulable = int(node_state[3])
            position_x = (node_state[4] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (node_state[5] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (node_state[6] - self.min_position_z) / (self.max_position_z - self.min_position_z)  if (self.max_position_z - self.min_position_z) > 0 else 1

            state = [id, node_type, is_mission_node, is_schedulable, position_x, position_y, position_z]
            encode_states.append(state)

        # 补齐长度
        valid_node_num = len(encode_states)
        if valid_node_num < max_node_num:
            for _ in range(max_node_num - valid_node_num):
                encode_states.append([0 for _ in range(dim_state)])  # 补充零

        return np.array(encode_states)

    def _encode_mission_states(self, mission_states, max_mission_num, dim_state):
        # [sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]
        # ['U',0.8,50,20,120,5,120.25,262.05,553.25,100]
        # 选取[sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]

        encode_states = []
        for mission_state in mission_states:
            sensor_type = mission_state[0]
            accuracy = mission_state[1]
            return_size = mission_state[2] / self.max_mission_size
            arrival_time = mission_state[3] / self.max_simulation_time
            TTL = mission_state[4] / self.max_simulation_time
            duration = mission_state[5] / self.max_simulation_time
            position_x = (mission_state[6] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (mission_state[7] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (mission_state[8] - self.min_position_z) / (self.max_position_z - self.min_position_z)  if (self.max_position_z - self.min_position_z) > 0 else 1
            distance_threshold = mission_state[9] / (self.max_position_x - self.min_position_x)

            state = [sensor_type, accuracy, return_size, arrival_time, TTL, duration, position_x, position_y,
                     position_z, distance_threshold]
            encode_states.append(state)

        # 补齐长度
        valid_mission_num = len(encode_states)
        if valid_mission_num < max_mission_num:
            for _ in range(max_mission_num - valid_mission_num):
                encode_states.append([0 for _ in range(dim_state)])  # 补充零

        return np.array(encode_states)

    def _encode_sensor_states(self, sensor_states, max_sensor_node_num, node_sensor_num, dim_state):
        # [node_id,node_type,id,type, accuracy,candidate]
        # [1, 'U', 3, 2, 0.8, True]
        # 选取[node_id,id,type, accuracy]

        encode_states = []
        encode_mask = []

        # 删除可能超出最大节点数的节点（一般是因为车辆超出）
        if len(sensor_states) > max_sensor_node_num:
            to_delete_num = len(sensor_states) - max_sensor_node_num
            for i in range(len(sensor_states) - 1, -1, -1):
                if sensor_states[i][0][1] == 'V':
                    del sensor_states[i]
                    to_delete_num -= 1
                if to_delete_num == 0:
                    break

        # 1. 删除不需要的属性
        for node_group in sensor_states:
            node_state_group = []
            node_mask_group = []
            for sensor_state in node_group:
                node_id = sensor_state[0]
                node_type = sensor_state[1]
                id = sensor_state[2]
                type = sensor_state[3]
                accuracy = sensor_state[4]
                processed_state = [node_id, id, type, accuracy]
                node_state_group.append(processed_state)
                node_mask_group.append(sensor_state[-1])
            encode_states.append(node_state_group)
            encode_mask.append(node_mask_group)

        # 2. 补齐长度
        valid_node_num = len(encode_states)
        if valid_node_num < max_sensor_node_num:
            for _ in range(max_sensor_node_num - valid_node_num):
                node_state_group = []
                node_mask_group = []
                for _ in range(node_sensor_num):
                    node_state_group.append([0 for _ in range(dim_state)])  # 补充零
                    node_mask_group.append(False)  # 补充False
                encode_states.append(node_state_group)
                encode_mask.append(node_mask_group)

        return np.array(encode_states), np.array(encode_mask)


    def scheduleStep(self, env: AirFogSimEnv):
        """The algorithm logic. Should be implemented by the subclass.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        self.scheduleReturning(env)
        self.scheduleOffloading(env)
        self.scheduleCommunication(env)
        self.scheduleComputing(env)
        self.scheduleMission(env)
        self.scheduleTraffic(env)

    def train(self, env: AirFogSimEnv):
        self.updateTAExperience(env) #
        loss=self.DDQN_env.train()
        return loss

    def saveModel(self, episode,final):
        self.DDQN_env.saveModel(episode,final)

    def scheduleMission(self, env: AirFogSimEnv):
        """The mission scheduling logic.
        Mission: Missions assigned to both vehicles and UAVs, decided by DDQN model.
        Sensor: decided by DDQN model.

        Args:
            env (AirFogSimEnv): The environment object.

        """
        cur_time = self.trafficScheduler.getCurrentTime(env)
        traffic_interval = self.trafficScheduler.getTrafficInterval(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []

        # UAV_num, UAVs_states = self.algorithmScheduler.getNodeStates(env, 'U', self.node_priority)
        # vehicle_num, vehicles_states = self.algorithmScheduler.getNodeStates(env, 'V', self.node_priority)
        # RSU_num, RSUs_states = self.algorithmScheduler.getNodeStates(env, 'I', self.node_priority)
        # combined_states = UAVs_states + vehicles_states + RSUs_states
        node_num,combined_states = self.algorithmScheduler.getNodeStates(env=env, node_priority=self.node_priority)
        max_node_num = self.DDQN_dim_args.m_u + self.DDQN_dim_args.m_v + self.DDQN_dim_args.m_r
        max_sensor_node_num=self.DDQN_dim_args.m_u + self.DDQN_dim_args.m_v
        node_state_dim = self.DDQN_dim_args.dim_node
        encode_node_states = self._encode_node_states(combined_states, max_node_num, node_state_dim)

        generate_num = 0
        allocate_num = 0
        for mission_profile in new_missions_profile:
            if mission_profile['mission_arrival_time'] > cur_time - traffic_interval:
                generate_num += 1
            mission_id = mission_profile['mission_id']
            mission_sensor_type = mission_profile['mission_sensor_type']
            mission_accuracy = mission_profile['mission_accuracy']
            mission_position = mission_profile['mission_routes'][0]

            mission_states = self.algorithmScheduler.getMissionStates(env, [mission_profile]) # 默认有多个mission
            encode_mission_state = self._encode_mission_states(mission_states, 1, self.DDQN_dim_args.dim_mission)
            valid_sensor_num, sensor_states = self.algorithmScheduler.getSensorStates(env,
                                                                                      mission_sensor_type,
                                                                                      mission_accuracy,
                                                                                      excluded_sensor_ids,
                                                                                      mission_position,
                                                                                      self.TA_distance_Veh,
                                                                                      self.TA_distance_UAV,
                                                                                      self.node_priority)
            if valid_sensor_num == 0:
                continue
            encode_sensor_states, encode_mask = self._encode_sensor_states(sensor_states, max_sensor_node_num,
                                                                           self.DDQN_dim_args.max_sensors,
                                                                           self.DDQN_dim_args.dim_sensor)

            # 展平各张量
            node_flat = encode_node_states.flatten()
            mission_flat = encode_mission_state.flatten()
            sensor_flat = encode_sensor_states.flatten()
            mask_flat = encode_mask.flatten()
            # 拼接所有特征
            combined_states = np.concatenate([node_flat, mission_flat, sensor_flat])  # 形状变为[total_features]
            # 获取动作
            is_random, max_q_value, action_index = self.DDQN_env.takeAction(combined_states, mask_flat)

            if action_index is None:
                continue
            appointed_node_type, appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.algorithmScheduler.getSensorInfoByAction(
                env, action_index, sensor_states)
            if appointed_node_id is not None and appointed_sensor_id is not None:
                if appointed_node_type == 'U':
                    route_with_time={
                        'position':mission_position,
                        'to_stay_time':mission_profile['mission_duration'][0]
                    }
                    self.trafficScheduler.addUAVRoute(env, appointed_node_id, route_with_time)
                mission_profile['appointed_node_id'] = appointed_node_id
                mission_profile['appointed_sensor_id'] = appointed_sensor_id
                mission_profile['appointed_sensor_accuracy'] = appointed_sensor_accuracy
                mission_profile['mission_start_time'] = cur_time
                for _ in mission_profile['mission_routes']:
                    task_set = []
                    mission_task_profile = {
                        'task_node_id': appointed_node_id,
                        'task_deadline': mission_profile['mission_deadline'],
                        'arrival_time': mission_profile['mission_arrival_time'],
                        'return_size': mission_profile['mission_size']
                    }
                    new_task = self.taskScheduler.generateTaskOfMission(env, mission_task_profile)
                    task_set.append(new_task)
                    mission_profile['mission_task_sets'].append(task_set)
                self.missionScheduler.generateAndAddMission(env, mission_profile)
                allocate_num += 1

                # set state buffer
                if self.last_mission_id is not None:
                    self.ta_buffer.setNextState(self.last_mission_id, combined_states, mask_flat, False)
                self.ta_buffer.add(mission_id, combined_states,mask_flat,action_index)
                self.last_mission_id = mission_id

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)

        self.missionScheduler.setMissionEvaluationIndicators(env, generate_num, allocate_num)
        self.missionScheduler.deleteBeAssignedMissionsProfile(env, delete_mission_profile_ids)

    def scheduleReturning(self, env: AirFogSimEnv):
        """The returning logic. Relay or direct is controlled by probability.
        Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
        Direct, select nearest RSU, return_route=[RSU]

        Args:
            env (AirFogSimEnv): The environment object.
        """
        super().scheduleReturning(env)

    def scheduleTraffic(self, env: AirFogSimEnv):
        """The UAV traffic scheduling logic.


        Args:
            env (AirFogSimEnv): The environment object.
        """
        super().scheduleTraffic(env)

    def scheduleOffloading(self, env: AirFogSimEnv):
        # super().scheduleOffloading(env)
        pass

    def scheduleCommunication(self, env: AirFogSimEnv):
        super().scheduleCommunication(env)

    def scheduleComputing(self, env: AirFogSimEnv):
        super().scheduleComputing(env)

    def getRewardByTask(self, env: AirFogSimEnv):
        return super().getRewardByTask(env)

    def getRewardByMission(self, env: AirFogSimEnv):
        return super().getRewardByMission(env)

    def updateTAExperience(self, env: AirFogSimEnv):
        last_step_succ_mission_infos = self.missionScheduler.getLastStepSuccMissionInfos(env)
        last_step_fail_mission_infos = self.missionScheduler.getLastStepFailMissionInfos(env)
        for mission_info in last_step_succ_mission_infos:
            reward = self.rewardScheduler.getRewardByMission(env, mission_info)
            # reward要转成标准float型，否则以Float对象存入会出问题
            exp = self.ta_buffer.completeAndPopExperience(mission_info['mission_id'], float(reward))
            if exp is not None:
                self.DDQN_env.addExperience(*exp)
        for mission_info in last_step_fail_mission_infos:
            reward = self.rewardScheduler.getPunishByMission(env, mission_info)
            # reward要转成标准float型，否则以Float对象存入会出问题
            exp = self.ta_buffer.completeAndPopExperience(mission_info['mission_id'], float(reward))
            if exp is not None:
                self.DDQN_env.addExperience(*exp)
