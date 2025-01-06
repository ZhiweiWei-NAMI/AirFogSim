import argparse
import random

import torch

from airfogsim.airfogsim_env import AirFogSimEnv
from airfogsim.algorithm.crowdsensing.TransDDQN.TransDDQN_env import TransDDQN_Env
from airfogsim.algorithm.crowdsensing.MADDPG.MADDPG_env import MADDPG_Env
from airfogsim.airfogsim_algorithm import BaseAlgorithmModule
from .ReplayBuffer import BaseReplayBuffer
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def parseTransDDQNTrainArgs():
    parser = argparse.ArgumentParser(description='TransDDQN train arguments')
    parser.add_argument('--buffer_size', type=int, default=500)  # 经验池容量
    parser.add_argument('--lr', type=float, default=2e-3)  # 学习率
    parser.add_argument('--gamma', type=float, default=0.9)  # 折扣因子
    parser.add_argument('--epsilon', type=float, default=0.9)  # 探索系数
    parser.add_argument('--eps_end', type=float, default=0.01)  # 最低探索系数
    parser.add_argument('--eps_dec', type=float, default=5e-7)  # 探索系数衰减率
    parser.add_argument('--target_update', type=int, default=200)  # 目标网络的参数的更新频率
    parser.add_argument('--batch_size', type=int, default=32)  # 每次训练选取的经验数量
    parser.add_argument('--dim_hidden', type=int, default=128)  # 隐含层神经元个数
    parser.add_argument('--train_min_size', type=int, default=200)  # 经验池超过200后再训练(train_min_size>batch_size)
    parser.add_argument('--tau', type=float, default=0.995)  # 目标网络软更新平滑因子(策略网络权重)
    parser.add_argument('--smooth_factor', type=float, default=0.995)  # 最大q值平滑因子（旧值权重）
    parser.add_argument('--device', type=str, default=device)  # 训练设备(GPU/CPU)
    args = parser.parse_args()
    return args


def parseTransDDQNDimArgs():
    parser = argparse.ArgumentParser(description='TransDDQN dimension arguments')
    # [type, is_mission_node, is_schedulable, x, y, z]
    parser.add_argument('--dim_node', type=int, default=6)  # Dimension of nodes (UAV/Veh/RSU)
    # [sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]
    parser.add_argument('--dim_mission', type=int, default=10)  # Dimension of mission
    # [type, accuracy]
    parser.add_argument('--dim_sensor', type=int, default=2)  # Dimension of sensor
    parser.add_argument('--max_sensors', type=int, default=4)  # Sensors on each mission node
    parser.add_argument('--m_u', type=int, default=10)  # Maximum UAVs
    parser.add_argument('--m_v', type=int, default=100)  # Maximum vehicles
    parser.add_argument('--m_r', type=int, default=4)  # Maximum RSUs
    parser.add_argument('--dim_model', type=int, default=512)  # Embedding feature dimension
    parser.add_argument('--nhead', type=float, default=4)  # Head num
    parser.add_argument('--num_layers', type=float, default=3)  # Transformer encoder layers num
    parser.add_argument('--dim_hiddens', type=float, default=512)  # Dimension of hidden layer
    args = parser.parse_args()
    return args


def parseMADDPGTrainArgs():
    parser = argparse.ArgumentParser(description='MADDPG train arguments')
    parser.add_argument('--buffer_size', type=int, default=1000)  # 经验池容量
    parser.add_argument('--lr', type=float, default=2e-3)  # 学习率
    parser.add_argument('--gamma', type=float, default=0.9)  # 折扣因子
    parser.add_argument('--var', type=float, default=1.0)  # 动作探索随机噪声
    parser.add_argument('--var_end', type=float, default=0.01)  # 最低噪声
    parser.add_argument('--var_dec', type=float, default=2e-5)  # 噪声衰减率
    parser.add_argument('--target_update', type=int, default=200)  # 目标网络的参数的更新频率
    parser.add_argument('--batch_size', type=int, default=32)  # 每次训练选取的经验数量
    parser.add_argument('--dim_hidden', type=int, default=128)  # 隐含层神经元个数
    parser.add_argument('--train_min_size', type=int, default=200)  # 经验池超过200后再训练(train_min_size>batch_size)
    parser.add_argument('--tau', type=float, default=0.995)  # 目标网络软更新平滑因子(策略网络权重)
    parser.add_argument('--device', type=str, default=device)  # 训练设备(GPU/CPU)
    args = parser.parse_args()
    return args


def parseMADDPGDimArgs():
    # [x, y, z]
    dim_neighbor_UAV = 3
    m_neighbor_UAVs = 5
    # [left_sensing_time, left_return_size, x, y, z]
    dim_trans_mission = 5
    m_trans_missions = 50
    # [sensor_type, accuracy, return_size, arrival_time, TTL, duration, x, y, z, distance_threshold]
    dim_todo_mission = 10
    m_todo_missions = 4
    # [x, y, z, energy]
    dim_self_UAV = 4
    dim_observation = dim_neighbor_UAV * m_neighbor_UAVs + dim_trans_mission * m_trans_missions + dim_todo_mission * m_todo_missions + dim_self_UAV

    parser = argparse.ArgumentParser(description='MADDPG dimension arguments')
    # 分解维度
    parser.add_argument('--dim_neighbor_UAV', type=int, default=dim_neighbor_UAV)
    parser.add_argument('--m_neighbor_UAVs', type=int, default=m_neighbor_UAVs)
    parser.add_argument('--dim_trans_mission', type=int, default=dim_trans_mission)
    parser.add_argument('--m_trans_missions', type=int, default=m_trans_missions)
    parser.add_argument('--dim_todo_mission', type=int, default=dim_todo_mission)
    parser.add_argument('--m_todo_missions', type=int, default=m_todo_missions)
    parser.add_argument('--dim_self_UAV', type=int, default=dim_self_UAV)

    # 算法实际使用的维度
    parser.add_argument('--dim_observation', type=int, default=dim_observation)  # Dimension of observation
    parser.add_argument('--dim_action', type=int, default=3)  # Dimension of action [angle, phi, speed]
    parser.add_argument('--n_agents', type=int, default=10)  # Number of agents
    parser.add_argument('--dim_hiddens', type=float, default=512)  # Dimension of hidden layer
    args = parser.parse_args()
    return args


class TransDDQN_MADDPG_AlgorithmModule(BaseAlgorithmModule):
    """
    Use different schedulers to interact with the environment before calling env.step(). Manipulate different environments with the same algorithm design at the same time for learning sampling efficiency.\n
    Any implementation of the algorithm should inherit this class and implement the algorithm logic in the `scheduleStep()` method.
    """

    '''
    scheduleOffloading: BaseAlgorithm.
    scheduleComputing: BaseAlgorithm.
    scheduleCommunication: BaseAlgorithm.
    scheduleMission: 
        Mission: Missions assigned to both vehicles and UAVs, decided by TransDDQN model.
        Sensor: Decided by TransDDQN model.
    scheduleReturning: Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
                       Direct, select nearest RSU, return_route=[RSU]
                       Relay or direct is controlled by probability.
    scheduleTraffic: 
        UAV: Fly to next position in route list and stay for a period of time.
    '''

    class TaskAllocationReplayBuffer(BaseReplayBuffer):
        def __init__(self):
            # 创建一个字典，长度不限
            super().__init__()

        def __expToFlattenArray(self, exp):
            node_state= exp['node_state']
            mission_state= exp['mission_state']
            sensor_state= exp['sensor_state']
            sensor_mask= exp['sensor_mask']
            action = exp['action']
            reward = exp['reward']
            next_node_state= exp['next_node_state']
            next_mission_state= exp['next_mission_state']
            next_sensor_state= exp['next_sensor_state']
            next_sensor_mask= exp['next_sensor_mask']
            done = exp['done']
            return np.array(node_state), np.array(mission_state), np.array(sensor_state), np.array(
                sensor_mask), action,  reward, np.array(next_node_state), np.array(next_mission_state), np.array(
                next_sensor_state), np.array(next_sensor_mask), done

        def add(self, exp_id, node_state, mission_state, sensor_state, sensor_mask,action, reward=None, next_node_state=None, next_mission_state=None, next_sensor_state=None, next_sensor_mask=None, done=None):
            self.buffer[exp_id] = {'node_state': node_state,'mission_state': mission_state,'sensor_state': sensor_state,'sensor_mask': sensor_mask, 'action': action, 'reward': reward,
                                   'next_node_state': next_node_state, 'next_mission_state': next_mission_state, 'next_sensor_state': next_sensor_state, 'next_sensor_mask': next_sensor_mask, 'done': done}

        def setNextState(self, exp_id, next_node_state, next_mission_state, next_sensor_state, next_sensor_mask, done):
            assert exp_id in self.buffer, "State_id is invalid."
            self.buffer[exp_id]['next_node_state'] = next_node_state
            self.buffer[exp_id]['next_mission_state'] = next_mission_state
            self.buffer[exp_id]['next_sensor_state'] = next_sensor_state
            self.buffer[exp_id]['next_sensor_mask'] = next_sensor_mask
            self.buffer[exp_id]['done'] = done

        def completeAndPopExperience(self, exp_id, reward):
            assert exp_id in self.buffer, "exp_id is invalid."
            self.buffer[exp_id]['reward'] = reward
            packed_exp = self.__expToFlattenArray(self.buffer[exp_id])
            del self.buffer[exp_id]
            return packed_exp

        def size(self):
            return super().size()

        def clear(self):
            super().clear()

    class PathPlanReplayBuffer(BaseReplayBuffer):
        def __init__(self):
            # 创建一个字典，长度不限
            super().__init__()

        def __expToFlattenArray(self, exp):
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp['next_state']
            done = exp['done']
            return np.array(state), action,reward, np.array(next_state), done

        def add(self, exp_id, state, action, reward=None, next_state=None,  done=None):
            self.buffer[exp_id] = {'state': state, 'action': action, 'reward': reward,
                                   'next_state': next_state, 'done': done}

        def setNextState(self, exp_id, next_state, done):
            assert exp_id in self.buffer, "State_id is invalid."
            self.buffer[exp_id]['next_state'] = next_state
            self.buffer[exp_id]['done'] = done

        def completeAndPopExperience(self, exp_id, reward):
            assert exp_id in self.buffer, "exp_id is invalid."
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

    def initialize(self, env: AirFogSimEnv, config={}, last_episode=None):
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
        self.min_UAV_speed, self.max_UAV_speed = self.trafficScheduler.getConfig('UAV_speed_range')
        self.max_n_vehicles = self.trafficScheduler.getConfig('max_n_vehicles')
        self.max_n_UAVs = self.trafficScheduler.getConfig('max_n_UAVs')
        self.max_n_RSUs = self.trafficScheduler.getConfig('max_n_RSUs')
        self.max_mission_size = self.missionScheduler.getConfig('mission_size_range')[1]
        self.max_energy = env.energy_manager.getConfig('initial_energy_range')[0]
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

        self.TransDDQN_dim_args = parseTransDDQNDimArgs()
        self.TransDDQN_train_args = parseTransDDQNTrainArgs()
        self.TransDDQN_env = TransDDQN_Env(self.TransDDQN_dim_args, self.TransDDQN_train_args)
        if last_episode is not None:
            self.TransDDQN_env.loadModel(last_episode)

        self.MADDPG_dim_args = parseMADDPGDimArgs()
        self.MADDPG_train_args = parseMADDPGTrainArgs()
        self.MADDPG_env = MADDPG_Env(self.MADDPG_dim_args, self.MADDPG_train_args)
        if last_episode is not None:
            self.MADDPG_env.loadModel(last_episode)

        self.last_mission_id = None  # Last allocated mission id,used in next state update
        self.ta_buffer = self.TaskAllocationReplayBuffer()
        self.pp_buffer=self.PathPlanReplayBuffer()
        self.UAV_states={}

    def reset(self, env: AirFogSimEnv):
        self.last_mission_id = None  # Last allocated mission id,used in next state update
        self.ta_buffer.clear()
        self.pp_buffer.clear()
        self.UAV_states.clear()

    def _encode_node_states(self, node_states, max_node_num, dim_state):
        # UAVs,Vehicles,RSUs
        # [id, type, is_mission_node, is_schedulable, x, y, z]
        # [1, 'U', True, True, 105.23, 568.15. 225.65]
        # 选取[type, is_mission_node, is_schedulable, x, y, z]

        encode_states = []
        # 按type,id排序
        sorted_node_states = sorted(node_states, key=lambda x: (self.node_priority[x[1]], x[0]))

        for node_state in sorted_node_states:
            node_type = self.node_type_dict.get(node_state[0], -1)
            is_mission_node = int(node_state[1])
            is_schedulable = int(node_state[2])
            position_x = (node_state[3] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (node_state[4] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (node_state[5] - self.min_position_z) / (self.max_position_z - self.min_position_z)

            state = [node_type, is_mission_node, is_schedulable, position_x, position_y, position_z]
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
            sensor_type = self.node_type_dict.get(mission_state[0], -1)
            accuracy = mission_state[1]
            return_size = mission_state[2] / self.max_mission_size
            arrival_time = mission_state[3] / self.max_simulation_time
            TTL = mission_state[4] / self.max_simulation_time
            duration = mission_state[5] / self.max_simulation_time
            position_x = (mission_state[6] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (mission_state[7] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (mission_state[8] - self.min_position_z) / (self.max_position_z - self.min_position_z)
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

    def _encode_sensor_states(self, sensor_states, max_node_num, node_sensor_num, dim_state):
        # [node_id,node_type,id,type, accuracy,candidate]
        # [1, 'U', 3, 0.8, True]
        # 选取[type, accuracy]

        encode_states = []
        encode_mask = []
        # 1. 按 node_type,node_id 排序
        sorted_sensor_states = sorted(sensor_states, key=lambda x: (self.node_priority[x[0][1]], x[0][0]))

        # 2. 删除不需要的属性
        for node_group in sorted_sensor_states:
            node_state_group = []
            node_mask_group = []
            for sensor_state in node_group:
                processed_state = sensor_state[3:4]  # 去除 node_id
                node_state_group.append(processed_state)
                node_mask_group.append(sensor_state[-1])
            encode_states.append(node_state_group)
            encode_mask.append(node_mask_group)

        # 3. 补齐长度
        valid_node_num = len(encode_states)
        if valid_node_num < max_node_num:
            for _ in range(max_node_num - valid_node_num):
                node_state_group = []
                node_mask_group = []
                for _ in range(node_sensor_num):
                    node_state_group.append([0 for _ in range(dim_state)])  # 补充零
                    node_mask_group.append(False)  # 补充False
                encode_states.append(node_state_group)
                encode_mask.append(node_mask_group)

        return np.array(encode_states), np.array(encode_mask)

    def _encode_neighbor_UAV_states(self, UAV_states, max_UAV_num, dim_state):
        # [x, y, z]
        # [105.23, 568.15. 225.65]
        # 选取[x, y, z]

        encode_states = []
        for UAV_state in UAV_states:
            position_x = (UAV_state[0] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (UAV_state[1] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (UAV_state[2] - self.min_position_z) / (self.max_position_z - self.min_position_z)

            state = [position_x, position_y, position_z]
            encode_states.append(state)

        # 补齐长度
        valid_UAV_num = len(encode_states)
        if valid_UAV_num < max_UAV_num:
            for _ in range(max_UAV_num - valid_UAV_num):
                encode_states.append([0 for _ in range(dim_state)])  # 补充零

        return np.array(encode_states)

    def _encode_trans_mission_states(self, trans_mission_states, max_mission_num, dim_state):
        # [left_sensing_time, left_return_size, x, y, z]
        # [5,30,120.25,262.05,553.25]
        # 选取[left_sensing_time, left_return_size, x, y, z]

        encode_states = []
        for mission_state in trans_mission_states:
            left_sensing_time = mission_state[3] / self.max_simulation_time
            left_return_size = mission_state[2] / self.max_mission_size
            position_x = (mission_state[6] - self.min_position_x) / (self.max_position_x - self.min_position_x)
            position_y = (mission_state[7] - self.min_position_y) / (self.max_position_y - self.min_position_y)
            position_z = (mission_state[8] - self.min_position_z) / (self.max_position_z - self.min_position_z)

            state = [left_sensing_time, left_return_size, position_x, position_y, position_z]
            encode_states.append(state)

        # 补齐长度
        valid_mission_num = len(encode_states)
        if valid_mission_num < max_mission_num:
            for _ in range(max_mission_num - valid_mission_num):
                encode_states.append([0 for _ in range(dim_state)])  # 补充零

        return np.array(encode_states)

    def _encode_self_UAV_states(self, self_UAV_states):
        # [x, y, z, energy]
        # [105.23, 568.15, 225.65, 12000]
        # 选取[x, y, z, energy]

        position_x = (self_UAV_states[0] - self.min_position_x) / (self.max_position_x - self.min_position_x)
        position_y = (self_UAV_states[1] - self.min_position_y) / (self.max_position_y - self.min_position_y)
        position_z = (self_UAV_states[2] - self.min_position_z) / (self.max_position_z - self.min_position_z)
        energy = self_UAV_states[3] / self.max_energy
        state = [position_x, position_y, position_z, energy]

        return np.array(state)

    def _transformUAVMobilityPattern(self,norm_angle,norm_phi,norm_speed):
        angle=(norm_angle * 2 * np.pi) - np.pi
        phi=(norm_phi * 2 * np.pi) - np.pi

        speed_range_length=self.max_UAV_speed-self.min_UAV_speed
        speed=norm_speed*speed_range_length+self.min_UAV_speed
        return angle,phi,speed



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

    def scheduleMission(self, env: AirFogSimEnv):
        """The mission scheduling logic.
        Mission: Missions assigned to both vehicles and UAVs, each type has a probability of sum of 1.
        Sensor: Assigned to vehicle, select the sensor closest to PoI from the idle sensors with accuracy higher than required(Distance First).
                Assigned to RSU, select the sensor with the lowest accuracy from the idle sensors with accuracy higher than required(Accuracy Lowerbound).

        Args:
            env (AirFogSimEnv): The environment object.

        """

        cur_time = self.trafficScheduler.getCurrentTime(env)
        traffic_interval = self.trafficScheduler.getTrafficInterval(env)
        new_missions_profile = self.missionScheduler.getToBeAssignedMissionsProfile(env, cur_time)
        delete_mission_profile_ids = []
        excluded_sensor_ids = []

        UAVs_states = self.algorithmScheduler.getNodeStates(env, 'U', self.max_n_UAVs)
        vehicles_states = self.algorithmScheduler.getNodeStates(env, 'V', self.max_n_vehicles)
        RSUs_states = self.algorithmScheduler.getNodeStates(env, 'I', self.max_n_RSUs)
        combined_states = UAVs_states + vehicles_states + RSUs_states
        max_node_num = self.TransDDQN_dim_args.m_u + self.TransDDQN_dim_args.m_v + self.TransDDQN_dim_args.m_r
        node_state_dim = self.TransDDQN_dim_args.dim_node
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

            mission_state = self.algorithmScheduler.getMissionStates(env, mission_profile)
            encode_mission_state = self._encode_mission_states([mission_state], 1, self.TransDDQN_dim_args.dim_mission)
            valid_sensor_num, sensor_states = self.algorithmScheduler.getSensorStates(env,
                                                                                      mission_sensor_type,
                                                                                      mission_accuracy,
                                                                                      excluded_sensor_ids)
            encode_sensor_states,encode_mask = self._encode_sensor_states(sensor_states, max_node_num,
                                                              self.TransDDQN_dim_args.max_sensors,
                                                              self.TransDDQN_dim_args.dim_sensor)
            if valid_sensor_num == 0:
                continue
            is_random, max_q_value, action_index = self.TransDDQN_env.takeAction(encode_node_states,encode_mission_state,encode_sensor_states, encode_mask)

            if self.last_mission_id is not None:
                self.ta_buffer.setNextState(self.last_mission_id, encode_node_states,encode_mission_state,encode_sensor_states, encode_mask, False)
            self.ta_buffer.add(mission_id, encode_node_states,encode_mission_state,encode_sensor_states, encode_mask,action_index)
            self.last_mission_id = mission_id

            appointed_node_type, appointed_node_id, appointed_sensor_id, appointed_sensor_accuracy = self.algorithmScheduler.getSensorInfoByAction(
                env, action_index, sensor_states,self.node_type_dict)
            if appointed_node_id != None and appointed_sensor_id != None:
                if appointed_node_type == 'U':
                    self.trafficScheduler.addUAVRoute(env, appointed_node_id, mission_position)
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

                delete_mission_profile_ids.append(mission_profile['mission_id'])
                excluded_sensor_ids.append(appointed_sensor_id)

        self.missionScheduler.setMissionEvaluationIndicators(generate_num, allocate_num)
        self.missionScheduler.deleteBeAssignedMissionsProfile(env, delete_mission_profile_ids)

    def scheduleReturning(self, env: AirFogSimEnv):
        """The returning logic. Relay or direct is controlled by probability.
        Relay(only for task assigned to vehicle), select nearest UAV and nearest RSU, return_route=[UAV,RSU]
        Direct, select nearest RSU, return_route=[RSU]

        Args:
            env (AirFogSimEnv): The environment object.
        """
        waiting_to_return_tasks = self.taskScheduler.getWaitingToReturnTaskInfos(env)
        for task_node_id, tasks in waiting_to_return_tasks.items():
            for task in tasks:
                current_node_id = task.getCurrentNodeId()
                current_node_type = self.entityScheduler.getNodeTypeById(env, current_node_id)
                vehicle_num = self.entityScheduler.getNodeNumByType(env, 'V')
                UAV_num = self.entityScheduler.getNodeNumByType(env, 'U')
                RSU_num = self.entityScheduler.getNodeNumByType(env, 'I')
                if current_node_type == 'V':
                    if UAV_num > 0:
                        V2U_distance = np.zeros((UAV_num))
                        for u_idx in range(UAV_num):
                            u_id = self.entityScheduler.getNodeInfoByIndexAndType(env, u_idx, 'U')['id']
                            distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, u_id)
                            V2U_distance[u_idx] = distance
                        nearest_u_distance = np.max(V2U_distance)
                        nearest_u_idx = np.unravel_index(np.argmax(V2U_distance), V2U_distance.shape)
                        nearest_u_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_u_idx[0]), 'U')[
                            'id']

                    if RSU_num > 0:
                        V2R_distance = np.zeros((RSU_num))
                        for r_idx in range(RSU_num):
                            r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'I')['id']
                            distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                            V2R_distance[r_idx] = distance
                        nearest_r_distance = np.max(V2R_distance)
                        nearest_r_idx = np.unravel_index(np.argmax(V2R_distance), V2R_distance.shape)
                        nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'I')[
                            'id']

                    relay_probability = 0.5
                    if random.random() < relay_probability and UAV_num > 0:
                        return_route = [nearest_u_id, nearest_r_id]
                    else:
                        return_route = [nearest_r_id]

                elif current_node_type == 'U':
                    U2R_distance = np.zeros((RSU_num))
                    for r_idx in range(RSU_num):
                        r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, r_idx, 'I')['id']
                        distance = self.trafficScheduler.getDistanceBetweenNodesById(env, current_node_id, r_id)
                        U2R_distance[r_idx] = distance
                    nearest_r_distance = np.max(U2R_distance)
                    nearest_r_idx = np.unravel_index(np.argmax(U2R_distance), U2R_distance.shape)
                    nearest_r_id = self.entityScheduler.getNodeInfoByIndexAndType(env, int(nearest_r_idx[0]), 'I')['id']
                    return_route = [nearest_r_id]
                else:
                    raise TypeError('Node type is invalid')

                self.taskScheduler.setTaskReturnRoute(env, task.getTaskId(), return_route)

    def scheduleTraffic(self, env: AirFogSimEnv):
        """The UAV traffic scheduling logic. Should be implemented by the subclass. Default is move to the next
         mission sensing or task position. If there is no mission allocated to UAV, movement is random.

        Args:
            env (AirFogSimEnv): The environment object.
        """
        distance_threshold=self.missionScheduler.getConfig(env,'distance_threshold')
        UAVs_info = self.trafficScheduler.getUAVTrafficInfos(env)
        UAVs_mobile_pattern = {}
        for UAV_id, UAV_info in UAVs_info.items():
            UAV_index=int(UAV_id.split('_')[-1])  # 转换为整数
            current_position = UAV_info['position']

            neighbor_UAV_states=self.algorithmScheduler.getNeighborUAVStates(env,current_position,distance_threshold,self.MADDPG_dim_args.m_neighbor_UAVs)
            trans_mission_states=self.algorithmScheduler.getTransMissionStates(env,current_position,distance_threshold,self.MADDPG_dim_args.m_trans_missions)
            todo_mission_profiles=self.missionScheduler.getExecutingMissionProfiles(env,UAV_id)
            todo_mission_states=self.algorithmScheduler.getMissionStates(env,todo_mission_profiles)
            self_UAV_state=self.algorithmScheduler.getSelfUAVStates(env,UAV_id)

            encode_neighbor_UAV_states=self._encode_neighbor_UAV_states(neighbor_UAV_states,self.MADDPG_dim_args.m_neighbor_UAVs,self.MADDPG_dim_args.dim_neighbor_UAV).flatten()
            encode_trans_mission_states=self._encode_trans_mission_states(trans_mission_states,self.MADDPG_dim_args.m_trans_missions,self.MADDPG_dim_args.dim_trans_mission).flatten()
            encode_todo_mission_states=self._encode_mission_states(todo_mission_states,self.MADDPG_dim_args.m_todo_missions,self.MADDPG_dim_args.dim_todo_mission).flatten()
            encode_self_UAV_state=self._encode_self_UAV_states(self_UAV_state).flatten()
            combined_state=np.concatenate((encode_neighbor_UAV_states,encode_trans_mission_states,encode_todo_mission_states,encode_self_UAV_state))

            norm_action=self.MADDPG_env.takeAction(combined_state)
            norm_angle, norm_phi, norm_speed=norm_action
            angle,phi,speed=self.transformUAVMobilityPattern(norm_angle=norm_angle,norm_phi=norm_phi,norm_speed=norm_speed)
            mobility_pattern = {'angle': angle, 'phi': phi, 'speed': speed}
            UAVs_mobile_pattern[UAV_id] = mobility_pattern

            last_state=self.UAV_states.get(UAV_id,None)
            if last_state is not None:
                self.pp_buffer.add(UAV_id,last_state,norm_action)
                self.pp_buffer.setNextState(UAV_id,combined_state,False)
            self.UAV_states[UAV_id]=combined_state

            # target_position = self.missionScheduler.getNearestMissionPosition(env, UAV_id, UAV_info['position'])
            # if target_position is None:
            #     # 悬停
            #     mobility_pattern = {'angle': 0, 'phi': 0, 'speed': 0}
            #     UAVs_mobile_pattern[UAV_id] = mobility_pattern
            # else:
            #     # Update stay time at first position in route
            #     distance_threshold = self.trafficScheduler.getConfig(env, 'distance_threshold')
            #     distance = np.linalg.norm(np.array(target_position) - np.array(current_position))
            #     if distance < distance_threshold:
            #         self.trafficScheduler.updateRoute(env, UAV_id, self.trafficScheduler.getTrafficInterval())
            #
            #     delta_x = target_position[0] - current_position[0]
            #     delta_y = target_position[1] - current_position[1]
            #     delta_z = target_position[2] - current_position[2]
            #
            #     # 计算 xy 平面的方位角
            #     angle = np.arctan2(delta_y, delta_x)
            #
            #     # 计算 z 相对于 xy 平面的仰角
            #     distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
            #     phi = np.arctan2(delta_z, distance_xy)
            #
            #     mobility_pattern = {'angle': angle, 'phi': phi}
            #     UAV_speed_range = self.trafficScheduler.getConfig(env, 'UAV_speed_range')
            #     mobility_pattern['speed'] = random.uniform(UAV_speed_range[0], UAV_speed_range[1])
            #     UAVs_mobile_pattern[UAV_id] = mobility_pattern
        self.trafficScheduler.setUAVMobilityPatterns(env, UAVs_mobile_pattern)

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
            exp = self.replay_buffer.completeAndPopExperience(mission_info['mission_id'], reward)
            self.TransDDQN_env.addExperience(*exp)
        for mission_info in last_step_fail_mission_infos:
            reward = self.rewardScheduler.getPunishByMission(env, mission_info)
            exp = self.replay_buffer.completeAndPopExperience(mission_info['mission_id'], reward)
            self.TransDDQN_env.addExperience(*exp)

    def updatePPExperience(self,env: AirFogSimEnv):
        if self.pp_buffer.size() == 0:
            return

        UAV_energy_consumptions,UAV_trans_datas,UAV_sensing_datas = self.algorithmScheduler.getUAVStepRecord(env)
        UAV_ids = UAV_energy_consumptions.keys()

        for UAV_id in UAV_ids:
            energy_consumption = UAV_energy_consumptions[UAV_id]
            trans_data = UAV_trans_datas[UAV_id]
            sensing_data = UAV_sensing_datas[UAV_id]
            reward = trans_data + sensing_data - energy_consumption
            exp=self.pp_buffer.completeAndPopExperience(UAV_id,reward)
            self.MADDPG_env.addExperience(*exp)



