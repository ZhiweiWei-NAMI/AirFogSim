
from ..all_channels import V2IChannel, V2UChannel, V2VChannel, U2IChannel, U2UChannel, I2IChannel
import numpy as np
class ChannelManager:
    """ChannelManager is the class for managing the wireless communication channels in the airfogsim environment. It provides the APIs for the agent to interact with the channels."""
    def __init__(self, n_RSU=1, n_UAV=1, n_Veh=1, hei_UAVs=100, RSU_positions=[], simulation_interval=0.1):
        self.V2V_power_dB = 23 # dBm 记录的都是最大功率
        self.V2I_power_dB = 26
        self.V2U_power_dB = 26
        self.U2U_power_dB = 29
        self.U2I_power_dB = 29
        self.U2V_power_dB = 29
        self.I2I_power_dB = 29
        self.I2V_power_dB = 29
        self.I2U_power_dB = 29
        self.sig2_dB = -104
        self.sig2 = 10**(self.sig2_dB/10)
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.V2U_Shadowing = []
        self.U2U_Shadowing = []
        self.U2V_Shadowing = []
        self.U2I_Shadowing = []
        self.veh_delta_distance = []
        self.uav_delta_distance = []
        self.n_RB = 10 # LTE场景，大致是100RB
        self.RB_bandwidth = 2 # 180kHz = 12子载波 * 15kHz
        self.bandwidth = self.n_RB * self.RB_bandwidth
        self.V2VChannel = None
        self.V2IChannel = None
        self.V2UChannel = None
        self.U2UChannel = None
        self.U2IChannel = None
        self.I2IChannel = None

        self.V2V_Interference = None
        self.V2I_Interference = None
        self.V2U_Interference = None
        self.U2U_Interference = None
        self.U2I_Interference = None
        self.I2I_Interference = None
        # 是否可以连接的信道, 这里双向都要有，因为功率不同
        self.V2V_active_links = None
        self.V2I_active_links = None
        self.V2U_active_links = None
        self.U2U_active_links = None
        self.U2V_active_links = None
        self.U2I_active_links = None
        self.I2I_active_links = None
        self.I2U_active_links = None
        self.I2V_active_links = None

        self.V2V_Rate = None
        self.V2I_Rate = None
        self.V2U_Rate = None
        self.U2U_Rate = None
        self.U2I_Rate = None
        self.U2V_Rate = None
        self.I2I_Rate = None
        self.I2U_Rate = None
        self.I2V_Rate = None

        self.n_Veh = n_Veh
        self.n_RSU = n_RSU
        self.n_UAV = n_UAV
        self.hei_UAVs = hei_UAVs
        self.RSU_positions = RSU_positions
        self.simulation_interval = simulation_interval

        self._initialize_Channels()
        self._initialize_Interference_and_active()
        self.resetActiveLinks()



    def _initialize_Channels(self):
        self.V2VChannel = V2VChannel(self.n_Veh, self.n_RB)  # number of vehicles
        self.V2IChannel = V2IChannel(self.n_Veh, self.n_RSU, self.n_RB, self.RSU_positions)
        self.V2UChannel = V2UChannel(self.n_Veh, self.n_RB, self.n_UAV, self.hei_UAVs)
        self.U2UChannel = U2UChannel(self.n_RB, self.n_UAV, self.hei_UAVs)
        self.U2IChannel = U2IChannel(self.n_RB, self.n_RSU, self.n_UAV, self.hei_UAVs, self.RSU_positions)
        self.I2IChannel = I2IChannel(self.n_RB, self.n_RSU, self.RSU_positions)
    
    def _initialize_Interference_and_active(self):
        self.V2I_Interference = np.zeros((self.n_Veh, self.n_RSU)) + self.sig2 # 默认每个车辆归属于一个基站
        self.V2V_Interference = np.zeros((self.n_Veh, self.n_Veh)) + self.sig2
        self.V2U_Interference = np.zeros((self.n_Veh, self.n_UAV)) + self.sig2
        self.U2I_Interference = np.zeros((self.n_UAV, self.n_RSU)) + self.sig2
        self.U2V_Interference = np.zeros((self.n_UAV, self.n_Veh)) + self.sig2
        self.U2U_Interference = np.zeros((self.n_UAV, self.n_UAV)) + self.sig2
        self.I2I_Interference = np.zeros((self.n_RSU, self.n_RSU)) + self.sig2
        self.I2V_Interference = np.zeros((self.n_RSU, self.n_Veh)) + self.sig2
        self.I2U_Interference = np.zeros((self.n_RSU, self.n_UAV)) + self.sig2

    def resetActiveLinks(self):
        self.V2V_active_links = np.zeros((self.n_Veh, self.n_Veh, self.n_RB), dtype='bool')
        self.V2I_active_links = np.zeros((self.n_Veh, self.n_RSU, self.n_RB), dtype='bool')
        self.V2U_active_links = np.zeros((self.n_Veh, self.n_UAV, self.n_RB), dtype='bool')
        self.U2U_active_links = np.zeros((self.n_UAV, self.n_UAV, self.n_RB), dtype='bool')
        self.U2V_active_links = np.zeros((self.n_UAV, self.n_Veh, self.n_RB), dtype='bool')
        self.U2I_active_links = np.zeros((self.n_UAV, self.n_RSU, self.n_RB), dtype='bool')
        self.I2U_active_links = np.zeros((self.n_RSU, self.n_UAV, self.n_RB), dtype='bool')
        self.I2V_active_links = np.zeros((self.n_RSU, self.n_Veh, self.n_RB), dtype='bool')
        self.I2I_active_links = np.zeros((self.n_RSU, self.n_RSU, self.n_RB), dtype='bool')
    
    def updateNodes(self, n_Veh, n_UAV, n_RSU):
        self.n_Veh = n_Veh
        self.n_UAV = n_UAV
        self.n_RSU = n_RSU
        self._initialize_Channels()
        self._initialize_Interference_and_active()
        self.resetActiveLinks()

    def updateFastFading(self, UAVs, vehicles, vid_index, uav_index):
        """Renew the channels with fast fading.

        Args:
            vehicles (dict): The vehicles in the environment.
            UAVs (dict): The UAVs in the environment.
            vid_index (dict): The projection from the vehicle id to the index.
            uav_index (dict): The projection from the UAV id to the index.

        """
        if self.n_Veh == 0:
            return
        self._renew_channel(vehicles, UAVs, vid_index, uav_index)
        self._update_small_fading()
        # 为什么要减去快速衰落?
        V2VChannel_with_fastfading = np.repeat(self.V2VChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2VChannel_with_fastfading = V2VChannel_with_fastfading - self.V2VChannel.FastFading
        V2IChannel_with_fastfading = np.repeat(self.V2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2IChannel_with_fastfading = V2IChannel_with_fastfading - self.V2IChannel.FastFading
        V2UChannel_with_fastfading = np.repeat(self.V2UChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2UChannel_with_fastfading = V2UChannel_with_fastfading - self.V2UChannel.FastFading
        U2UChannel_with_fastfading = np.repeat(self.U2UChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2UChannel_with_fastfading = U2UChannel_with_fastfading - self.U2UChannel.FastFading
        U2IChannel_with_fastfading = np.repeat(self.U2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.U2IChannel_with_fastfading = U2IChannel_with_fastfading - self.U2IChannel.FastFading
        I2IChannel_with_fastfading = np.repeat(self.I2IChannel_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.I2IChannel_with_fastfading = I2IChannel_with_fastfading

    def _update_small_fading(self):
        self.V2IChannel.update_fast_fading()
        self.V2VChannel.update_fast_fading()
        self.U2IChannel.update_fast_fading()
        self.U2UChannel.update_fast_fading()
        self.V2UChannel.update_fast_fading()

    def _renew_channel(self, vehicles, UAVs, vid_index, uav_index):
        veh_positions = [c.position for c in sorted(vehicles.values(), key=lambda x: vid_index[x.id])]
        uav_positions = [c.position for c in sorted(UAVs.values(), key=lambda x: uav_index[x.id])]
        self._update_large_fading(veh_positions, uav_positions, vehicles, UAVs, vid_index, uav_index)
        self.V2VChannel_abs = self.V2VChannel.PathLoss + self.V2VChannel.Shadow
        self.V2IChannel_abs = self.V2IChannel.PathLoss + self.V2IChannel.Shadow
        self.V2UChannel_abs = self.V2UChannel.PathLoss + self.V2UChannel.Shadow
        self.U2UChannel_abs = self.U2UChannel.PathLoss + self.U2UChannel.Shadow
        self.U2IChannel_abs = self.U2IChannel.PathLoss + self.U2IChannel.Shadow
        self.I2IChannel_abs = self.I2IChannel.PathLoss + self.I2IChannel.Shadow
    
    def _update_large_fading(self, veh_positions, uav_positions, vehicles, UAVs, vid_index, uav_index):
        self.V2IChannel.update_positions(veh_positions)
        self.V2VChannel.update_positions(veh_positions)
        self.U2IChannel.update_positions(uav_positions)
        self.U2UChannel.update_positions(uav_positions)
        self.V2UChannel.update_positions(veh_positions, uav_positions)
        
        # 更新path loss
        self.V2IChannel.update_pathloss()
        self.V2VChannel.update_pathloss()
        self.U2IChannel.update_pathloss()
        self.U2UChannel.update_pathloss()
        self.V2UChannel.update_pathloss()
        self.I2IChannel.update_pathloss()
        # 计算距离差，根据self.vid_index的index数值排序
        veh_delta_distance = self.simulation_interval * np.asarray([c.velocity for c in sorted(vehicles.values(), key=lambda x: vid_index[x.id])])
        uav_delta_distance = self.simulation_interval * np.asarray([c.velocity for c in sorted(UAVs.values(), key=lambda x: uav_index[x.id])])
        # 更新阴影
        self.V2IChannel.update_shadow(veh_delta_distance)
        self.V2VChannel.update_shadow(veh_delta_distance)
        self.U2IChannel.update_shadow(uav_delta_distance)
        self.U2UChannel.update_shadow(uav_delta_distance)
        self.V2UChannel.update_shadow(veh_delta_distance, uav_delta_distance)
        self.I2IChannel.update_shadow()

    def activateLink(self, transmitter_idx, receiver_idx, allocated_RBs, channel_type):
        """Activate the link between the transmitter and the receiver.

        Args:
            transmitter_idx (int): The index of the transmitter corresponding to its type.
            receiver_idx (int): The index of the receiver corresponding to its type.
            allocated_RBs (list): The allocated RBs.
            channel_type (str): The channel type.

        """
        assert transmitter_idx >= 0 and receiver_idx >= 0
        if channel_type == 'V2V':
            self.V2V_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'V2I':
            self.V2I_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'V2U':
            self.V2U_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'U2U':
            self.U2U_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'U2V':
            self.U2V_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'U2I':
            self.U2I_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'I2U':
            self.I2U_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'I2V':
            self.I2V_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'I2I':
            self.I2I_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True
        elif channel_type == 'I2I':
            self.I2I_active_links[transmitter_idx, receiver_idx, allocated_RBs] = True

    def getRateByChannelType(self, transmitter_idx, receiver_idx, channel_type):
        """Get the rate by the channel type.

        Args:
            transmitter_idx (int): The index of the transmitter corresponding to its type.
            receiver_idx (int): The index of the receiver corresponding to its type.
            channel_type (str): The channel type.

        Returns:
            np.ndarray: The rate of the channel in each RB.
        """
        if channel_type == 'V2V':
            return self.V2V_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'V2I':
            return self.V2I_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'V2U':
            return self.V2U_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'U2U':
            return self.U2U_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'U2V':
            return self.U2V_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'U2I':
            return self.U2I_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'I2U':
            return self.I2U_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'I2V':
            return self.I2V_Rate[transmitter_idx, receiver_idx, :]
        elif channel_type == 'I2I':
            return self.I2I_Rate[transmitter_idx, receiver_idx, :]
        
    def computeRate(self, activated_task_dict):
        """Compute the rate of the activated links considering the interference.

        Args:
            activated_task_dict (dict): The activated tasks. The key is the node ID, and the value is dict {tx_idx, rx_idx, channel_type, task}.
        """
        X2I_Interference = np.zeros((self.n_RSU, self.n_RB))
        X2V_Interference = np.zeros((self.n_Veh, self.n_RB))
        X2U_Interference = np.zeros((self.n_UAV, self.n_RB))
        V2V_Signal = np.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        V2U_Signal = np.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        V2I_Signal = np.zeros((self.n_Veh, self.n_RSU, self.n_RB))
        U2U_Signal = np.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        U2V_Signal = np.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        U2I_Signal = np.zeros((self.n_UAV, self.n_RSU, self.n_RB))
        I2U_Signal = np.zeros((self.n_RSU, self.n_UAV, self.n_RB))
        I2V_Signal = np.zeros((self.n_RSU, self.n_Veh, self.n_RB))
        I2I_Signal = np.zeros((self.n_RSU, self.n_RSU, self.n_RB))
        interference_power_matrix_vtx_x2i = np.zeros((self.n_Veh, self.n_RSU, self.n_RB))
        interference_power_matrix_vtx_x2v = np.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        interference_power_matrix_vtx_x2u = np.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        interference_power_matrix_utx_x2i = np.zeros((self.n_UAV, self.n_RSU, self.n_RB))
        interference_power_matrix_utx_x2v = np.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        interference_power_matrix_utx_x2u = np.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        interference_power_matrix_itx_x2i = np.zeros((self.n_RSU, self.n_RSU, self.n_RB))
        interference_power_matrix_itx_x2v = np.zeros((self.n_RSU, self.n_Veh, self.n_RB))
        interference_power_matrix_itx_x2u = np.zeros((self.n_RSU, self.n_UAV, self.n_RB))
        # 1. 计算所有的signal, 如果信号源同时传输多个数据,信号强度叠加,当然,干扰也叠加
        # 遍历所有的车辆
        for task_id, task_profile in enumerate(activated_task_dict):
            channel_type = task_profile['channel_type']
            txidx = task_profile['tx_idx']
            rxidx = task_profile['rx_idx']
            power_db = None
            if channel_type == 'V2V':
                rb_nos = self.V2V_active_links[txidx, rxidx, :]
                V2V_Signal[txidx, rxidx, :] += 10 ** ((self.V2V_power_dB - self.V2VChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2v[txidx, rxidx, :] -= 10 ** ((self.V2V_power_dB - self.V2VChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2V_power_dB
            elif channel_type == 'V2U':
                rb_nos = self.V2U_active_links[txidx, rxidx, :]
                V2U_Signal[txidx, rxidx, :] += 10 ** ((self.V2U_power_dB - self.V2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2u[txidx, rxidx, :] -= 10 ** ((self.V2U_power_dB - self.V2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2U_power_dB
            elif channel_type == 'V2I':
                rb_nos = self.V2I_active_links[txidx, rxidx, :]
                V2I_Signal[txidx, rxidx, :] += 10 ** ((self.V2I_power_dB - self.V2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2i[txidx, rxidx, :] -= 10 ** ((self.V2I_power_dB - self.V2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.V2I_power_dB
            elif channel_type == 'U2U':
                rb_nos = self.U2U_active_links[txidx, rxidx, :]
                U2U_Signal[txidx, rxidx, :] += 10 ** ((self.U2U_power_dB - self.U2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2u[txidx, rxidx, :] -= 10 ** ((self.U2U_power_dB - self.U2UChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.U2U_power_dB
            elif channel_type == 'U2V':
                rb_nos = self.U2V_active_links[txidx, rxidx, :]
                U2V_Signal[txidx, rxidx, :] += 10 ** ((self.U2V_power_dB - self.V2UChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2v[txidx, rxidx, :] -= 10 ** ((self.U2V_power_dB - self.V2UChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.U2V_power_dB
            elif channel_type == 'U2I':
                rb_nos = self.U2I_active_links[txidx, rxidx, :]
                U2I_Signal[txidx, rxidx, :] += 10 ** ((self.U2I_power_dB - self.U2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2i[txidx, rxidx, :] -= 10 ** ((self.U2I_power_dB - self.U2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.U2I_power_dB
            elif channel_type == 'I2U': # channel有对称性，所以直接用现有的channel就行了
                rb_nos = self.I2U_active_links[txidx, rxidx, :]
                I2U_Signal[txidx, rxidx, :] += 10 ** ((self.I2U_power_dB - self.U2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2u[txidx, rxidx, :] -= 10 ** ((self.I2U_power_dB - self.U2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.I2U_power_dB
            elif channel_type == 'I2V':
                rb_nos = self.I2V_active_links[txidx, rxidx, :]
                I2V_Signal[txidx, rxidx, :] += 10 ** ((self.I2V_power_dB - self.V2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2v[txidx, rxidx, :] -= 10 ** ((self.I2V_power_dB - self.V2IChannel_with_fastfading[rxidx, txidx, :]) / 10) * rb_nos
                power_db = self.I2V_power_dB
            elif channel_type == 'I2I':
                rb_nos = self.I2I_active_links[txidx, rxidx, :]
                I2I_Signal[txidx, rxidx, :] += 10 ** ((self.I2I_power_dB - self.I2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2i[txidx, rxidx, :] -= 10 ** ((self.I2I_power_dB - self.I2IChannel_with_fastfading[txidx, rxidx, :]) / 10) * rb_nos
                power_db = self.I2I_power_dB
            if channel_type[0] == 'V':
                interference_power_matrix_vtx_x2i[txidx, :, :] += 10 ** ((power_db - self.V2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2VChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_vtx_x2u[txidx, :, :] += 10 ** ((power_db - self.V2UChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
            elif channel_type[0] == 'U':
                interference_power_matrix_utx_x2i[txidx, :, :] += 10 ** ((power_db - self.U2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2UChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos
                interference_power_matrix_utx_x2u[txidx, :, :] += 10 ** ((power_db - self.U2UChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
            elif channel_type[0] == 'I':
                interference_power_matrix_itx_x2i[txidx, :, :] += 10 ** ((power_db - self.I2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2v[txidx, :, :] += 10 ** ((power_db - self.V2IChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos
                interference_power_matrix_itx_x2u[txidx, :, :] += 10 ** ((power_db - self.U2IChannel_with_fastfading[:, txidx, :]) / 10) * rb_nos

        # 2. 分别计算每个链路对X2I, X2U, X2V的干扰，同一个RB的情况下
        # 2.1 X2I Interference
        interference_v2x_x2i = np.sum(interference_power_matrix_vtx_x2i, axis = 0) # 车辆作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_u2x_x2i = np.sum(interference_power_matrix_utx_x2i, axis = 0) # 无人机作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_i2x_x2i = np.sum(interference_power_matrix_itx_x2i, axis = 0)
        X2I_Interference = interference_v2x_x2i + interference_u2x_x2i + interference_i2x_x2i

        # 2.2 X2V Interference
        interference_v2x_x2v = np.sum(interference_power_matrix_vtx_x2v, axis = 0) # 车辆作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_u2x_x2v = np.sum(interference_power_matrix_utx_x2v, axis = 0) # 无人机作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_i2x_x2v = np.sum(interference_power_matrix_itx_x2v, axis = 0)
        X2V_Interference = interference_v2x_x2v + interference_u2x_x2v + interference_i2x_x2v

        # 2.3 X2U Interference
        interference_v2x_x2u = np.sum(interference_power_matrix_vtx_x2u, axis = 0) # 车辆作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_u2x_x2u = np.sum(interference_power_matrix_utx_x2u, axis = 0) # 无人机作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_i2x_x2u = np.sum(interference_power_matrix_itx_x2u, axis = 0)
        X2U_Interference = interference_v2x_x2u + interference_u2x_x2u + interference_i2x_x2u

        # 3. 最后再计算rate
        # 对于每一个车辆V2I的干扰，如果他自身进行了传输，那么干扰就减去自身的传输功率
        V2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        V2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        V2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_Veh, axis = 0)
        self.V2V_Interference = V2V_Interference + self.sig2
        self.V2U_Interference = V2U_Interference + self.sig2
        self.V2I_Interference = V2I_Interference + self.sig2
        self.V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference)) # bps, 小b
        self.V2I_Rate = np.log2(1 + np.divide(V2I_Signal, self.V2I_Interference))
        self.V2U_Rate = np.log2(1 + np.divide(V2U_Signal, self.V2U_Interference))
        
        U2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        U2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        U2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_UAV, axis = 0)
        self.U2U_Interference = U2U_Interference + self.sig2
        self.U2V_Interference = U2V_Interference + self.sig2
        self.U2I_Interference = U2I_Interference + self.sig2
        self.U2U_Rate = np.log2(1 + np.divide(U2U_Signal, self.U2U_Interference))
        self.U2V_Rate = np.log2(1 + np.divide(U2V_Signal, self.U2V_Interference))
        self.U2I_Rate = np.log2(1 + np.divide(U2I_Signal, self.U2I_Interference))

        I2U_Interference = np.repeat(X2U_Interference[np.newaxis, :, :], self.n_RSU, axis = 0)
        I2V_Interference = np.repeat(X2V_Interference[np.newaxis, :, :], self.n_RSU, axis = 0)
        I2I_Interference = np.repeat(X2I_Interference[np.newaxis, :, :], self.n_RSU, axis = 0)
        self.I2U_Interference = I2U_Interference + self.sig2
        self.I2V_Interference = I2V_Interference + self.sig2
        self.I2I_Interference = I2I_Interference + self.sig2
        self.I2U_Rate = np.log2(1 + np.divide(I2U_Signal, self.I2U_Interference))
        self.I2V_Rate = np.log2(1 + np.divide(I2V_Signal, self.I2V_Interference))
        self.I2I_Rate = np.log2(1 + np.divide(I2I_Signal, self.I2I_Interference))

        # 加上bandwidth
        avg_band = self.RB_bandwidth
        self.V2V_Rate = avg_band * self.V2V_Rate
        self.V2U_Rate = avg_band * self.V2U_Rate
        self.V2I_Rate = avg_band * self.V2I_Rate
        self.U2U_Rate = avg_band * self.U2U_Rate
        self.U2V_Rate = avg_band * self.U2V_Rate
        self.U2I_Rate = avg_band * self.U2I_Rate
        self.I2U_Rate = avg_band * self.I2U_Rate
        self.I2V_Rate = avg_band * self.I2V_Rate
        self.I2I_Rate = avg_band * self.I2I_Rate