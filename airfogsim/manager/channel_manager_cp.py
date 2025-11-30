
from ..all_channels import V2IChannel, V2UChannel, V2VChannel, U2IChannel, U2UChannel, I2IChannel
from ..channel_callback import OutageProbCallback
import os

# 检测是否使用CuPy
USE_CUPY = os.environ.get('useCUPY') == 'True'
if USE_CUPY:
    try:
        import cupy as cp
        print("✅ Using CuPy backend for GPU acceleration")
    except ImportError:
        import numpy as cp
        USE_CUPY = False
        print("❌ CuPy not available. Falling back to NumPy.")
else:
    import numpy as cp
    print("🔧 Using NumPy backend")


def batch_add_matrix(target_matrix, power_db, channel_matrix, rb_masks, tx_indices, rx_indices, inverse_flags=None):
    """
    批量版本的addMatrix操作，优化GPU性能

    Args:
        target_matrix: 目标矩阵
        power_db: 功率值
        channel_matrix: 信道矩阵
        rb_masks: RB掩码列表
        tx_indices: 发送端索引列表
        rx_indices: 接收端索引列表
        inverse_flags: 是否反转的标志列表
    """
    if len(tx_indices) == 0:
        return

    if inverse_flags is None:
        inverse_flags = [False] * len(tx_indices)

    # 批量处理每个操作
    for tx, rx, rb_mask, inverse in zip(tx_indices, rx_indices, rb_masks, inverse_flags):
        # 检查边界
        if inverse:
            if tx < channel_matrix.shape[1] and rx < channel_matrix.shape[0]:
                increment = 10 ** ((power_db - channel_matrix[rx, tx, :]) / 10) * rb_mask
                target_matrix[tx, rx, :] += increment
        else:
            if tx < channel_matrix.shape[0] and rx < channel_matrix.shape[1]:
                increment = 10 ** ((power_db - channel_matrix[tx, rx, :]) / 10) * rb_mask
                target_matrix[tx, rx, :] += increment


def batch_sub_matrix(target_matrix, power_db, channel_matrix, rb_masks, tx_indices, rx_indices, inverse_flags=None):
    """
    批量版本的subMatrix操作，优化GPU性能
    """
    if len(tx_indices) == 0:
        return

    if inverse_flags is None:
        inverse_flags = [False] * len(tx_indices)

    # 批量处理每个操作
    for tx, rx, rb_mask, inverse in zip(tx_indices, rx_indices, rb_masks, inverse_flags):
        # 检查边界
        if inverse:
            if tx < channel_matrix.shape[1] and rx < channel_matrix.shape[0]:
                decrement = 10 ** ((power_db - channel_matrix[rx, tx, :]) / 10) * rb_mask
                target_matrix[tx, rx, :] -= decrement
        else:
            if tx < channel_matrix.shape[0] and rx < channel_matrix.shape[1]:
                decrement = 10 ** ((power_db - channel_matrix[tx, rx, :]) / 10) * rb_mask
                target_matrix[tx, rx, :] -= decrement


def batch_add_two_matrix(target_matrix, power_db, channel_matrix, rb_masks, tx_indices, inverse_flags=None):
    """
    批量版本的addTwoMatrix操作，优化GPU性能
    """
    if len(tx_indices) == 0:
        return

    if inverse_flags is None:
        inverse_flags = [False] * len(tx_indices)

    # 批量处理每个操作
    for tx, rb_mask, inverse in zip(tx_indices, rb_masks, inverse_flags):
        # 检查边界
        if inverse:
            if tx < channel_matrix.shape[1]:
                increment = 10 ** ((power_db - channel_matrix[:, tx, :]) / 10) * rb_mask
                target_matrix[tx, :, :] += increment
        else:
            if tx < channel_matrix.shape[0]:
                increment = 10 ** ((power_db - channel_matrix[tx, :, :]) / 10) * rb_mask
                target_matrix[tx, :, :] += increment


class ChannelManagerCP:
    """ChannelManager is the class for managing the wireless communication channels in the airfogsim environment. It provides the APIs for the agent to interact with the channels."""
    def __init__(self, config_channel, n_RSU=1, n_UAV=1, n_Veh=1, hei_UAVs=100, RSU_positions=[], simulation_interval=0.1, 
                 n_RB=50, start_freq=2.4, RB_bandwidth=2, noise_power=-114):
        self._config_channel = config_channel
        self.V2V_power_dB = 23 # dBm 记录的都是最大功率
        self.V2I_power_dB = 26
        self.V2U_power_dB = 26
        self.U2U_power_dB = 29
        self.U2I_power_dB = 29
        self.U2V_power_dB = 29
        self.I2I_power_dB = 29
        self.I2V_power_dB = 29
        self.I2U_power_dB = 29
        self.sig2_dB = noise_power
        self.sig2 = 10**(self.sig2_dB/10)
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.V2U_Shadowing = []
        self.U2U_Shadowing = []
        self.U2V_Shadowing = []
        self.U2I_Shadowing = []
        self.veh_delta_distance = []
        self.uav_delta_distance = []
        self.RB_bandwidth = RB_bandwidth # MHz
        self.n_RB = n_RB
        self.start_freq = start_freq # GHz
        # 2.4GHz开始，共n_RB个资源块，每个资源块 RB_bandwidth MHz
        self.RB_frequencies = [self.start_freq + i * self.RB_bandwidth / 1000 for i in range(self.n_RB)]
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

        self.outageProbCallback = OutageProbCallback(config_channel['outage_model'])
        self.snr_threshold = config_channel['outage_snr_threshold']

        self.n_Veh = n_Veh
        self.n_RSU = n_RSU
        self.n_UAV = n_UAV
        self.hei_UAVs = hei_UAVs
        self.RSU_positions = RSU_positions
        self.simulation_interval = simulation_interval

        self._last_timeslot_receive = {} # node_id -> { transmit_size }
        self._last_timeslot_send = {}  # node_id -> { transmit_size }
        # 最大可容忍的断续传输时间
        self._transmission_timeout_threshold = 0.5

        self._initialize_Channels()
        self._initialize_Interference_and_active()
        self.resetActiveLinks()

    def reset(self):
        self._initialize_Channels()
        self._initialize_Interference_and_active()
        self.resetActiveLinks()

    def getNoisePower(self, is_dBm=False):
        if is_dBm:
            return self.sig2_dB
        return self.sig2

    def transmissionTimeOut(self, last_transmission_time, simulation_time):
        assert last_transmission_time >= 0 and simulation_time >= 0
        return simulation_time - last_transmission_time > self._transmission_timeout_threshold

    def _initialize_Channels(self):
        self.V2VChannel = V2VChannel(self.n_Veh, self.RB_frequencies,
                                        pathloss_type = self._config_channel.get('V2V', {}).get('pathloss_model', 'V2V_urban_tr37885'),
                                        shadowing_type = self._config_channel.get('V2V', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('V2V', {}).get('fastfading_model', 'Rayleigh'))
        self.V2IChannel = V2IChannel(self.n_Veh, self.n_RSU, self.RB_frequencies, self.RSU_positions,
                                        pathloss_type = self._config_channel.get('V2I', {}).get('pathloss_model', 'UMa_LOS_tr38901'),
                                        shadowing_type = self._config_channel.get('V2I', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('V2I', {}).get('fastfading_model', 'Rayleigh'))
        self.V2UChannel = V2UChannel(self.n_Veh, self.n_UAV, self.RB_frequencies, self.hei_UAVs,
                                        pathloss_type = self._config_channel.get('V2U', {}).get('pathloss_model', 'V2V_urban_tr37885'),
                                        shadowing_type = self._config_channel.get('V2U', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('V2U', {}).get('fastfading_model', 'Rayleigh'))
        self.U2UChannel = U2UChannel(self.n_UAV, self.RB_frequencies, self.hei_UAVs,
                                        pathloss_type = self._config_channel.get('U2U', {}).get('pathloss_model', 'free_space'),
                                        shadowing_type = self._config_channel.get('U2U', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('U2U', {}).get('fastfading_model', 'Rayleigh'))
        self.U2IChannel = U2IChannel(self.n_RSU, self.n_UAV, self.RB_frequencies, self.hei_UAVs, self.RSU_positions,
                                        pathloss_type = self._config_channel.get('U2I', {}).get('pathloss_model', 'free_space'),
                                        shadowing_type = self._config_channel.get('U2I', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('U2I', {}).get('fastfading_model', 'Rayleigh'))
        self.I2IChannel = I2IChannel(self.n_RSU, self.RB_frequencies, self.RSU_positions,
                                        pathloss_type = self._config_channel.get('I2I', {}).get('pathloss_model', 'UMa_LOS_tr38901'),
                                        shadowing_type = self._config_channel.get('I2I', {}).get('shadowing_model', '3GPP_LogNormal'),
                                        fastfading_type = self._config_channel.get('I2I', {}).get('fastfading_model', 'Rayleigh'))

    def _update_Channels(self, removed_veh_indexes, added_veh_nums):
        # removed_veh_indexes是一个个删掉的，所以不用考虑乱序
        if len(removed_veh_indexes) + added_veh_nums < 10:
            for v_index in removed_veh_indexes:
                self.V2VChannel.remove_vehicle_shadow(v_index)
            for _ in range(added_veh_nums): self.V2VChannel.add_vehicle_shadow()
            if self.V2VChannel.n_Veh != self.n_Veh:
                self.V2VChannel = V2VChannel(self.n_Veh, self.RB_frequencies)
            for v_index in removed_veh_indexes:
                self.V2IChannel.remove_vehicle_shadow(v_index)
            for _ in range(added_veh_nums): self.V2IChannel.add_vehicle_shadow()
            if self.V2IChannel.n_Veh != self.n_Veh:
                self.V2IChannel = V2IChannel(self.n_Veh, self.n_RSU, self.RB_frequencies, self.RSU_positions)
            for v_index in removed_veh_indexes:
                self.V2UChannel.remove_vehicle_shadow(v_index)
            for _ in range(added_veh_nums): self.V2UChannel.add_vehicle_shadow()
        if self.V2UChannel.n_Veh != self.n_Veh or self.V2UChannel.n_UAV != self.n_UAV:
            self.V2UChannel = V2UChannel(self.n_Veh, self.n_UAV, self.RB_frequencies, self.hei_UAVs)
        if self.U2UChannel.n_UAV != self.n_UAV:
            self.U2UChannel = U2UChannel(self.n_UAV, self.RB_frequencies, self.hei_UAVs)
        if self.U2IChannel.n_UAV != self.n_UAV:
            self.U2IChannel = U2IChannel(self.n_RSU, self.n_UAV, self.RB_frequencies, self.hei_UAVs, self.RSU_positions)


    def _initialize_Interference_and_active(self):
        self.V2I_Interference = cp.zeros((self.n_Veh, self.n_RSU)) + self.sig2
        self.V2V_Interference = cp.zeros((self.n_Veh, self.n_Veh)) + self.sig2
        self.V2U_Interference = cp.zeros((self.n_Veh, self.n_UAV)) + self.sig2
        self.U2I_Interference = cp.zeros((self.n_UAV, self.n_RSU)) + self.sig2
        self.U2V_Interference = cp.zeros((self.n_UAV, self.n_Veh)) + self.sig2
        self.U2U_Interference = cp.zeros((self.n_UAV, self.n_UAV)) + self.sig2
        self.I2I_Interference = cp.zeros((self.n_RSU, self.n_RSU)) + self.sig2
        self.I2V_Interference = cp.zeros((self.n_RSU, self.n_Veh)) + self.sig2
        self.I2U_Interference = cp.zeros((self.n_RSU, self.n_UAV)) + self.sig2

    def resetActiveLinks(self):
        self.V2V_active_links = cp.zeros((self.n_Veh, self.n_Veh, self.n_RB), dtype='bool')
        self.V2I_active_links = cp.zeros((self.n_Veh, self.n_RSU, self.n_RB), dtype='bool')
        self.V2U_active_links = cp.zeros((self.n_Veh, self.n_UAV, self.n_RB), dtype='bool')
        self.U2U_active_links = cp.zeros((self.n_UAV, self.n_UAV, self.n_RB), dtype='bool')
        self.U2V_active_links = cp.zeros((self.n_UAV, self.n_Veh, self.n_RB), dtype='bool')
        self.U2I_active_links = cp.zeros((self.n_UAV, self.n_RSU, self.n_RB), dtype='bool')
        self.I2U_active_links = cp.zeros((self.n_RSU, self.n_UAV, self.n_RB), dtype='bool')
        self.I2V_active_links = cp.zeros((self.n_RSU, self.n_Veh, self.n_RB), dtype='bool')
        self.I2I_active_links = cp.zeros((self.n_RSU, self.n_RSU, self.n_RB), dtype='bool')

    def updateNodes(self, n_Veh, n_UAV, n_RSU, removed_veh_indexes, added_veh_nums):
        self.n_Veh = n_Veh
        self.n_UAV = n_UAV
        self.n_RSU = n_RSU
        self._update_Channels(removed_veh_indexes, added_veh_nums)
        # self._initialize_Channels()
        self._initialize_Interference_and_active()
        self.resetActiveLinks()

    def updateFastFading(self, UAVs, vehicles, vid_index, uav_index):
        """Renew the channels with fast fading.

        Args:
            vehicles (dict): The vehicles in the environment.
            UAVs (dict): The UAVs in the environment.
            vid_index (list): The projection from the vehicle id to the index.
            uav_index (list): The projection from the UAV id to the index.

        """
        # vid_index is list, each element is vehicle_id, first turn vid_index to dict
        vid_index = {vid: idx for idx, vid in enumerate(vid_index)}
        uav_index = {uav: idx for idx, uav in enumerate(uav_index)}
        self._renew_channel(vehicles, UAVs, vid_index, uav_index)
        self._update_small_fading()
        self.V2VChannel_with_fastfading = self.V2VChannel_abs - self.V2VChannel.FastFading
        self.V2IChannel_with_fastfading = self.V2IChannel_abs - self.V2IChannel.FastFading
        self.V2UChannel_with_fastfading = self.V2UChannel_abs - self.V2UChannel.FastFading
        self.U2UChannel_with_fastfading = self.U2UChannel_abs - self.U2UChannel.FastFading
        self.U2IChannel_with_fastfading = self.U2IChannel_abs - self.U2IChannel.FastFading
        self.I2IChannel_with_fastfading = self.I2IChannel_abs
        # 把U2U, V2V, I2I的对角线元素设为inf.由于Channel是[tx, rx, RB]的形式，所以对角线元素是[tx, tx, RB]的形式
        self.V2VChannel_with_fastfading = cp.where(cp.eye(self.n_Veh)[:, :, cp.newaxis], cp.inf, self.V2VChannel_with_fastfading)
        self.U2UChannel_with_fastfading = cp.where(cp.eye(self.n_UAV)[:, :, cp.newaxis], cp.inf, self.U2UChannel_with_fastfading)
        self.I2IChannel_with_fastfading = cp.where(cp.eye(self.n_RSU)[:, :, cp.newaxis], cp.inf, self.I2IChannel_with_fastfading)

    def _update_small_fading(self):
        self.V2IChannel.update_fast_fading()
        self.V2VChannel.update_fast_fading()
        self.U2IChannel.update_fast_fading()
        self.U2UChannel.update_fast_fading()
        self.V2UChannel.update_fast_fading()

    def _renew_channel(self, vehicles, UAVs, vid_index, uav_index):
        veh_positions = [c.getPosition() for c in sorted(vehicles.values(), key=lambda x: vid_index[x.getId()])]
        uav_positions = [c.getPosition() for c in sorted(UAVs.values(), key=lambda x: uav_index[x.getId()])]
        self._update_large_fading(veh_positions, uav_positions, vehicles, UAVs, vid_index, uav_index)
        self.V2VChannel_abs = self.V2VChannel.PathLoss + self.V2VChannel.Shadow[:, :, cp.newaxis]
        self.V2IChannel_abs = self.V2IChannel.PathLoss + self.V2IChannel.Shadow[:, :, cp.newaxis]
        self.V2UChannel_abs = self.V2UChannel.PathLoss + self.V2UChannel.Shadow[:, :, cp.newaxis]
        self.U2UChannel_abs = self.U2UChannel.PathLoss + self.U2UChannel.Shadow[:, :, cp.newaxis]
        self.U2IChannel_abs = self.U2IChannel.PathLoss + self.U2IChannel.Shadow[:, :, cp.newaxis]
        self.I2IChannel_abs = self.I2IChannel.PathLoss + self.I2IChannel.Shadow[:, :, cp.newaxis]

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
        veh_delta_distance = self.simulation_interval * cp.asarray([c.getSpeed() for c in sorted(vehicles.values(), key=lambda x: vid_index[x.getId()])])
        uav_delta_distance = self.simulation_interval * cp.asarray([c.getSpeed() for c in sorted(UAVs.values(), key=lambda x: uav_index[x.getId()])])
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
        if transmitter_idx == receiver_idx and channel_type in ['V2V', 'U2U', 'I2I']:
            return
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

    def getRateByChannelType(self, transmitter_idx, receiver_idx, channel_type, allocated_RB_Nos = None):
        """Get the rate by the channel type.

        Args:
            transmitter_idx (int): The index of the transmitter corresponding to its type.
            receiver_idx (int): The index of the receiver corresponding to its type.
            channel_type (str): The channel type. The channel type can be 'V2V', 'V2I', 'V2U', 'U2U', 'U2V', 'U2I', 'I2U', 'I2V', 'I2I'.

        Returns:
            cp.ndarray: The rate of the channel in each RB.
        """
        if allocated_RB_Nos is None:
            allocated_RB_Nos = cp.arange(self.n_RB)
        channel_type = channel_type.lower()
        if channel_type == 'v2v':
            rate = self.V2V_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'v2i':
            rate = self.V2I_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'v2u':
            rate = self.V2U_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'u2u':
            rate = self.U2U_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'u2v':
            rate = self.U2V_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'u2i':
            rate = self.U2I_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'i2u':
            rate = self.I2U_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'i2v':
            rate = self.I2V_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        elif channel_type == 'i2i':
            rate = self.I2I_Rate[transmitter_idx, receiver_idx, allocated_RB_Nos]
        return rate

    def _get_power_db(self, channel_type):
        """获取指定信道类型的功率值"""
        power_mapping = {
            'V2V': self.V2V_power_dB,
            'V2U': self.V2U_power_dB,
            'V2I': self.V2I_power_dB,
            'U2U': self.U2U_power_dB,
            'U2V': self.U2V_power_dB,
            'U2I': self.U2I_power_dB,
            'I2U': self.I2U_power_dB,
            'I2V': self.I2V_power_dB,
            'I2I': self.I2I_power_dB,
        }
        return power_mapping.get(channel_type, 0)

    def computeRate(self, activated_task_dict):
        """Compute the rate of the activated links considering the interference.
        Optimized version with batch operations for better GPU performance.

        Args:
            activated_task_dict (dict): The activated tasks. The key is the node ID, and the value is dict {tx_idx, rx_idx, channel_type, task}.
        """
        # 初始化所有矩阵
        X2I_Interference = cp.zeros((self.n_RSU, self.n_RB))
        X2V_Interference = cp.zeros((self.n_Veh, self.n_RB))
        X2U_Interference = cp.zeros((self.n_UAV, self.n_RB))
        V2V_Signal = cp.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        V2U_Signal = cp.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        V2I_Signal = cp.zeros((self.n_Veh, self.n_RSU, self.n_RB))
        U2U_Signal = cp.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        U2V_Signal = cp.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        U2I_Signal = cp.zeros((self.n_UAV, self.n_RSU, self.n_RB))
        I2U_Signal = cp.zeros((self.n_RSU, self.n_UAV, self.n_RB))
        I2V_Signal = cp.zeros((self.n_RSU, self.n_Veh, self.n_RB))
        I2I_Signal = cp.zeros((self.n_RSU, self.n_RSU, self.n_RB))
        interference_power_matrix_vtx_x2i = cp.zeros((self.n_Veh, self.n_RSU, self.n_RB))
        interference_power_matrix_vtx_x2v = cp.zeros((self.n_Veh, self.n_Veh, self.n_RB))
        interference_power_matrix_vtx_x2u = cp.zeros((self.n_Veh, self.n_UAV, self.n_RB))
        interference_power_matrix_utx_x2i = cp.zeros((self.n_UAV, self.n_RSU, self.n_RB))
        interference_power_matrix_utx_x2v = cp.zeros((self.n_UAV, self.n_Veh, self.n_RB))
        interference_power_matrix_utx_x2u = cp.zeros((self.n_UAV, self.n_UAV, self.n_RB))
        interference_power_matrix_itx_x2i = cp.zeros((self.n_RSU, self.n_RSU, self.n_RB))
        interference_power_matrix_itx_x2v = cp.zeros((self.n_RSU, self.n_Veh, self.n_RB))
        interference_power_matrix_itx_x2u = cp.zeros((self.n_RSU, self.n_UAV, self.n_RB))

        # 1. 收集所有任务信息，按信道类型分组以便批量处理
        task_groups = {
            'V2V': {'tx': [], 'rx': [], 'rb_masks': []},
            'V2U': {'tx': [], 'rx': [], 'rb_masks': []},
            'V2I': {'tx': [], 'rx': [], 'rb_masks': []},
            'U2U': {'tx': [], 'rx': [], 'rb_masks': []},
            'U2V': {'tx': [], 'rx': [], 'rb_masks': []},
            'U2I': {'tx': [], 'rx': [], 'rb_masks': []},
            'I2U': {'tx': [], 'rx': [], 'rb_masks': []},
            'I2V': {'tx': [], 'rx': [], 'rb_masks': []},
            'I2I': {'tx': [], 'rx': [], 'rb_masks': []},
        }

        # 收集干扰计算所需的信息
        interference_tasks = {'V': [], 'U': [], 'I': []}

        for task_profile in activated_task_dict.values():
            channel_type = task_profile['channel_type']
            txidx = task_profile['tx_idx']
            rxidx = task_profile['rx_idx']
            rb_nos_idx = task_profile['RB_Nos']

            # 跳过自环连接
            if txidx == rxidx and channel_type in ['V2V', 'U2U', 'I2I']:
                continue

            # 把rb_nos从index变成对应坐标为True的形式
            rb_nos = cp.zeros(self.n_RB, dtype='bool')
            rb_nos[rb_nos_idx] = True

            # 按信道类型分组收集任务信息
            if channel_type in task_groups:
                task_groups[channel_type]['tx'].append(txidx)
                task_groups[channel_type]['rx'].append(rxidx)
                task_groups[channel_type]['rb_masks'].append(rb_nos)

            # 收集干扰计算信息
            interference_tasks[channel_type[0]].append({
                'tx': txidx,
                'rb_mask': rb_nos,
                'power_db': self._get_power_db(channel_type)
            })

        # 2. 批量处理信号计算
        # V2V信号
        if len(task_groups['V2V']['tx']) > 0:
            batch_add_matrix(V2V_Signal, self.V2V_power_dB, self.V2VChannel_with_fastfading,
                           task_groups['V2V']['rb_masks'], task_groups['V2V']['tx'], task_groups['V2V']['rx'])
            batch_sub_matrix(interference_power_matrix_vtx_x2v, self.V2V_power_dB, self.V2VChannel_with_fastfading,
                           task_groups['V2V']['rb_masks'], task_groups['V2V']['tx'], task_groups['V2V']['rx'])

        # V2U信号
        if len(task_groups['V2U']['tx']) > 0:
            batch_add_matrix(V2U_Signal, self.V2U_power_dB, self.V2UChannel_with_fastfading,
                           task_groups['V2U']['rb_masks'], task_groups['V2U']['tx'], task_groups['V2U']['rx'])
            batch_sub_matrix(interference_power_matrix_vtx_x2u, self.V2U_power_dB, self.V2UChannel_with_fastfading,
                           task_groups['V2U']['rb_masks'], task_groups['V2U']['tx'], task_groups['V2U']['rx'])

        # V2I信号
        if len(task_groups['V2I']['tx']) > 0:
            batch_add_matrix(V2I_Signal, self.V2I_power_dB, self.V2IChannel_with_fastfading,
                           task_groups['V2I']['rb_masks'], task_groups['V2I']['tx'], task_groups['V2I']['rx'])
            batch_sub_matrix(interference_power_matrix_vtx_x2i, self.V2I_power_dB, self.V2IChannel_with_fastfading,
                           task_groups['V2I']['rb_masks'], task_groups['V2I']['tx'], task_groups['V2I']['rx'])

        # U2U信号
        if len(task_groups['U2U']['tx']) > 0:
            batch_add_matrix(U2U_Signal, self.U2U_power_dB, self.U2UChannel_with_fastfading,
                           task_groups['U2U']['rb_masks'], task_groups['U2U']['tx'], task_groups['U2U']['rx'])
            batch_sub_matrix(interference_power_matrix_utx_x2u, self.U2U_power_dB, self.U2UChannel_with_fastfading,
                           task_groups['U2U']['rb_masks'], task_groups['U2U']['tx'], task_groups['U2U']['rx'])

        # U2V信号 (需要inverse=True)
        if len(task_groups['U2V']['tx']) > 0:
            inverse_flags = [True] * len(task_groups['U2V']['tx'])
            batch_add_matrix(U2V_Signal, self.U2V_power_dB, self.V2UChannel_with_fastfading,
                           task_groups['U2V']['rb_masks'], task_groups['U2V']['tx'], task_groups['U2V']['rx'], inverse_flags)
            batch_sub_matrix(interference_power_matrix_utx_x2v, self.U2V_power_dB, self.V2UChannel_with_fastfading,
                           task_groups['U2V']['rb_masks'], task_groups['U2V']['tx'], task_groups['U2V']['rx'], inverse_flags)

        # U2I信号
        if len(task_groups['U2I']['tx']) > 0:
            batch_add_matrix(U2I_Signal, self.U2I_power_dB, self.U2IChannel_with_fastfading,
                           task_groups['U2I']['rb_masks'], task_groups['U2I']['tx'], task_groups['U2I']['rx'])
            batch_sub_matrix(interference_power_matrix_utx_x2i, self.U2I_power_dB, self.U2IChannel_with_fastfading,
                           task_groups['U2I']['rb_masks'], task_groups['U2I']['tx'], task_groups['U2I']['rx'])

        # I2U信号 (需要inverse=True)
        if len(task_groups['I2U']['tx']) > 0:
            inverse_flags = [True] * len(task_groups['I2U']['tx'])
            batch_add_matrix(I2U_Signal, self.I2U_power_dB, self.U2IChannel_with_fastfading,
                           task_groups['I2U']['rb_masks'], task_groups['I2U']['tx'], task_groups['I2U']['rx'], inverse_flags)
            batch_sub_matrix(interference_power_matrix_itx_x2u, self.I2U_power_dB, self.U2IChannel_with_fastfading,
                           task_groups['I2U']['rb_masks'], task_groups['I2U']['tx'], task_groups['I2U']['rx'], inverse_flags)

        # I2V信号 (需要inverse=True)
        if len(task_groups['I2V']['tx']) > 0:
            inverse_flags = [True] * len(task_groups['I2V']['tx'])
            batch_add_matrix(I2V_Signal, self.I2V_power_dB, self.V2IChannel_with_fastfading,
                           task_groups['I2V']['rb_masks'], task_groups['I2V']['tx'], task_groups['I2V']['rx'], inverse_flags)
            batch_sub_matrix(interference_power_matrix_itx_x2v, self.I2V_power_dB, self.V2IChannel_with_fastfading,
                           task_groups['I2V']['rb_masks'], task_groups['I2V']['tx'], task_groups['I2V']['rx'], inverse_flags)

        # I2I信号
        if len(task_groups['I2I']['tx']) > 0:
            batch_add_matrix(I2I_Signal, self.I2I_power_dB, self.I2IChannel_with_fastfading,
                           task_groups['I2I']['rb_masks'], task_groups['I2I']['tx'], task_groups['I2I']['rx'])
            batch_sub_matrix(interference_power_matrix_itx_x2i, self.I2I_power_dB, self.I2IChannel_with_fastfading,
                           task_groups['I2I']['rb_masks'], task_groups['I2I']['tx'], task_groups['I2I']['rx'])

        # 3. 批量处理干扰计算
        # V类型发送端的干扰
        if len(interference_tasks['V']) > 0:
            v_tx_indices = [task['tx'] for task in interference_tasks['V']]
            v_rb_masks = [task['rb_mask'] for task in interference_tasks['V']]
            v_power_dbs = [task['power_db'] for task in interference_tasks['V']]

            for power_db, rb_mask, tx in zip(v_power_dbs, v_rb_masks, v_tx_indices):
                batch_add_two_matrix(interference_power_matrix_vtx_x2i, power_db, self.V2IChannel_with_fastfading, [rb_mask], [tx])
                batch_add_two_matrix(interference_power_matrix_vtx_x2v, power_db, self.V2VChannel_with_fastfading, [rb_mask], [tx])
                batch_add_two_matrix(interference_power_matrix_vtx_x2u, power_db, self.V2UChannel_with_fastfading, [rb_mask], [tx])

        # U类型发送端的干扰
        if len(interference_tasks['U']) > 0:
            u_tx_indices = [task['tx'] for task in interference_tasks['U']]
            u_rb_masks = [task['rb_mask'] for task in interference_tasks['U']]
            u_power_dbs = [task['power_db'] for task in interference_tasks['U']]

            for power_db, rb_mask, tx in zip(u_power_dbs, u_rb_masks, u_tx_indices):
                batch_add_two_matrix(interference_power_matrix_utx_x2i, power_db, self.U2IChannel_with_fastfading, [rb_mask], [tx])
                batch_add_two_matrix(interference_power_matrix_utx_x2v, power_db, self.V2UChannel_with_fastfading, [rb_mask], [tx], [True])
                batch_add_two_matrix(interference_power_matrix_utx_x2u, power_db, self.U2UChannel_with_fastfading, [rb_mask], [tx])

        # I类型发送端的干扰
        if len(interference_tasks['I']) > 0:
            i_tx_indices = [task['tx'] for task in interference_tasks['I']]
            i_rb_masks = [task['rb_mask'] for task in interference_tasks['I']]
            i_power_dbs = [task['power_db'] for task in interference_tasks['I']]

            for power_db, rb_mask, tx in zip(i_power_dbs, i_rb_masks, i_tx_indices):
                batch_add_two_matrix(interference_power_matrix_itx_x2i, power_db, self.I2IChannel_with_fastfading, [rb_mask], [tx])
                batch_add_two_matrix(interference_power_matrix_itx_x2v, power_db, self.V2IChannel_with_fastfading, [rb_mask], [tx], [True])
                batch_add_two_matrix(interference_power_matrix_itx_x2u, power_db, self.U2IChannel_with_fastfading, [rb_mask], [tx], [True])

        # 4. 分别计算每个链路对X2I, X2U, X2V的干扰，同一个RB的情况下
        # 2.1 X2I Interference
        interference_v2x_x2i = cp.sum(interference_power_matrix_vtx_x2i, axis = 0) # 车辆作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_u2x_x2i = cp.sum(interference_power_matrix_utx_x2i, axis = 0) # 无人机作为信源, 基站作为接收端, 所有X2I干扰的总和
        interference_i2x_x2i = cp.sum(interference_power_matrix_itx_x2i, axis = 0)
        X2I_Interference = interference_v2x_x2i + interference_u2x_x2i + interference_i2x_x2i

        # 2.2 X2V Interference
        interference_v2x_x2v = cp.sum(interference_power_matrix_vtx_x2v, axis = 0) # 车辆作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_u2x_x2v = cp.sum(interference_power_matrix_utx_x2v, axis = 0) # 无人机作为信源, 车辆作为接收端, 所有X2V干扰的总和
        interference_i2x_x2v = cp.sum(interference_power_matrix_itx_x2v, axis = 0)
        X2V_Interference = interference_v2x_x2v + interference_u2x_x2v + interference_i2x_x2v

        # 2.3 X2U Interference
        interference_v2x_x2u = cp.sum(interference_power_matrix_vtx_x2u, axis = 0) # 车辆作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_u2x_x2u = cp.sum(interference_power_matrix_utx_x2u, axis = 0) # 无人机作为信源, 无人机作为接收端, 所有X2U干扰的总和
        interference_i2x_x2u = cp.sum(interference_power_matrix_itx_x2u, axis = 0)
        X2U_Interference = interference_v2x_x2u + interference_u2x_x2u + interference_i2x_x2u

        # 5. 最后再计算rate
        # 对于每一个车辆V2I的干扰，如果他自身进行了传输，那么干扰就减去自身的传输功率
        V2V_Interference = cp.repeat(X2V_Interference[cp.newaxis, :, :], self.n_Veh, axis = 0)
        V2U_Interference = cp.repeat(X2U_Interference[cp.newaxis, :, :], self.n_Veh, axis = 0)
        V2I_Interference = cp.repeat(X2I_Interference[cp.newaxis, :, :], self.n_Veh, axis = 0)
        self.V2V_Interference = V2V_Interference + self.sig2
        self.V2U_Interference = V2U_Interference + self.sig2
        self.V2I_Interference = V2I_Interference + self.sig2
        # 添加一个小的epsilon值(1e-10)来避免除以零的问题
        self.V2V_SINR = 10*cp.log10(cp.divide(V2V_Signal + 1e-10, self.V2V_Interference + 1e-10))  # dB
        self.V2I_SINR = 10*cp.log10(cp.divide(V2I_Signal + 1e-10, self.V2I_Interference + 1e-10))
        self.V2U_SINR = 10*cp.log10(cp.divide(V2U_Signal + 1e-10, self.V2U_Interference + 1e-10))
        # 保障sinr > 1e-9
        self.V2V_SINR = cp.where(self.V2V_SINR < 1e-9, 1e-9, self.V2V_SINR)
        self.V2I_SINR = cp.where(self.V2I_SINR < 1e-9, 1e-9, self.V2I_SINR)
        self.V2U_SINR = cp.where(self.V2U_SINR < 1e-9, 1e-9, self.V2U_SINR)
        # 判断是否中断。使用随机矩阵，如果小于outage概率，那么就是中断
        V2V_sampling = cp.random.rand(*self.V2V_SINR.shape)
        V2V_outage_prob = self.outageProbCallback(self.V2V_SINR, self.snr_threshold)
        self.is_V2V_outage = V2V_sampling < V2V_outage_prob
        self.is_V2I_outage = cp.random.rand(*self.V2I_SINR.shape) < self.outageProbCallback(self.V2I_SINR, self.snr_threshold)
        self.is_V2U_outage = cp.random.rand(*self.V2U_SINR.shape) < self.outageProbCallback(self.V2U_SINR, self.snr_threshold)
        # 将dB单位的SINR转换为线性比值后再计算速率
        V2V_SINR_linear = 10**(self.V2V_SINR/10)  # 从dB转换为线性比值
        V2I_SINR_linear = 10**(self.V2I_SINR/10)
        V2U_SINR_linear = 10**(self.V2U_SINR/10)
        self.V2V_Rate = cp.log2(1 + V2V_SINR_linear) # bps, 小b
        self.V2I_Rate = cp.log2(1 + V2I_SINR_linear)
        self.V2U_Rate = cp.log2(1 + V2U_SINR_linear)
        # isOutage的部分速率设置为0
        self.V2V_Rate = cp.where(self.is_V2V_outage, 0, self.V2V_Rate)
        self.V2I_Rate = cp.where(self.is_V2I_outage, 0, self.V2I_Rate)
        self.V2U_Rate = cp.where(self.is_V2U_outage, 0, self.V2U_Rate)

        U2U_Interference = cp.repeat(X2U_Interference[cp.newaxis, :, :], self.n_UAV, axis = 0) #单位是mW
        U2V_Interference = cp.repeat(X2V_Interference[cp.newaxis, :, :], self.n_UAV, axis = 0)
        U2I_Interference = cp.repeat(X2I_Interference[cp.newaxis, :, :], self.n_UAV, axis = 0)
        self.U2U_Interference = U2U_Interference + self.sig2
        self.U2V_Interference = U2V_Interference + self.sig2
        self.U2I_Interference = U2I_Interference + self.sig2
        # 添加一个小的epsilon值(1e-10)来避免除以零的问题
        self.U2U_SINR = 10*cp.log10(cp.divide(U2U_Signal + 1e-10, self.U2U_Interference + 1e-10))
        self.U2V_SINR = 10*cp.log10(cp.divide(U2V_Signal + 1e-10, self.U2V_Interference + 1e-10))
        self.U2I_SINR = 10*cp.log10(cp.divide(U2I_Signal + 1e-10, self.U2I_Interference + 1e-10))
        # 保障sinr > 1e-9
        self.U2U_SINR = cp.where(self.U2U_SINR < 1e-9, 1e-9, self.U2U_SINR)
        self.U2V_SINR = cp.where(self.U2V_SINR < 1e-9, 1e-9, self.U2V_SINR)
        self.U2I_SINR = cp.where(self.U2I_SINR < 1e-9, 1e-9, self.U2I_SINR)
        self.is_U2U_outage = cp.random.rand(*self.U2U_SINR.shape) < self.outageProbCallback(self.U2U_SINR, self.snr_threshold)
        self.is_U2V_outage = cp.random.rand(*self.U2V_SINR.shape) < self.outageProbCallback(self.U2V_SINR, self.snr_threshold)
        self.is_U2I_outage = cp.random.rand(*self.U2I_SINR.shape) < self.outageProbCallback(self.U2I_SINR, self.snr_threshold)
        # 将dB单位的SINR转换为线性比值后再计算速率
        U2U_SINR_linear = 10**(self.U2U_SINR/10)  # 从dB转换为线性比值
        U2V_SINR_linear = 10**(self.U2V_SINR/10)
        U2I_SINR_linear = 10**(self.U2I_SINR/10)
        self.U2U_Rate = cp.log2(1 + U2U_SINR_linear)
        self.U2V_Rate = cp.log2(1 + U2V_SINR_linear)
        self.U2I_Rate = cp.log2(1 + U2I_SINR_linear)
        # is_U2I_outage 为 False的坐标
        # fU2I_index = cp.where(~self.is_U2I_outage)
        self.U2U_Rate = cp.where(self.is_U2U_outage, 0, self.U2U_Rate)
        self.U2V_Rate = cp.where(self.is_U2V_outage, 0, self.U2V_Rate)
        self.U2I_Rate = cp.where(self.is_U2I_outage, 0, self.U2I_Rate)
        # 如果u2v_rate有nan
        if cp.isnan(cp.sum(self.U2V_Rate)):
            # 获取nan的行列数
            nan_index = cp.where(cp.isnan(self.U2V_Rate))
            print(nan_index)
            print(self.U2V_SINR[nan_index])
            print(self.U2V_Interference[nan_index])
            print(U2V_Signal[nan_index])

        I2U_Interference = cp.repeat(X2U_Interference[cp.newaxis, :, :], self.n_RSU, axis = 0)
        I2V_Interference = cp.repeat(X2V_Interference[cp.newaxis, :, :], self.n_RSU, axis = 0)
        I2I_Interference = cp.repeat(X2I_Interference[cp.newaxis, :, :], self.n_RSU, axis = 0)
        self.I2U_Interference = I2U_Interference + self.sig2
        self.I2V_Interference = I2V_Interference + self.sig2
        self.I2I_Interference = I2I_Interference + self.sig2
        # 添加一个小的epsilon值(1e-10)来避免除以零的问题
        self.I2U_SINR = 10*cp.log10(cp.divide(I2U_Signal + 1e-10, self.I2U_Interference + 1e-10))
        self.I2V_SINR = 10*cp.log10(cp.divide(I2V_Signal + 1e-10, self.I2V_Interference + 1e-10))
        self.I2I_SINR = 10*cp.log10(cp.divide(I2I_Signal + 1e-10, self.I2I_Interference + 1e-10))
        # 保障sinr > 1e-9
        self.I2U_SINR = cp.where(self.I2U_SINR < 1e-9, 1e-9, self.I2U_SINR)
        self.I2V_SINR = cp.where(self.I2V_SINR < 1e-9, 1e-9, self.I2V_SINR)
        self.I2I_SINR = cp.where(self.I2I_SINR < 1e-9, 1e-9, self.I2I_SINR)
        self.is_I2U_outage = cp.random.rand(*self.I2U_SINR.shape) < self.outageProbCallback(self.I2U_SINR, self.snr_threshold)
        self.is_I2V_outage = cp.random.rand(*self.I2V_SINR.shape) < self.outageProbCallback(self.I2V_SINR, self.snr_threshold)
        self.is_I2I_outage = cp.random.rand(*self.I2I_SINR.shape) < self.outageProbCallback(self.I2I_SINR, self.snr_threshold)
        # 将dB单位的SINR转换为线性比值后再计算速率
        I2U_SINR_linear = 10**(self.I2U_SINR/10)  # 从dB转换为线性比值
        I2V_SINR_linear = 10**(self.I2V_SINR/10)
        I2I_SINR_linear = 10**(self.I2I_SINR/10)
        self.I2U_Rate = cp.log2(1 + I2U_SINR_linear)
        self.I2V_Rate = cp.log2(1 + I2V_SINR_linear)
        self.I2I_Rate = cp.log2(1 + I2I_SINR_linear)
        self.I2U_Rate = cp.where(self.is_I2U_outage, 0, self.I2U_Rate)
        self.I2V_Rate = cp.where(self.is_I2V_outage, 0, self.I2V_Rate)
        self.I2I_Rate = cp.where(self.is_I2I_outage, 0, self.I2I_Rate)

        # 加上bandwidth
        avg_band = self.RB_bandwidth
        self.V2V_Rate = avg_band * self.V2V_Rate # Mbps
        self.V2U_Rate = avg_band * self.V2U_Rate
        self.V2I_Rate = avg_band * self.V2I_Rate
        self.U2U_Rate = avg_band * self.U2U_Rate
        self.U2V_Rate = avg_band * self.U2V_Rate
        self.U2I_Rate = avg_band * self.U2I_Rate
        self.I2U_Rate = avg_band * self.I2U_Rate
        self.I2V_Rate = avg_band * self.I2V_Rate
        self.I2I_Rate = avg_band * self.I2I_Rate

    def getSignalPowerByType(self, tx_type, rx_type, is_dBm = False):
        tx_type = tx_type.upper()
        rx_type = rx_type.upper()
        power_in_db = 0
        if tx_type == 'V' and rx_type == 'V':
            power_in_db = self.V2V_power_dB
        elif tx_type == 'V' and rx_type == 'U':
            power_in_db = self.V2U_power_dB
        elif tx_type == 'V' and rx_type == 'I':
            power_in_db = self.V2I_power_dB
        elif tx_type == 'U' and rx_type == 'U':
            power_in_db = self.U2U_power_dB
        elif tx_type == 'U' and rx_type == 'V':
            power_in_db = self.U2V_power_dB
        elif tx_type == 'U' and rx_type == 'I':
            power_in_db = self.U2I_power_dB
        elif tx_type == 'I' and rx_type == 'U':
            power_in_db = self.I2U_power_dB
        elif tx_type == 'I' and rx_type == 'V':
            power_in_db = self.I2V_power_dB
        elif tx_type == 'I' and rx_type == 'I':
            power_in_db = self.I2I_power_dB
        else:
            raise ValueError("The transmitter type or the receiver type is not correct.")
        if is_dBm:
            return power_in_db
        else:
            return 10**(power_in_db/10)

    def getCSI(self, tx_idx, rx_idx, transmitter_type, receiver_type):
        """Get the channel state information.

        Args:
            tx_idx (int): The index of the transmitter.
            rx_idx (int): The index of the receiver.
            transmitter_type (str): The type of the transmitter.
            receiver_type (str): The type of the receiver.

        Returns:
            cp.ndarray: The channel state information.
        """
        transmitter_type = transmitter_type.upper()
        receiver_type = receiver_type.upper()
        assert receiver_type in ['V', 'U', 'I'] and transmitter_type in ['V', 'U', 'I']
        if transmitter_type == 'V' and receiver_type == 'V': # V2V
            return self.V2VChannel_with_fastfading[tx_idx, rx_idx]
        elif transmitter_type == 'V' and receiver_type == 'U': # V2U
            return self.V2UChannel_with_fastfading[tx_idx, rx_idx]
        elif transmitter_type == 'V' and receiver_type == 'I': # V2I
            return self.V2IChannel_with_fastfading[tx_idx, rx_idx]
        elif transmitter_type == 'U' and receiver_type == 'U': # U2U
            return self.U2UChannel_with_fastfading[tx_idx, rx_idx]
        elif transmitter_type == 'U' and receiver_type == 'V': # U2V
            return self.V2UChannel_with_fastfading[rx_idx, tx_idx]
        elif transmitter_type == 'U' and receiver_type == 'I': # U2I
            return self.U2IChannel_with_fastfading[tx_idx, rx_idx]
        elif transmitter_type == 'I' and receiver_type == 'U': # I2U
            return self.U2IChannel_with_fastfading[rx_idx, tx_idx]
        elif transmitter_type == 'I' and receiver_type == 'V': # I2V
            return self.V2IChannel_with_fastfading[rx_idx, tx_idx]
        elif transmitter_type == 'I' and receiver_type == 'I': # I2I
            return self.I2IChannel_with_fastfading[tx_idx, rx_idx]
        else:
            raise ValueError("The transmitter type or the receiver type is not correct.")

    def setThisTimeslotTransSize(self,send_size_dict:dict,receive_size_dict:dict):
        """Set send size and receive size of a node.

        Args:
            send_size_dict (dict): node_id(str) -> size(int)
            receive_size_dict (dict): node_id(str) -> size(int)
        """
        self._last_timeslot_send=send_size_dict
        self._last_timeslot_receive=receive_size_dict

    def getThisTimeslotTransSizeByNodeId(self,node_id):
        """Set send size and receive size of a node.

        Args:
            node_id (str): Node id

        Returns:
            send_size (int): Data size that send in this timeslot
            receive_size (int): Data size that receive in this timeslot
        """
        send_size=self._last_timeslot_send.get(node_id,0)
        receive_size=self._last_timeslot_receive.get(node_id,0)
        return send_size, receive_size