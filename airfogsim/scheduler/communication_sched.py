import numpy as np

from .base_sched import BaseScheduler
class CommunicationScheduler(BaseScheduler):
    """The communication scheduler for channels.
    """

    @staticmethod
    def getEstimatedRateBetweenNodeIds(env, transmitter_ids, receiver_ids):
        """Get the channel state information between the transmitter and the receiver.

        Args:
            transmitter_ids (list): The list of transmitter ids in String.
            receiver_ids (list): The list of receiver ids in String.

        Returns:
            float: The channel state information.
        """
        node_rate_matrix = np.zeros((len(transmitter_ids), len(receiver_ids)))
        for transmitter_id in transmitter_ids:
            for receiver_id in receiver_ids:
                transmitter_idx = env._getNodeIdxById(transmitter_id)
                transmitter_type = env._getNodeTypeById(transmitter_id)
                receiver_idx = env._getNodeIdxById(receiver_id)
                receiver_type = env._getNodeTypeById(receiver_id)
                channel_state = env.channel_manager.getCSI(transmitter_idx, receiver_idx, transmitter_type, receiver_type)
                signal_power = env.channel_manager.getSignalPowerByType(transmitter_type, receiver_type, is_dBm=True)
                noise_power = env.channel_manager.getNoisePower(is_dBm=True)
                estimated_snr_db = signal_power - noise_power - channel_state
                estimated_snr = 10 ** (estimated_snr_db / 10)
                band = env.channel_manager.RB_bandwidth
                rate_rb = band * np.log2(1 + estimated_snr)
                rate = np.mean(rate_rb)
                node_rate_matrix[transmitter_ids.index(transmitter_id)][receiver_ids.index(receiver_id)] = rate
        # 每一行的最大值，取所有行的均值作为rate的估计
        expected_rate = np.mean(np.max(node_rate_matrix, axis=1))
        waiting_queue = env.task_manager.getOffloadingTasks()
        wait_to_offload_datasize = 0
        for task_list in waiting_queue.values():
            for task in task_list:
                wait_to_offload_datasize += max(0, task._task_size - task._transmitted_size)
        expected_wait_delay = wait_to_offload_datasize / max(0.1, expected_rate)
        return node_rate_matrix, expected_wait_delay
    
    @staticmethod
    def getNumberOfRB(env):
        """Get the number of resource blocks.

        Args:
            env (AirFogSimEnv): The environment.

        Returns:
            int: The number of resource blocks.
        """
        return env.channel_manager.n_RB
    
    @staticmethod
    def setCommunicationWithRB(env, task_id: str, RB_nos: list):
        """Set the communication with the resource blocks.

        Args:
            env (AirFogSimEnv): The environment.
            task_id (str): The task id.
            RB_nos (list): The list of resource block numbers.
        """
        # 确保RB_nos中每个数字都在0到n_RB-1之间，n_RB = getNumberOfRB(env)
        n_RB = CommunicationScheduler.getNumberOfRB(env)
        RB_nos = [RB_no % n_RB for RB_no in RB_nos]
        env.activated_offloading_tasks_with_RB_Nos[task_id] = RB_nos

    @staticmethod
    def getSumRateByChannelType(env, transmitter_idx, receiver_idx, channel_type):
        """Get the rate by the channel type.

        Args:
            transmitter_idx (int): The index of the transmitter corresponding to its type.
            receiver_idx (int): The index of the receiver corresponding to its type.
            channel_type (str): The channel type. The channel type can be 'V2V', 'V2I', 'V2U', 'U2U', 'U2V', 'U2I', 'I2U', 'I2V', 'I2I'.

        Returns:
            float: Sum of communication blocks rate
        """
        comm_rate=env.channel_manager.getRateByChannelType( transmitter_idx, receiver_idx, channel_type)
        return np.sum(comm_rate)


