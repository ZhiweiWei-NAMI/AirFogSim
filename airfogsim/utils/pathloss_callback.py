import os
if os.environ.get('useCUPY') == 'True':
    try:
        # 尝试导入 cupy
        import cupy as cp
    except ImportError:
        # 如果导入失败，回退到 numpy
        import numpy as np
        cp = np  # 使用 numpy 作为替代
        print("CuPy not available. Using NumPy instead.")
else:
    import numpy as cp
def addMatrix(add_ma, value, add_mb, rb_nos, txidx, rxidx, inverse=False):
    # 判断txidx和rxidx是否都在ma和mb的范围内
    if txidx < add_ma.shape[0] and rxidx < add_ma.shape[1] and txidx < add_mb.shape[0] and rxidx < add_mb.shape[1]:
        if inverse:
            increment = 10 ** ((value - add_mb[rxidx, txidx, :]) / 10) * rb_nos
        else:
            increment = 10 ** ((value - add_mb[txidx, rxidx, :]) / 10) * rb_nos
        # 更新 add_ma
        add_ma[txidx, rxidx, :] += increment

def subMatrix(sub_ma, value, sub_mb, rb_nos, txidx, rxidx, inverse=False):
    # 判断txidx和rxidx是否都在ma和mb的范围内
    if txidx < sub_ma.shape[0] and rxidx < sub_ma.shape[1] and txidx < sub_mb.shape[0] and rxidx < sub_mb.shape[1]:
        if inverse:
            decrement = 10 ** ((value - sub_mb[rxidx, txidx, :]) / 10) * rb_nos
        else:
            decrement = 10 ** ((value - sub_mb[txidx, rxidx, :]) / 10) * rb_nos
        sub_ma[txidx, rxidx, :] -= decrement

def addTwoMatrix(add_ma, value, add_mb, rb_nos, txidx, inverse=False):
    # interference_power_matrix_vtx_x2i[txidx, :, :] += 10 ** ((power_db - self.V2IChannel_with_fastfading[txidx, :, :]) / 10) * rb_nos
    if txidx < add_ma.shape[0] and txidx < add_mb.shape[0]:
        if inverse:
            increment = 10 ** ((value - add_mb[:, txidx, :]) / 10) * rb_nos
        else:
            increment = 10 ** ((value - add_mb[txidx, :, :]) / 10) * rb_nos
        add_ma[txidx, :, :] += increment

MAX_PL = 500 # The maximum path loss in dB, does not have a physical meaning

def FastFadingCallback(fastfading_type):
    """The callback function to get the fast fading.

    Args:
        fastfading_type (str): The fast fading type.

    Returns:
        function: The callback function to get the fast fading.
    """
    if fastfading_type == 'Rayleigh':
        return Rayleigh_fastfading
    else:
        raise ValueError(f"Invalid fast fading type: {fastfading_type}")
        

def PathLossCallback(pathloss_type):
    """The callback function to get the path loss.

    Args:
        pathloss_type (str): The path loss type.

    Returns:
        function: The callback function to get the path loss.
    """
    if pathloss_type == 'UMa_LOS_tr38901':
        return UMa_LOS_pathloss
    elif pathloss_type == 'V2V_highway_tr37885':
        return V2V_highway_pathloss
    elif pathloss_type == 'V2V_urban_tr37885':
        return V2V_urban_pathloss
    elif pathloss_type == 'free_space':
        return FreeSpacePathLoss
    else:
        raise ValueError(f"Invalid path loss type: {pathloss_type}")
    
def ShadowingCallback(shadowing_type):
    """The callback function to get the shadowing.

    Args:
        shadowing_type (str): The shadowing type.

    Returns:
        function: The callback function to get the shadowing.
    """
    if shadowing_type == '3GPP_LogNormal':
        return LogNormal_Shadowing
    else:
        raise ValueError(f"Invalid shadowing type: {shadowing_type}")

def LogNormal_Shadowing(preShadowing, delta_distance=None, std=4, d_correlation=50):
    """The UMa LOS shadowing model.

    Args:
        preShadowing (cp.ndarray): The pre-shadowing in dB. The shape is (num_tx, num_rx).
        delta_distance (cp.ndarray): The distance difference in meters. The shape is (num_tx, num_rx).
        n_tx (int): The number of transmitters.
        n_rx (int): The number of receivers.
        std (float): The standard deviation of the shadowing in dB. The default value is 4.
        d_correlation (float): The decorrelation distance in meters. The default value is 50.

    Returns:
        cp.ndarray: The shadowing in dB. The shape is (num_tx, num_rx).
    """
    new_shadowing = cp.random.normal(0, std, preShadowing.shape)
    if delta_distance is None:
        delta_distance = cp.zeros(preShadowing.shape)
    shadowing = 10 * cp.log10(1e-9+cp.exp(-1*(delta_distance/d_correlation))* (10 ** (preShadowing / 10)) + cp.sqrt(1-cp.exp(-2*(delta_distance/d_correlation)))*(10**(new_shadowing/10)))
    return shadowing
    
def Rayleigh_fastfading(n_tx, n_rx, n_rb, std=1):
    """The Rayleigh fast fading model.

    Args:
        n_tx (int): The number of transmitters.
        n_rx (int): The number of receivers.
        n_rb (int): The number of resource blocks.
        std (float): The standard deviation of the fast fading in dB. The default value is 1.

    Returns:
        cp.ndarray: The fast fading in dB. The shape is (n_tx, n_rx, n_rb).
    """
    # 生成两个独立的高斯分布随机变量
    gaussian1 = cp.random.normal(0, std, size=(n_tx, n_rx))
    gaussian2 = cp.random.normal(0, std, size=(n_tx, n_rx))
    # 计算瑞利分布的信道增益
    r = cp.sqrt(gaussian1 ** 2 + gaussian2 ** 2)
    # 计算信道增益的平均值
    omega = cp.mean(r ** 2)
    # 计算瑞利分布的概率密度函数
    p_r = (2 * r / omega) * cp.exp(-r ** 2 / omega)
    # 计算信道增益
    h = 10 * cp.log10(1e-9+p_r)
    # repeat n_rb times
    h = h[:, :, None] * cp.ones((1, 1, n_rb))
    return h

def FreeSpacePathLoss(tx_positions, rx_positions, frequency_ranges):
    """The free space path loss model.

    Args:
        tx_positions (cp.ndarray): The transmitter positions.
        rx_positions (cp.ndarray): The receiver positions.
        frequency_ranges (cp.ndarray): The frequency range in GHz, e.g., [3.5, 3.6].

    Returns:
        cp.ndarray: The path loss in dB. The shape is (num_tx, num_rx, num_freq).
    """
    frequency_ranges = frequency_ranges[None, None, :] * cp.ones((tx_positions.shape[0], rx_positions.shape[0], len(frequency_ranges)))
    d_2d = cp.sqrt(cp.sum((tx_positions[:, None, :] - rx_positions[None, :, :])**2, axis=-1)) # The 2D distance in meters, shape: (num_tx, num_rx)
    d_2d = d_2d[:, :, None] * cp.ones((1, 1, frequency_ranges.shape[-1]))
    path_loss = 20 * cp.log10(1e-9+d_2d) + 20 * cp.log10(1e-9+frequency_ranges) + 32.44 # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    return path_loss

def V2V_highway_pathloss(tx_positions, rx_positions, frequency_ranges, h_tx=1.5, h_rx=1.5):
    frequency_ranges = frequency_ranges[None, None, :] * cp.ones((tx_positions.shape[0], rx_positions.shape[0], len(frequency_ranges)))
    d_2d = cp.sqrt(cp.sum((tx_positions[:, None, :] - rx_positions[None, :, :])**2, axis=-1)) # The 2D distance in meters, shape: (num_tx, num_rx)
    d_2d = d_2d[:, :, None] * cp.ones((1, 1, frequency_ranges.shape[-1]))
    d_3d = cp.sqrt(d_2d**2 + (h_tx-h_rx)**2) # The 3D distance in meters, shape: (num_tx, num_rx, num_freq)
    path_loss = 32.4 + 20 * cp.log10(1e-9+d_3d) + 20 * cp.log10(1e-9+frequency_ranges) # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    return path_loss

def V2V_urban_pathloss(tx_positions, rx_positions, frequency_ranges, h_tx=1.5, h_rx=1.5):
    frequency_ranges = frequency_ranges[None, None, :] * cp.ones((tx_positions.shape[0], rx_positions.shape[0], len(frequency_ranges)))
    d_2d = cp.sqrt(cp.sum((tx_positions[:, None, :] - rx_positions[None, :, :])**2, axis=-1)) # The 2D distance in meters, shape: (num_tx, num_rx)
    d_2d = d_2d[:, :, None] * cp.ones((1, 1, frequency_ranges.shape[-1]))
    d_3d = cp.sqrt(d_2d**2 + (h_tx-h_rx)**2) # The 3D distance in meters, shape: (num_tx, num_rx, num_freq)
    path_loss = 38.77 + 16.7 * cp.log10(1e-9+d_3d) + 18.2 * cp.log10(1e-9+frequency_ranges) # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    return path_loss

def UMa_LOS_pathloss(tx_positions_cp, rx_positions_cp, frequency_ranges, h_BS=25, h_UT=1.5, h_E=1):
    """The path loss model of UMa. 3GPP TR38.901 Table 7.4.1-1. Mostly used for V2I.

    Args:
        tx_positions_cp (cp.ndarray): The transmitter positions.
        rx_positions_cp (cp.ndarray): The receiver positions.
        frequency_range (cp.ndarray): The frequency range in GHz, e.g., [3.5, 3.6].
        h_BS (float): The height of the base station in meters.
        h_UT (float): The height of the user terminal in meters.
        h_E (float): The effective environment height. The default value is 1 meter for UMa.

    Returns:
        cp.ndarray: The path loss in dB. The shape is (num_tx, num_rx, num_freq).
    """
    # 1. Calculate the distance between the transmitter and the receiver
    # frequency_ranges repeated num_tx*num_rx times, shape: (num_tx, num_rx, num_freq)
    frequency_ranges = frequency_ranges[None, None, :] * cp.ones((tx_positions_cp.shape[0], rx_positions_cp.shape[0], len(frequency_ranges)))
    d_2d = cp.sqrt(cp.sum((tx_positions_cp[:, None, :] - rx_positions_cp[None, :, :])**2, axis=-1)) # The 2D distance in meters, shape: (num_tx, num_rx)
    # d_2d repeated len(frequency_ranges) times, shape: (num_tx, num_rx, num_freq)
    d_2d = d_2d[:, :, None] * cp.ones((1, 1, frequency_ranges.shape[-1]))
    d_3d = cp.sqrt(d_2d**2 + (h_BS-h_UT)**2) # The 3D distance in meters, shape: (num_tx, num_rx)
    # d_3d repeated len(frequency_ranges) times, shape: (num_tx, num_rx, num_freq)
    d_bp = 4 * (h_BS-h_E) * (h_UT-h_E) * frequency_ranges * 1e9 / 3e8 # Breakpoint distance in meters, shape: (num_freq,)
    # if 10m <= d_2d <= d_bp, then pathloss = PL1; if d_bp < d_2d <= 5000m, then pathloss = PL2; if d_2d < 10m, then pathloss = 0; if d_2d > 5000m, then pathloss = MAX_PL
    PL1 = 28 + 22 * cp.log10(1e-9+d_3d) + 20 * cp.log10(1e-9+frequency_ranges) # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    PL2 = PL1 + 18 * cp.log10(1e-9+d_3d) - 9 * cp.log10(1e-9+d_bp**2 + (h_BS-h_UT)**2) # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    path_loss = cp.where(d_2d < 10, 0, cp.where(d_2d <= d_bp, PL1, cp.where(d_2d <= 5000, PL2, MAX_PL))) # The path loss in dB, shape: (num_tx, num_rx, num_freq)
    return path_loss


if __name__ == '__main__':
    # 1. Test the UMa_LOS_pathloss
    # Random The transmitter positions, shape: (num_tx, 200)
    tx_positions_cp = cp.random.rand(200, 2) * 100 # The transmitter positions, shape: (num_tx, 2)
    rx_positions_cp = cp.random.rand(200, 2) * 100 # The receiver positions, shape: (num_rx, 2)
    frequency_ranges_cp = cp.linspace(2.4, 2.5, 100) # The frequency range in GHz
    import time
    start_time = time.time()
    path_loss = UMa_LOS_pathloss(tx_positions_cp, rx_positions_cp, frequency_ranges_cp)
    print(f"UMa_LOS_cp_pathloss time: {time.time()-start_time}")
    # print(f"UMa_LOS_cp_pathloss: {path_loss}")
    