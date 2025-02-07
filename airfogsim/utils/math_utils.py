import numpy as np


def generate_2d_normal_distribution(mu_x, mu_y, sigma_x, sigma_y, rho, x_range, y_range):
    """Get x,y from a 2d normal distribution.

    Args:
        mu_x (float): X-axis mean.
        mu_y (float): Y-axis mean.
        sigma_x (float): X-axis std.
        sigma_y (float): Y-axis std.
        rho (float): Related Coefficient.
        x_range (list): X-axis range.
        y_range (list): Y-axis range.

    Returns:
        int: x
        int: y

    Examples:
        generate_2d_normal_distribution(0.5,0.5,0.03,0.03,0,[0,1],[0,1])
    """
    # 均值向量和协方差矩阵
    mean = [mu_x, mu_y]
    cov = [[sigma_x ** 2, rho * sigma_x * sigma_y],
           [rho * sigma_x * sigma_y, sigma_y ** 2]]

    # 使用numpy生成符合二维正态分布的样本
    x, y = np.random.multivariate_normal(mean, cov)

    # 对生成的点进行截断
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])

    return x, y
