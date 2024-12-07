import collections
import random
import numpy as np

# ----------------------------------- #
# 经验回放池
# ----------------------------------- #
class ReplayBuffer:
    def __init__(self, buffer_size, train_min_size):
        # 创建一个队列，先进先出，队列长度不变
        self.buffer = collections.deque(maxlen=buffer_size)
        self.train_min_size = train_min_size

    # 填充经验池
    def add(self, state, action, mask, reward, next_state, next_mask, done):
        self.buffer.append((state, action, mask, reward, next_state, next_mask, done))

    # 随机采样batch组样本数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 分别取出这些数据，*获取list中的所有值
        state, action, mask, reward, next_state, next_mask, done = zip(*transitions)
        # 将state变成数组，后面方便计算
        return np.array(state), action, mask, reward, np.array(next_state), next_mask, done

    # 队列的长度
    def size(self):
        return len(self.buffer)

    def ready(self):
        return len(self.buffer) > self.train_min_size