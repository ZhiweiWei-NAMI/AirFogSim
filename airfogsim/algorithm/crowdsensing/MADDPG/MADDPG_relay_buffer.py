import collections
from collections import namedtuple
import random


class ReplayBuffer:
    def __init__(self, buffer_size, train_min_size):
        # 创建一个队列，先进先出，队列长度不变
        self.buffer_size=buffer_size
        self.train_min_size = train_min_size
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, state,action,next_state,reward):
        self.buffer.append((state,action,next_state,reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

    def ready(self):
        return len(self.buffer) > self.train_min_size