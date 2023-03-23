import torch.nn as nn
import torch
import torch.multiprocessing as mp
# from multiprocessing import Lock

import threading
# count_lock = threading.Lock()
# count = 0
# count_lock = Lock()


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))

class Counter():
    """enable the chief to access worker's total number of updates"""

    def __init__(self, val=True):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()
        # todo: change
        # self.lock = Lock()

    def get(self):
        # used by chief
        # with self.lock:
        # global count

        return self.val.value
        # return count

    def increment(self):
        # global count_lock
        # global count
        # # used by workers
        # with self.lock:
        #     self.val.value += 1

        # count_lock.acquire()
        # # count = count + 1
        # self.val.value += 1
        # count_lock.release()

        self.lock.acquire()
        self.val.value += 1
        self.lock.release()

    def reset(self):
        # used by chief
        # with self.lock:
        #     self.val.value = 0
        # global count
        # count = 0
        self.val.value = 0

class PerformanceCounter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self, update_threshold):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()
        self.reward = mp.Value("f", 0)
        self.update_threshold = update_threshold

    def get(self):
        # used by chief
        # with self.lock:
        if self.val.value < self.update_threshold:
            return False
        else:
            return True

    def get_reward(self):
        return self.reward.value / self.update_threshold

    def increment(self, reward):
        # # used by workers
        # with self.lock:
        #     self.val.value += 1
        self.lock.acquire()
        self.val.value += 1
        self.reward.value += reward
        self.lock.release()

    def reset(self):
        # used by chief
        # with self.lock:
        #     self.val.value = 0
        self.reward.value = 0
        self.val.value = 0

class TrafficLight:
    """used by chief to allow workers to run or not"""

    def __init__(self, val=True):
        self.val = mp.Value("b", False)
        self.lock = mp.Lock()

    def get(self):
        # with self.lock:
        return self.val.value

    def reset(self):
        # with self.lock:
        self.val.value = False

    def switch(self):
        with self.lock:
            self.val.value = (not self.val.value)
        # self.val.value = True
