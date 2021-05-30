from abc import abstractmethod
import numpy as np
from monai.transforms.compose import Transform

class BaseTransform(Transform):
    def __init__(self, keys: list):
        self.keys = keys

    @abstractmethod
    def __call__(self, data: dict):
        return data

class Transposed(BaseTransform):
    # Transpose data dict
    def __init__(self, keys: list, indices):
        super().__init__(keys)
        self.to_indices = indices

    def __call__(self, data: dict):
        for key in self.keys:
            if len(data[key].shape) > len(self.to_indices):
                # leave the unmentioned dim unchanged
                to_indices = tuple(range(len(data[key].shape)))[len(self.to_indices):]
                to_indices = self.to_indices + to_indices
            else:
                to_indices = self.to_indices
            data[key] = data[key].transpose(to_indices)
        return data

class Leakylabel(BaseTransform):
    def __init__(self,keys: list, leakylist, leaky):
        super().__init__(keys)
        self.leakylist=leakylist
        self.leaky=leaky
        if self.leaky =='liver':
            self.tag=1
        else:
            self.tag=2
    def __call__(self, data: dict):
       # for key in self.keys:
        for leakyname in self.leakylist:
            if leakyname in data[self.keys[0]]:
                data[self.keys[0]]=self.tag
                return data
        data[self.keys[0]]=0
        return data

class LeakylabelALLFALSE(BaseTransform):
    def __init__(self,keys: list):
        super().__init__(keys)
    def __call__(self, data: dict):

        data[self.keys[0]]=0


        return data
class NormalizeLabeld(BaseTransform):
    def __init__(self, keys: list, from_list, to_list):
        super().__init__(keys)
        assert len(from_list) == len(to_list)
        self.from_list = from_list
        self.to_list = to_list

    def __call__(self, data: dict):
        for key in self.keys:
            if key in data.keys():
                for f, t in zip(self.from_list, self.to_list):
                    data[key][data[key] == f] = t
        return data
