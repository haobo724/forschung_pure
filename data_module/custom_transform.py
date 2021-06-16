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
    '''
    如果leaky是liver那么附加通道给的tag就是1
    如果leaky是lung那么附加通道给的tag就是2
    leaky list是包含要设置为缺省数据的病人的名字
    原leaky通道装的是字符串，img的完整路径名
    （已修复）好像可以优化去掉for循环直接给leaky通道赋值为对应的tag，因为既然能调用这个类那必定是缺省数据集，就不需要if判断当前数据名字是否在leakyname名单里

    '''
    def __init__(self,keys: list, leakylist, leaky):
        super().__init__(keys)
        self.leakylist=leakylist
        self.leaky=leaky
        if self.leaky =='liver':
            self.tag=1
        else:
            self.tag=2
    #Todo: 去掉for
    def __call__(self, data: dict):
        data[self.keys[0]] = self.tag
       # for key in self.keys:
       #  for leakyname in self.leakylist:
       #      if leakyname in data[self.keys[0]]:
       #          data[self.keys[0]]=self.tag
       #          return data
       #  data[self.keys[0]]=0
        return data

class LeakylabelALLFALSE(BaseTransform):
    '''

    将非缺省数据集的leaky通道设为0，传入的key为leaky
    '''
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
