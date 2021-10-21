import numpy as np
from abc import ABC, abstractmethod
import tqdm

'''reference: FAU deep learning exercise'''

'''
Measures for multi-class segmentation

Input: labels and predictions must be of the same shape
    labels: numpy.ndarray, shape=[batch_size, ...]
    predictions: numpy.ndarray, shape=[batch_size, ...]
    
Available measures:
    Accuracy,
    Precision,
    Recall,
    MeanIoU,
    FrequencyWeightedIoU,
'''

# TODO: classwise evaluation


class MeasureBase(ABC):
    def __init__(self, class_names, **kwargs):
        self._class_name = class_names
        self._eval_mat = None
        self.kwargs = kwargs

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def _new_batch(self, eval_mat):
        # add the new input batch for evaluation
        pass

    def add_batch(self, eval_mat):
        # labels and predictions must be of the same shape
        self._eval_mat = eval_mat
        self._new_batch(self._eval_mat)

    @abstractmethod
    def value(self):
        # calculate the measured value
        pass

    def labels(self):
        return self._class_name

    # evaluation matrix
    @property
    def eval_mat(self):
        return self._eval_mat

    @eval_mat.setter
    def eval_mat(self, x):
        self._eval_mat = x


# def calculate_eval_matrix(num_cls, labels, predictions):
#     # labels & predictions: 1D reshaped vector
#     # return:
#     #       eval_mat[n_batch, i, j]: num of pixels of class i, predicted as class j
#     #print(labels.shape, predictions.shape)
#     assert labels.shape == predictions.shape
#
#     # convert to same data type
#     n_batch = labels.shape[0]
#
#     labels = labels.astype(np.uint8).flatten()
#     predictions = predictions.astype(np.uint8).flatten()
#
#     eval_mat = np.zeros([num_cls, num_cls])
#     for i in range(num_cls):
#         for j in range(num_cls):
#                 #eval_mat[b, i, j] = np.sum(labels==i & predictions==j)
#             eval_mat[i, j] = np.sum(np.logical_and(labels==i, predictions==j))
#     return eval_mat


def calculate_eval_matrix(num_cls, gt, eval_vol,batch_size=100):
    # flatten the inputs
    confusion = np.zeros([num_cls, num_cls])

    start = 0
    end = batch_size
    pbar = tqdm.tqdm(total=int(eval_vol.shape[0]) + 1)
    while start < eval_vol.shape[0]:
        if end > eval_vol.shape[0]: end = eval_vol.shape[0]
        gt_batch = gt[..., start: end]
        eval_vol_batch = eval_vol[..., start: end]

        gt_batch = np.array(gt_batch).flatten()
        eval_vol_batch = np.array(eval_vol_batch).flatten()
        # confusion[i, j]: num of pixels of class i, predicted as class j
        for i in range(num_cls):
            for j in range(num_cls):
                confusion[i, j] += np.sum(np.logical_and(gt_batch == i, eval_vol_batch == j))
        start = end
        end = start + batch_size
        pbar.update(1)

    pbar.close()
    return confusion.astype(np.int32)

def calculate_union(eval_mat):
    # calculate the union for IoU evaluation
    # eval_mat.shape: [n_batch, n_cls, n_cls]
    # return: union.shape = [n_batch, n_cls]
    n_batch = eval_mat.shape[0]
    n_cls = eval_mat.shape[1]

    union = np.sum(eval_mat, axis=1) + np.sum(eval_mat, axis=0) - np.diagonal(eval_mat, axis1=0, axis2=1)

    # assert union.shape[0] == n_cls
    # assert union.shape[1] == n_cls

    return union
def calculate_intersection(eval_mat):
    return np.diagonal(eval_mat, axis1=0, axis2=1)

def calculate_IoU(eval_mat):
    return np.around(calculate_intersection(eval_mat)/calculate_union(eval_mat),decimals=3)

def calculate_dice(eval_mat):
    TP=calculate_intersection(eval_mat)
    return np.around(TP*2/(calculate_union(eval_mat)+TP),decimals=3)


class Accuracy(MeasureBase):
    def __init__(self, class_names, **kwargs):
        super().__init__(class_names, **kwargs)
        self.count = None
        self.count_correct = None

        # exclude bg for better measurement
        self.no_background = kwargs.get('no_background', False)

    @property
    def name(self):
        return "Accuracy"

    def _new_batch(self, eval_mat):
        # count the correctly predicted pixels
        # eval_mat: [batch, num_cls, num_cls]
        if self.no_background is False:
            pass
        else:
            eval_mat[:, 0, :] = 0

        self.count = np.sum(eval_mat[0])
        count_correct = np.diagonal(eval_mat, axis1=1, axis2=2)
        self.count_correct = np.sum(count_correct, axis=-1)

    def value(self):
        if self.count > 0:
            return np.mean(self.count_correct)/self.count
        else:
            return 0


# class ClasswiseAccuracy(MeasureBase):
#     def __init__(self, class_names):
#         super().__init__(class_names)
#         self._class_names = class_names
#         self._count = None
#         self._count_correct = None
#
#     @property
#     def name(self):
#         return "ClasswiseAccuracy"
#
#     def _new_batch(self, eval_mat):
#
#         self._count = np.sum(eval_mat, axis=-1)
#         self._count_correct = np.diagonal(eval_mat, axis1=1, axis2=2)
#
#     def value(self):
#         res = np.zeros(len(self._class_names))
#         for i in range(len(res)):
#             res_tmp = np.divide(self._count_correct[:, i], self._count[:, i],
#                                out = np.zeros_like(self._count[:, i]), where=self._count[:, i]!=0)
#             res[i] = np.mean(res_tmp)
#
#         return res

class Precision(MeasureBase):
    def __init__(self, class_names, **kwargs):
        super().__init__(class_names, **kwargs)
        self._count = None
        self._sum = None
        self.no_background = kwargs.get('no_background', False)

    @property
    def name(self):
        return 'Precision'

    def _new_batch(self, eval_mat):
        if self.no_background is False:
            pass
        else:
            eval_mat = eval_mat[:, 1:, 1:]
        self._count = np.diagonal(eval_mat, axis1=1, axis2=2)
        self._sum = np.sum(eval_mat, axis=1)

    def value(self):
        res = np.mean(self._count/self._sum, axis=1)
        return np.mean(res)


class Recall(MeasureBase):
    def __init__(self, class_names, **kwargs):
        super().__init__(class_names, **kwargs)
        self._count = None
        self._sum = None
        self.no_background = kwargs.get('no_background', False)

    @property
    def name(self):
        return 'Recall'

    def _new_batch(self, eval_mat):
        if self.no_background is False:
            pass
        else:
            eval_mat = eval_mat[:, 1:, 1:]
        self._count = np.diagonal(eval_mat, axis1=1, axis2=2)
        self._sum = np.sum(eval_mat, axis=2)

    def value(self):
        res = np.mean(self._count/self._sum)
        return res


class MeanIoU(MeasureBase):
    def __init__(self, class_names, **kwargs):
        super().__init__(class_names, **kwargs)
        self._count_I = None
        self._count_U = None
        self.no_background = kwargs.get('no_background', False)
        self.smooth = kwargs.get('smooth', 1.0)

    @property
    def name(self):
        return 'MeanIoU'

    def _new_batch(self, eval_mat):
        self._count_I = np.diagonal(eval_mat, axis1=1, axis2=2)
        self._count_U = calculate_union(eval_mat) + self.smooth

    def value(self):
        IoU = self._count_I / self._count_U
        if self.no_background is True:
            IoU = IoU[:, 1:]
        res = np.mean(IoU, axis=1)
        return np.mean(res)


class FrequencyWeightedIoU(MeasureBase):
    def __init__(self, class_names, **kwargs):
        super(FrequencyWeightedIoU, self).__init__(class_names, **kwargs)
        self._count_I = None
        self._count_U = None
        self._count_cls = None

        self.no_background = kwargs.get('no_background', False)

    @property
    def name(self):
        return 'FrequencyWeightedIoU'

    def _new_batch(self, eval_mat):
        self._count_I = np.diagonal(eval_mat, axis1=1, axis2=2)
        self._count_U = calculate_union(eval_mat)
        self._count_cls = np.sum(eval_mat, axis=2)

    def value(self):
        IoU = self._count_I / self._count_U
        if self.no_background is True:
            IoU = IoU[:, 1:]
            self._count_cls = self._count_cls[:, 1:]

        res = np.sum(IoU * self._count_cls, axis=1) / np.sum(self._count_cls, axis=1)
        return np.mean(res)


