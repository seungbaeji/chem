# -*- coding: utf-8 -*

from __future__ import print_function
from __future__ import division

import numpy as np

from sklearn.model_selection import KFold


class SplitToTrainAndValid(KFold):

    def __init__(self):
        super(SplitToTrainAndValid, self).__init__()


def k_fold_supervised(n_splits=10, selection=0, shuffle=True, random_state=0,
                      instance_array=None, label_array=None, index_array=None):
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    for i, (train_index, validation_index) in enumerate(kf.split(instance_array)):
        if i == selection:
            x_train, x_validation = instance_array[train_index], instance_array[validation_index]
            y_train, y_validation = label_array[train_index], label_array[validation_index]
            index_train, index_validation = index_array[train_index], index_array[validation_index]
            # x = zip(x_train, x_validation)
            # y = zip(y_train, y_validation)
            # index = zip(index_train, index_validation)
    return x_train, x_validation, y_train, y_validation, index_train, index_validation