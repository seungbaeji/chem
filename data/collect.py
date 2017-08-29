# -*- coding: utf-8 -*

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import abc

from scipy import io
from config import dli_3331_instance_file_info, dli_3331_label_file_info
from config import dli_tf_record


class CollectedDataInfo(object):

    def __init__(self, instance_file_info, label_file_info):
        """
        information for data set
        :param instance_file_info: named tuple
            information of instance file path, separator, index column
        :param label_file_info: named tuple
            information of label file path
        """
        assert len(instance_file_info) == 3 and len(label_file_info) == 3
        self._instance_path = instance_file_info.file_path
        self._instance_sep = instance_file_info.sep
        self._instance_index_col = instance_file_info.index_col
        self._label_path = label_file_info.file_path
        self._label_sep = label_file_info.sep
        self._label_index_col = label_file_info.index_col

    def __repr__(self):
        return 'instance path="{}", sep="{}", index_col="{}"\n' \
               'label_path="{}", sep="{}", index_col="{}"'\
            .format(self._instance_path, self._instance_sep, self._instance_index_col,
                    self._label_path, self._label_sep, self._label_index_col)

    @abc.abstractmethod
    def data_set(self):
        """
        DataSet of the collected data information
        :return: instance, label, index
        """
        pass


class CollectedDliDataInfo(CollectedDataInfo):

    def __init__(self, instance_file_info, label_file_info):
        super(CollectedDliDataInfo, self).__init__(instance_file_info, label_file_info)

    def data_set(self):
        _instance_df = pd.read_csv(self._instance_path, self._instance_sep, self._instance_index_col, engine='python')
        instance = _instance_df.values
        index = _instance_df.index.values
        label = io.mmread(self._label_path).todense()
        assert instance.shape[0] == index.shape[0] == label.shape[0]
        return instance, label, index




# class
#
# def tf_record_of_dli(file_path):
#     file_path = dli_tf_record.file_path


if __name__ == '__main__':
    # raw_dli = RawDli()
    # raw_target = RawTarget()
    collected_dli_data_info = CollectedDliDataInfo(dli_3331_instance_file_info, dli_3331_label_file_info)
    print(collected_dli_data_info)
    collected_data_set = collected_dli_data_info.data_set()


