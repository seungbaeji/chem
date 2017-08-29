# -*- coding: utf-8 -*

from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import pandas as pd
import collections

from scipy import io
from preprocessing import RawDli, RawTarget
from config import instance_file_info, label_file_info
from config import dli_tf_record


class CollectedDataInfo(object):

    def __init__(self, instance_file_info_, label_file_info_):
        assert len(instance_file_info_) == 3 and len(label_file_info_) == 3
        self._instance_path = instance_file_info_.file_path
        self._instance_sep = instance_file_info_.sep
        self._instance_index_col = instance_file_info_.index_col
        self._label_file_path = label_file_info_.file_path

    def __repr__(self):
        return 'CollectedDataInfo(instance path="{}", sep="{}", index_col="{}" / label_path="{}")'\
            .format(self._instance_path, self._instance_sep, self._instance_index_col, self._label_file_path)

    def data_set(self):
        """
        DataSet of the collected data
        :return: instance, label, index
        """
        _instance_df = pd.read_csv(self._instance_path, self._instance_sep, self._instance_index_col, engine='python')
        instance = _instance_df.values
        index = _instance_df.index.values
        label = io.mmread(self._label_file_path).todense()
        return instance, label, index




# class
#
# def tf_record_of_dli(file_path):
#     file_path = dli_tf_record.file_path


if __name__ == '__main__':
    # raw_dli = RawDli()
    # raw_target = RawTarget()
    collected_data_info = CollectedDataInfo(instance_file_info, label_file_info)
    print(collected_data_info)
    collected_data_set = collected_data_info.data_set()


