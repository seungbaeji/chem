# -*- coding: utf-8 -*

from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
import math

from scipy.sparse import csr_matrix
from scipy import io
from array import array
from config import dli_file_info, target_file_info


class RawDli(object):

    file_path = dli_file_info.file_path
    sep = dli_file_info.sep
    index_col = dli_file_info.index_col

    def __init__(self):
        self._raw_dli = pd.read_csv(self.file_path, sep=self.sep, index_col=self.index_col)
        assert isinstance(self._raw_dli, pd.DataFrame)
        del self._raw_dli['dls']
        # input dimension of neural network is 25
        assert self._raw_dli.values.shape[1] == 25

    def __len__(self):
        return len(self._raw_dli)

    def data_frame(self):
        return self._raw_dli

    def array(self):
        return self._raw_dli.values

    def normalization(self):

        def norm_func(dli_input):
            mean = dli_input.mean()
            std = dli_input.std()
            normalized_dli = (dli_input - mean) / std
            conditions = 0 < normalized_dli
            # numpy.log gives this warning when its input is negative
            logarithm_dli = array('d', (math.log(x+1) if c else -math.log(-x+1)
                                        for (c, x) in zip(conditions, normalized_dli)))
            return logarithm_dli

        res = np.array([norm_func(dli_data) for dli_data in self._raw_dli.values.T]).T
        result = pd.DataFrame(res, index=self._raw_dli.index.values)
        return result


class RawTarget(object):

    file_path = target_file_info.file_path
    sep = target_file_info.sep
    index_col = target_file_info.index_col

    def __init__(self):
        self._raw_target = pd.read_csv(self.file_path, sep=self.sep, index_col=self.index_col)
        assert isinstance(self._raw_target, pd.DataFrame)

    def __len__(self):
        return len(self._raw_target)

    def data_frame(self):
        return self._raw_target

    def array(self):
        return self._raw_target.values

    # def


if __name__ == '__main__':
    raw_dli = RawDli()
    raw_dli_arr = raw_dli.array()
    print(raw_dli_arr.shape)
    norm_dli_df = raw_dli.normalization()
    norm_dli_df.to_csv('processed_data/dli_instance.csv', header=None)
