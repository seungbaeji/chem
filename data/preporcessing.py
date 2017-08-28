# -*- coding: utf-8 -*

from __future__ import print_function
from __future__ import division
import pandas as pd
import math
import numpy as np
from config import dli_file


class RawDli(object):

    file_path = dli_file.file_path
    sep = dli_file.sep
    index_col = dli_file.index_col

    def __init__(self):
        self._raw_dli = pd.read_csv(self.file_path, sep=self.sep, index_col=self.index_col)
        del self._raw_dli['dls']

    def normalization(self):

        def norm_func(dli_input):
            mean = dli_input.mean()
            std = dli_input.std()
            normalized_dli = (dli_input - mean) / std
            conditions = 0 < normalized_dli
            # numpy.log gives this warning when its input is negative
            logarithm_dli = [math.log(x+1) if c else -math.log(-x+1) for (c, x) in zip(conditions, normalized_dli)]
            return logarithm_dli

        result = np.array([norm_func(dli_data) for dli_data in self._raw_dli.values.T]).T
        return result


if __name__ == '__main__':
    raw_dli = RawDli()
    norm_arr = raw_dli.normalization()
    print(norm_arr.shape)

print(norm_arr[:100])
