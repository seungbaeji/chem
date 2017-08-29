# -*- coding: utf-8 -*

from __future__ import print_function

import tensorflow as tf
import numpy as np

from preprocessing import RawDli
from config import dli_tf_record
from test import np_to_tfrecords

raw_dli = RawDli()
raw_dli_arr = raw_dli.array()

np_to_tfrecords(raw_dli_arr, raw_dli_arr, 'dli_tfr', True)


def ndarray_to_tfrecord(ndarray, file_path, verbose=True):

    def _dtype_feature(ndarray):
        """match appropriate tf.train.Feature class with dtype of ndarray. """
        assert isinstance(ndarray, np.ndarray)
        dtype_ = ndarray.dtype
        if dtype_ == np.float64 or dtype_ == np.float32:
            return lambda array: tf.train.Feature(float_list=tf.train.FloatList(value=array))
        elif dtype_ == np.int64:
            return lambda array: tf.train.Feature(int64_list=tf.train.Int64List(value=array))
        else:
            raise ValueError("The input should be numpy ndarray. \
                               Instaed got {}".format(ndarray.dtype))

    file_path = dli_tf_record.file_path
    assert isinstance(ndarray, np.ndarray)

    with tf.python_io.TFRecordWriter(file_path) as tf_writer:
        tf_writer

        tf.train.Example



