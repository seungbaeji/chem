# -*- coding: utf-8 -*

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from preporcessing import RawDli


class DliDataSet(object):

    def __init__(self):
        self._raw_dli = RawDli()
        self._norm_arr = self._raw_dli.normalization()
        self.dli_dataset = tf.contrib.data.Dataset.from_tensor_slices(self._norm_arr)

    def next_batch(self, batch_size=512):
        batched_dataset = self.dli_dataset.batch(batch_size)
        iterator = batched_dataset.make_one_shot_iterator()
        return iterator.get_next()

    # iterator = dli_dataset.make_initialzable_iterator()
    # data_placeholder = tf.placeholder(dli_dataset.output_types, dli_dataset.output_shapes)


if __name__ == '__main__':
    with tf.Session() as sess:
        # value = sess.run(iterator.initializer, feed_dict={data_placeholder: norm_arr})
        dli_dataset = DliDataSet()
        for i in range(10):
            value = sess.run(dli_dataset.next_batch(1024))
            print(value)
