# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

FLAGS = None

class CPrintall:
    def __enter__(self):
        self.opt = np.get_printoptions()
        #np.set_printoptions(threshold=np.nan)
        np.set_printoptions(threshold=np.inf)
    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self.opt)
    def prt(self, s):
        print(s)

def printall(*args, **kwargs):
    with CPrintall():
        print(*args, **kwargs)

def main(_):
  '''
  a = np.arange(10000)
  printall('a', a)
  print(a)
  return
  '''
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  #mnist = input_data.read_data_sets('/home/work/muou/tf_study/MNIST_data', one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  # W = tf.Variable(tf.random_uniform([784, 10], -1.0, 1.0))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # sess = tf.Session()
  # init = tf.initialize_all_variables()
  # sess.run(init)
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #print('batch_xs:'+batch_xs)
    #print('batch_ys:'+batch_ys)
    py, pstep = sess.run([y, train_step], feed_dict={x: batch_xs, y_: batch_ys})
    # print('count:', _)
    if _ % 200 == 0:
        print('count:', _)
        print('step:', pstep)
        print('py:', py)
        print('cross_entropy:', sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
        # print('y_:', batch_ys)
        #printall('W:', sess.run(W))
        # print('b:', sess.run(b))
        # print('ret:', ret);

  # Test trained model
  print('Test trained model:')
  print('W->END:', W.eval())
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print('argmax(y, 1):', sess.run(tf.argmax(y, 1), feed_dict={x:mnist.test.images}))
  print('argmax(y_, 1):', sess.run(tf.argmax(y_, 1), feed_dict={y_:mnist.test.labels}))
  print('accuracy:', sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='../MNIST_data/',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
