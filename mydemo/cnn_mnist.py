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

"""Trains and Evaluates the MNIST network using a feed dictionary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
#import os.path
import time
import argparse
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
'''
import mnist
'''

# Basic model parameters as external flags.
FLAGS = None

dropout = 1.0 # Dropout, probability to keep units
n_classes = 10


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=[batch_size, 28*28])
  labels_placeholder = tf.placeholder(tf.float32, shape=[batch_size, n_classes])
  return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl, keep_prob):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_prob: dropout
  }
  return feed_dict


class Network():
    def __init__(self):
        self.graph = tf.Graph()
        self.wc1 = None
        self.wc2 = None
        self.bc1 = None
        self.bc2 = None
        self.wfc1 = None
        self.bfc1 = None
        self.wfc2 = None
        self.bfc2 = None
        self.train_samples = None
        self.train_labels= None
        self.test_samples = None
        self.test_labels= None
        self.keep_prob = None
        self.logits = None
        self.loss = None
        self.optimizer = None
        self.accuracy = None

        self.merge_train_summary = None
        self.merge_test_summary = None
        self.train_summaries = []
        self.test_summaries = []

        # initiation graph
        self.def_graph()
        #self.session = tf.Session(graph=self.graph)
        self.session = tf.Session()
        print('sess:', self.session)
        #self.session = tf.Session()
        #self.write_train = tf.train.SummaryWriter('./train', self.graph)
        #self.write_train = tf.train.SummaryWriter('./train', self.session.graph)
        self.write_train = tf.summary.FileWriter(FLAGS.log_dir+'/train', self.session.graph)
        self.write_test = tf.summary.FileWriter(FLAGS.log_dir+'./test')

    def def_graph(self):
        #with self.graph.as_default():

        with tf.name_scope('inputs'):
            self.train_samples, self.train_labels = placeholder_inputs(FLAGS.batch_size)
            self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
            #image_shaped_input = tf.reshape(self.train_samples, [-1, FLAGS.batch_size, 784, 1])
            image_shaped_input = tf.reshape(self.train_samples, [-1, 28, 28, 1])
            self.train_summaries.append(tf.summary.image('input_img', image_shaped_input, 3))

        with tf.name_scope('v_conv1'):
            #self.wc1 = tf.Variable(tf.random_normal([5, 5, 1, 32]))
            #self.bc1 = tf.Variable(tf.random_normal([32]))
            self.wc1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 32], stddev=0.1))
            self.bc1 = tf.Variable(tf.constant(0.1, shape = [32]))

        with tf.name_scope('v_conv2'):
            #self.wc2 = tf.Variable(tf.random_normal([5, 5, 32, 64]))
            #self.bc2 = tf.Variable(tf.random_normal([64]))
            self.wc2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
            self.bc2 = tf.Variable(tf.constant(0.1, shape = [64]))

        with tf.name_scope('v_fc1'):
            #self.wfc1 = tf.Variable(tf.random_normal([7*7*64, 1024]))
            #self.bfc1 = tf.Variable(tf.random_normal([1024]))
            self.wfc1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
            self.bfc1 = tf.Variable(tf.constant(0.1, shape = [1024]))

        with tf.name_scope('v_fc2_out'):
            #self.wfc2 = tf.Variable(tf.random_normal([1024, n_classes]))
            #self.bfc2 = tf.Variable(tf.random_normal([n_classes]))
            self.wfc2 = tf.Variable(tf.truncated_normal([1024, n_classes], stddev=0.1))
            self.bfc2 = tf.Variable(tf.constant(0.1, shape = [n_classes]))


        self.logits = self.inference(self.train_samples, self.keep_prob)
        self.optimizer = self.training(self.logits, self.train_labels)
        self.accuracy = self.evaluation(self.logits, self.train_labels)
        #self.summary = tf.summary.merge_all()
        self.merge_train_summary = tf.merge_summary(self. train_summaries)
        #self.merge_test_summary = tf.merge_summary(self. test_summaries)


    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, filter=W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')



    # inference
    def inference(self, samples, drop):
        x = tf.reshape(samples, shape=[-1, 28, 28, 1])
        with tf.name_scope('model_conv1'):
            # Convolution Layer
            conv1 = self.conv2d(x, self.wc1, self.bc1)
            # Max Pooling (down-sampling)
            conv1 = self.maxpool2d(conv1, k=2)
            image_shaped_input = tf.reshape(conv1, [-1, 14, 14, 1])
            self.train_summaries.append(tf.summary.image('m_conv1_img', image_shaped_input, 3))

        with tf.name_scope('model_conv2'):
            # Convolution Layer
            with tf.name_scope('conv2'):
                conv2 = self.conv2d(conv1, self.wc2, self.bc2)
            # Max Pooling (down-sampling)
            with tf.name_scope('maxpool2'):
                conv2 = self.maxpool2d(conv2, k=2)
            image_shaped_input = tf.reshape(conv2, [-1, 7, 7, 1])
            self.train_summaries.append(tf.summary.image('m_conv2_img', image_shaped_input, 3))

        with tf.name_scope('model_fc1'):
            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, self.wfc1.get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, self.wfc1), self.bfc1)
            fc1 = tf.nn.relu(fc1)
            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, drop)

        with tf.name_scope('model_fc2'):
            # Output, class prediction
            logits = tf.add(tf.matmul(fc1, self.wfc2), self.bfc2)
            #logits = tf.matmul(fc1, self.wfc2) + self.bfc2
        return logits

    def training(self, pred, y):
        with tf.name_scope("train"):
            # Define loss and optimizer
            with tf.name_scope('loss'):
                #self.cost = tf.nn.softmax_cross_entropy_with_logits(pred, y)
                #c = tf.nn.softmax_cross_entropy_with_logits(pred, y)
                #self.loss = tf.reduce_mean(c)
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
            with tf.name_scope('optimizer'):
                #optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(self.loss)
                optimizer = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
                # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        self.train_summaries.append(tf.summary.scalar('loss', self.loss))
        return optimizer

    # accuracy
    def evaluation(self, pred, y):
        with tf.name_scope('evaluate'):
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        self.train_summaries.append(tf.summary.scalar('accuracy', accuracy))
        return accuracy

    def do_eval(self,sess,
                eval_correct,
                images_placeholder,
                labels_placeholder,
                keep_prob,
                data_set):
      """Runs one evaluation against the full epoch of data.

      Args:
        sess: The session in which the model has been trained.
        eval_correct: The Tensor that returns the number of correct predictions.
        images_placeholder: The images placeholder.
        labels_placeholder: The labels placeholder.
        data_set: The set of images and labels to evaluate, from
          input_data.read_data_sets().
      """
      # And run one epoch of eval.
      true_count = 0  # Counts the number of correct predictions.
      steps_per_epoch = data_set.num_examples // FLAGS.batch_size
      num_examples = steps_per_epoch * FLAGS.batch_size
      for step in xrange(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder,
                                   keep_prob,
                                   )
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
      precision = float(true_count) / num_examples
      print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
            (num_examples, true_count, precision))



    def run(self):
        with self.session as sess:
        #with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            print('run ...')
            data_sets = input_data.read_data_sets(FLAGS.input_data_dir, one_hot=True)
            # data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
            # Start the training loop.
            for step in xrange(FLAGS.max_steps):
              start_time = time.time()

              # Fill a feed dictionary with the actual set of images and labels
              # for this particular training step.
              feed_dict = fill_feed_dict(data_sets.train,
                                         self.train_samples,
                                         self.train_labels,
                                         self.keep_prob)

              # Run one step of the model.  The return values are the activations
              # from the `train_op` (which is discarded) and the `loss` Op.  To
              # inspect the values of your Ops or variables, you may include them
              # in the list passed to sess.run() and the value tensors will be
              # returned in the tuple from the call.
              #co, loss_value = sess.run([self.cost, self.loss], feed_dict=feed_dict)


              _, logits, loss_value, acc  = sess.run([self.optimizer, self.logits, self.loss, self.accuracy],
                                       feed_dict=feed_dict)
              # Write the summaries and print an overview fairly often.
              if step % FLAGS.batch_size == 0:
                duration = time.time() - start_time
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec), acc = (%.5f)' % (step, loss_value, duration, acc))
                #print('loss:', loss_value)
                #print('logits:', logits)
                # Update the events file.
                summary_str = sess.run(self.merge_train_summary, feed_dict=feed_dict)
                self.write_train.add_summary(summary_str, step)
                self.write_train.flush()
              # Save a checkpoint and evaluate the model periodically.
              #if (step + 1) % FLAGS.batch_size == 0 or (step + 1) == FLAGS.max_steps:
              if (step + 1) == FLAGS.max_steps and  False:
                #checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
                #saver.save(sess, checkpoint_file, global_step=step)
                # Evaluate against the training set.
                print('Training Data Eval:')
                self.do_eval(sess,
                        self.accuracy,
                        self.train_samples,
                        self.train_labels,
                        self.keep_prob,
                        data_sets.train)

                # Evaluate against the test set.
                print('Test Data Eval:')
                self.do_eval(sess,
                        self.accuracy,
                        self.train_samples,
                        self.train_labels,
                        self.keep_prob,
                        data_sets.test)

                '''
                # Evaluate against the validation set.
                print('Validation Data Eval:')
                do_eval(sess,
                        eval_correct,
                        images_placeholder,
                        labels_placeholder,
                        data_sets.validation)

                '''

        self.write_train.close()
        self.write_test.close()

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  net = Network()
  net.run()
  # run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--max_steps',
      type=int,
      default=100,
      help='Number of steps to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=10,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--input_data_dir',
      type=str,
      default='../MNIST_data/',
      help='Directory to put the input data.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='./log_cnn',
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
