#!/usr/bin/env python
#-*- coding: utf-8 -*-

"""
File Name: cnn-mnist.py
Created Time: 2017-01-12
"""


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/home/work/muou/tf_study/MNIST_data/", one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()


# Build the computation graph by creating nodes for input images and target output classes
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])



def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME')


# CONV1

# compute 32 features for each 5x5 patch;
# weight tensor will have shape [5,5,1,32]; patch size, patch size, input channels, output channels
W_conv1 = weight_variable([5, 5, 1, 32])
# bias vector with component for each output channel
b_conv1 = bias_variable([32])

# to apply the layer, need reshape x to a 4d tensor, with 2nd and 3rd dim. corresponding 
# to image width and height, and final dim. to number of color channels
x_image = tf.reshape(x, [-1, 28, 28, 1])

# We convolve x_image with the weight tensor, add bias, apply ReLU, and max pool.
# The max_pool_2x2 method will reduce the image size to 14x14
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


# CONV2

# We need to stack several conv layers to achieve deepness
# The second layer will have 64 features for each 5x5 patch
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# Feed in output from layer 1, which is result of max pool,
# and convolve with Weights and bias from this level
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# Fully Connected Layer 1

# Now the image size has been reduced to 7x7,
# we add a fully connected layer with 1024 neurons
# to allow processing on the entire image
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
# we reshape the pooling layer into a batch of vectors,
# multiply by weight matrix, add bias, apply relu
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


# Dropout

# To reduce overfitting, we apply dropout before the readout layer
# Create a placeholder for the probability that a neuron's output is 
# kept during dropout. This allows us to turn dropout ON for training,
# OFF during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# Readout layer

# Finally add a fully connected layer for readout
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# Results of readout layer is dropout output matmul with weight matrix
# of fc2 and adding bias of fc2
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2




## LOSS function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())#global_variables_initializer())
for i in range(100):
    batch = mnist.train.next_batch(5)
    if i%5 == 0:
        loss = cross_entropy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob:1.0})
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob:1.0})
        logits = y_conv.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        print('loss:', loss)
        print('logits:', logits)
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
#print("test accuracy %g"%accuracy.eval(feed_dict={ x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

