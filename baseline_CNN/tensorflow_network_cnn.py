# coding: utf-8

import tensorflow as tf
import numpy as np


def weight_variable(shape):
    # init parameter weight
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # init parameter bias
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # conv2d
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_1x2(x):
    # max_pool
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                          strides=[1, 1, 2, 1], padding='SAME')


# Init session
sess = tf.InteractiveSession()
# Placeholder x, y_
x = tf.placeholder(tf.float32, shape=[None, 102, 40])
y_ = tf.placeholder(tf.float32, shape=[None, 7])

x_image = tf.reshape(x, [-1, 102, 40, 1])

w_conv1 = weight_variable([1, 5, 1, 3])
b_conv1 = bias_variable([3])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_1x2(h_conv1)

w_conv2 = weight_variable([1, 5, 3, 5])
b_conv2 = bias_variable([5])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_1x2(h_conv2)

w_fc1 = weight_variable([102*10*5, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 102*10*5])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

w_fc2 = weight_variable([1024, 7])
b_fc2 = bias_variable([7])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def one_hot(label, total=7):
    assert(max(label) < total)
    num = len(label)
    mat = np.zeros((num, total))
    for j in range(num):
        mat[j][int(label[j])] = 1
    return mat


def freeze(mat):
    label_ = np.argmax(mat, 1)
    return label_ + 1


def next_batch(data, label, size):
    num = len(label)
    assert(num > size)
    perm = np.array(range(num))
    np.random.shuffle(perm)
    select = perm[:size]
    batch = []
    batch.append(data[select])
    batch.append(one_hot(label[select]))
    return batch


def train_CNN(data, label, num=4000, size=2000):
    # Ready to go
    sess.run(tf.global_variables_initializer())
    print('Training')
    for j in range(num+1):
        batch = next_batch(data, label, size=size)
        feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
        train_step.run(feed_dict=feed_dict)
        if j % 100 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict=feed_dict)
            print('%d|%d, loss %f' % (j, num, loss / size))


def test_CNN(data):
    print('Testing')
    feed_dict = {x: data, keep_prob: 1.0}
    ymat = sess.run(y_conv, feed_dict=feed_dict)
    return freeze(ymat)
