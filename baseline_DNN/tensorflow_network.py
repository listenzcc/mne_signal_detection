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


# Init session
sess = tf.InteractiveSession()
# Placeholder x, y_
x = tf.placeholder(tf.float32, shape=[None, 102])
y_ = tf.placeholder(tf.float32, shape=[None, 7])


def set_para():
    # Layer 1
    w1 = weight_variable([102, 30])
    b1 = bias_variable([30])
    # Layer 2
    w2 = weight_variable([30, 7])
    b2 = bias_variable([7])
    return w1, b1, w2, b2


w1, b1, w2, b2 = set_para()

h1 = tf.nn.softmax(tf.matmul(x, w1) + b1)
y = tf.nn.softmax(tf.matmul(h1, w2) + b2)
# Loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# Training step, best speed 0.005
train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)


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


def train_DNN(data, label, num=50000, size=2000):
    w1, b1, w2, b2 = set_para()
    # Ready to go
    sess.run(tf.global_variables_initializer())
    print('Training')
    for j in range(num+1):
        batch = next_batch(data, label, size=size)
        feed_dict = {x: batch[0], y_: batch[1]}
        train_step.run(feed_dict=feed_dict)
        if j % 100 == 0:
            loss = sess.run(cross_entropy,
                            feed_dict=feed_dict)
            print('%d|%d, loss %f' % (j, num, loss / size))


def test_DNN(data):
    print('Testing')
    feed_dict = {x: data}
    ymat = sess.run(y, feed_dict=feed_dict)
    return freeze(ymat)
