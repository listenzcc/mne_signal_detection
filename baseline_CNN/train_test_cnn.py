# coding: utf-8

import os
import sys
import numpy as np
import threading
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tensorflow_network_cnn import train_CNN, test_CNN, restore_CNN
sys.path.append('..')
from load_preprocess import get_epochs


def vstack(a, b):
    if len(a) == 0:
        return b
    return np.vstack((a, b))


def scale(data):
    shape = data.shape
    for j in range(shape[0]):
        for k in range(shape[1]):
            baseline = data[j, k, 0:250]
            m = np.mean(baseline)
            s = np.std(baseline)
            data[j][k] = (data[j][k] - m) / s
    return data


ranges = [250, 350, 550, 650]
range_id = [1, 2, 1]


def get_Xy_from_data(data, ranges=ranges, span=100,
                     range_id=range_id):
    X = []
    y = []
    shape = data.shape
    data = shift(data, ranges=ranges)
    # test if shift works
    # tested it works
    # fig, axes = plt.subplots(2, 1)
    # axes[0].plot(np.mean(data, 0).transpose())
    # axes[1].plot(np.mean(data_, 0).transpose())
    # plt.show()
    # assert(1 == 0)
    for k in range(len(ranges)-1):
        left, right = ranges[k], ranges[k+1]
        id = range_id[k]
        for j in range(left, right):
            X = vstack(X, data[:, :, j-span:j])
            y = vstack(y, id+np.zeros(shape[0]).reshape(shape[0], 1))
    return X, y


def shift(data, ranges=ranges):
    # I do not think this can be data driven method
    assert(ranges == [250, 350, 550, 650])
    ref = np.zeros(500)
    for j in range(200, 300):
        ref[j] = 1

    shape = data.shape
    s = np.zeros([shape[0], shape[1]])
    for j in range(shape[0]):
        for k in range(shape[1]):
            d = data[j][k][200:700]
            c = np.correlate(d, ref, 'same')
            argmax = np.argmax(c)
            # we assume center is on 200,
            # equals to 400 in original range
            s_ = argmax - 200
            if s_ > 0:
                # if s_ larger than center, left shift
                d = data[j][k][s_::].copy()
                data[j][k][0: len(d)] = d.copy()
            if s_ < 0:
                # if s_ smaller than center, right shift
                # only shift 1000 points, enough now
                data[j][k][-s_:-s_+1000] = data[j][k][0:1000].copy()

    return data


def plot(X, y, axe, clf, title='title'):
    axe.plot(y+0.1)
    predict = clf.predict(X)
    axe.plot(predict)
    prob = clf.predict_proba(X) - 1
    axe.plot(prob)
    for j in range(0, len(y), 400):
        axe.plot([j, j], list(axe.get_ylim()),
                 color='gray')
    y = np.ravel(y)
    predict = np.ravel(predict)


# check model_save_path
model_path = 'model_save_path'
assert(not(os.path.exists(model_path)))
os.mkdir(model_path)

# Prepare filename QYJ, ZYF
filedir = 'd:/BeidaShuju/rawdata/QYJ'
fname_training_list = list(os.path.join(
    filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
    for j in range(1, 6))
fname_testing_list = list(os.path.join(
    filedir, 'MultiTest_%d_raw_tsss.fif' % j)
    for j in range(1, 9))
ortids_training = [2, 6, 9, 14, 17, 33]
ortids_testing = [8, 16, 32, 64]
train = True
ortids = ortids_training
fname_list = fname_training_list

data_X = fname_list.copy()
data_y = fname_list.copy()
for j in range(len(fname_list)):
    print(fname_list[j])
    epochs = get_epochs(fname=fname_list[j], train=train, envlop=True)
    data = epochs.get_data()
    data_X[j] = ortids.copy()
    data_y[j] = ortids.copy()
    for k in range(len(ortids)):
        data_ = data[epochs.events[:, 2] == ortids[k]]
        data_ = scale(np.mean(data_, 0)[np.newaxis, :])
        data_X[j][k], data_y[j][k] = get_Xy_from_data(
            data_, range_id=[1, k+2, 1])

fig, axes = plt.subplots(5, 1)

save_path = os.path.join('model_save_path', 'QYJ_%d')
model_name = 'CNNmodel'
for test_run in range(5):
    model_path = os.path.join(save_path % test_run, model_name)
    # test_run = 3
    # data prepare
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for j in range(len(fname_list)):
        if j == test_run:
            for k in range(len(ortids)):
                X_test = vstack(X_test, data_X[j][k])
                y_test = vstack(y_test, data_y[j][k])
            continue
        for k in range(len(ortids)):
            X_train = vstack(X_train, data_X[j][k])
            y_train = vstack(y_train, data_y[j][k])

    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)

    # CNN training and testing
    train_CNN(X_train, y_train-1, model_path=model_path)
    # restore_CNN(model_path=model_path)
    y_guess = test_CNN(X_test)

    # Plot
    axe = axes[test_run]
    axe.plot(y_test)
    axe.plot(y_guess)
    acc = np.count_nonzero(
        (y_test > 1) == (y_guess > 1))/len(y_test)
    title = '%d, acc %.2f' % (test_run, acc)
    axe.set_title(title)

plt.show()
