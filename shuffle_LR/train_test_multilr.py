# coding: utf-8

import os
import sys
import numpy as np
import threading
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
sys.path.append('..')
from load_preprocess import get_epochs

def shuffle(X, y):
    n = len(y)
    s = np.array(range(n))
    np.random.shuffle(s)
    return X[s], y[s]


def merge(X, y, span=12, sample=5):
    mergedX = []
    mergedy = []
    for j in range(sample):
        a, b = j*span, (j*span+span)
        tmpX = scale(np.mean(X[a: b], 0)[np.newaxis, :])
        tmpy = np.mean(y[a: b], 0)[np.newaxis, :]
        mergedX = vstack(mergedX, tmpX)
        mergedy = vstack(mergedy, tmpy)
    return mergedX, mergedy

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


def get_Xy_from_data(data, ranges=ranges,
                     range_id=range_id):
    X = []
    y = []
    shape = data.shape
    for k in range(len(ranges)-1):
        left, right = ranges[k], ranges[k+1]
        id = range_id[k]
        for j in range(left, right):
            X = vstack(X, data[:, :, j])
            y = vstack(y, id+np.zeros(shape[0]).reshape(shape[0], 1))
    return X, y


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
    acc = np.count_nonzero(
        (y > 1) == (predict > 1))/len(y)
    title += ', acc %.2f' % acc
    axe.set_title(title)


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

data_XX = []
data_yy = []
for j in range(len(fname_list)):
    print(fname_list[j])
    epochs = get_epochs(fname=fname_list[j], train=train, envlop=False)
    X = epochs.get_data()
    y = epochs.events[:, 2]
    data_XX = vstack(data_XX, X)
    data_yy = vstack(data_yy, y.reshape(len(y), 1))

fig, axes = plt.subplots(5, 1)

for shuffle_id in range(5):

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for k in range(len(ortids)):
        target_yy = np.ravel(data_yy == ortids[k])
        ort_XX = data_XX[target_yy]
        ort_yy = data_yy[target_yy]
        shuffle_XX, shuffle_yy = shuffle(ort_XX, ort_yy)
        merged_XX, merged_yy = merge(shuffle_XX, shuffle_yy)
        assert(merged_XX.shape[0] == 5)
        for j in range(4):
            tmp_X, tmp_y = get_Xy_from_data(
                merged_XX[j][np.newaxis, :], range_id=[1, k+2, 1])
            X_train = vstack(X_train, tmp_X)
            y_train = vstack(y_train, tmp_y)
        j = 4
        tmp_X, tmp_y = get_Xy_from_data(
            merged_XX[j][np.newaxis, :], range_id=[1, k+2, 1])
        X_test = vstack(X_test, tmp_X)
        y_test = vstack(y_test, tmp_y)

    # MLR training and testing
    clf = LogisticRegression(multi_class='multinomial',
                             solver='newton-cg',
                             penalty='l2')
    clf.fit(X_train, np.ravel(y_train))
    title = '%d' % shuffle_id
    plot(X_test, y_test, axes[shuffle_id], clf, title=title)
    print('%d done' % shuffle_id)

plt.show()