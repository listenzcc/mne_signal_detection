# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
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


ranges = [250, 350, 550, 700]
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
    x_ = np.array(range(ranges[0], ranges[-1]))
    axe.plot(x_, y+0.1)
    predict = clf.predict(X)
    axe.plot(x_, predict)
    prob = clf.predict_proba(X)
    axe.plot(x_, prob)
    axe.set_title(title)


# Prepare filename QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/QYJ'
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
    epochs = get_epochs(fname=fname_list[j], train=train)
    data = epochs.get_data()
    data_X[j] = ortids.copy()
    data_y[j] = ortids.copy()
    for k in range(len(ortids)):
        data_ = data[epochs.events[:, 2] == ortids[k]]
        data_ = scale(np.mean(data_, 0)[np.newaxis, :])
        data_X[j][k], data_y[j][k] = get_Xy_from_data(data_)


def train_clf(test_run, test_ort,
              data_X=data_X, data_y=data_y):
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for j in range(len(fname_list)):
        if j == test_run:
            X_test = vstack(X_test, data_X[j][test_ort])
            y_test = vstack(y_test, data_y[j][test_ort])
            continue
        for k in range(len(ortids)):
            if test_ort == k:
                continue
            X_train = vstack(X_train, data_X[j][k])
            y_train = vstack(y_train, data_y[j][k])
    clf = LogisticRegression(solver='saga',
                             multi_class='multinomial',
                             penalty='l1')
    clf.fit(X_train, np.ravel(y_train))
    return clf, X_test, y_test, X_train, y_train


fig, axes = plt.subplots(5, 6)
for test_run in range(5):
    for test_ort in range(6):
        title = '%d, %d' % (test_run, test_ort)
        print(title)
        clf, X_test, y_test, X_train, y_train = train_clf(test_run, test_ort)
        plot(X_test, y_test, axes[test_run, test_ort],
             clf, title=title)

plt.show()
