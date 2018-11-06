# coding: utf-8

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
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


# Prepare filename QYJ, ZYF
filedir = 'D:/BeidaShuju/rawdata/QYJ'
fname_training_list = list(os.path.join(
    filedir, 'MultiTraining_%d_raw_tsss.fif' % j)
    for j in range(1, 6))
fname_testing_list = list(os.path.join(
    filedir, 'MultiTest_%d_raw_tsss.fif' % j)
    for j in range(1, 9))
train = True
fname_list = fname_training_list

data_X = fname_list.copy()
data_y = fname_list.copy()
for j in range(len(fname_list)):
    epochs = get_epochs(fname=fname_list[j], train=train)
    data = epochs.get_data()
    data = scale(np.mean(data, 0)[np.newaxis, :])
    data_X[j], data_y[j] = get_Xy_from_data(data)

test_id = 4
X_train = []
y_train = []
X_test = []
y_test = []
for j in range(len(fname_list)):
    if j == test_id:
        X_test = vstack(X_test, data_X[j])
        y_test = vstack(y_test, data_y[j])
        continue

    X_train = vstack(X_train, data_X[j])
    y_train = vstack(y_train, data_y[j])

# pca = PCA(n_components=10)
# pca.fit(X_train)
# X_train = pca.fit_transform(X_train)
# X_test = pca.fit_transform(X_test)

clf = SVC(kernel='linear')
clf.fit(X_train, np.ravel(y_train))

fig, axes = plt.subplots(2, 1)


def plot(X, y, axe1, axe2,
         ranges=ranges, clf=clf, epochs=epochs):
    x_ = np.array(range(ranges[0], ranges[-1]))
    axe1.plot(x_, y+0.1)
    predict = clf.predict(X)
    axe1.plot(x_, predict)
    decision = clf.decision_function(X)
    axe1.plot(x_, decision / np.std(decision))
    axe1.plot(x_, y*0)
    axe2.plot(epochs.times[ranges[0]:ranges[-1]], X)


plot(X_test, y_test, axes[0], axes[1])

plt.show()
