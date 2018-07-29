#!/usr/bin/env python

from os.path import join
import scipy as sp
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pydot
import pydotplus
from sklearn.externals.six import StringIO

DATA_DIR = './data'

def build_feature_dict(data):
    train = data['train_x']
    transpose = []
    for i in range(len(train[0])):
        transpose.append(list(zip(*train)[i]))
    missing_feat = []
    for i in range(len(train[0])):
        categorical = Counter([s for s in transpose[i] if not s.isdigit() and s.strip() != '?'])
        continuous = np.mean([int(x) for x in transpose[i] if x.isdigit()])
        if len(categorical.keys()) > 0:
            missing_feat.append(categorical.most_common(1)[0][0])
        else:
            missing_feat.append(continuous)
    for sample in train:
        for i in range(len(sample)):
            if sample[i].strip() == '?':
                sample[i] = missing_feat[i]
    return

def expand_sample(elements, new_sample, target):
    for element in elements:
        if element == target:
            new_sample.append(1)
        else:
            new_sample.append(0)
    return

def make_numeric(data, f_name = 'features.txt', x = 'train_x', y = 'train_y'):
    features = {}
    with open(join(DATA_DIR, f_name)) as file:
        index = 0
        for line in file:
            feature_values = line.strip().split(':')[1][:-1].split(',')
            features[index] = feature_values
            index += 1

    train = data[x]
    new_samples = []
    for sample in train:
        new_sample = []
        for i in range(len(sample)):
            if type(sample[i]) == int:
                new_sample.append(sample[i])
            else:
                expand_sample(features[i], new_sample, sample[i])
        new_samples.append(new_sample)

    data[x] = new_samples
    data[y] = map(lambda x: 1 if x == '<=50K' else -1, data[y])
    return

def read_file(f_name):
    data = []
    with open('./data/' + f_name) as file:
        for line in file:
            data.append(line.strip().split(','))
    return data

def feature_transform(raw_data):
    data_x = []
    data_y = []
    for i in range(len(raw_data)):
        data_x.append(raw_data[i][:12])
        data_y.append(raw_data[i][12:][0].strip(' .\t\n'))

    return data_x, data_y

def sample_split(data):
    data_x = data['train_x']
    data_y = data['train_y']

    split_data = train_test_split(data_x, data_y, train_size = 0.3)
    data['train_x'], data['train_y'] = split_data[0], split_data[2]
    data['val_x'], data['val_y'] = split_data[1], split_data[3]
    return

def DT_Test(data, clf, x, y):
    test_x = data[x]
    test_y = data[y]
    return clf.score(test_x, test_y)

def optimize_params(data):
    train_x = data['train_x']
    train_y = data['train_y']
    max_depth = range(1,31)
    min_samples_leaf = range(1,51)

    acc_opt = -1
    max_depth_opt = 0
    min_samples_leaf_opt = 0

    graph_vals = []
    for i in max_depth:
        clf = tree.DecisionTreeClassifier(max_depth = i).fit(train_x, train_y)
        train_acc = DT_Test(data, clf, 'train_x', 'train_y')
        val_acc = DT_Test(data, clf, 'val_x', 'val_y')
        acc_opt, max_depth_opt = (val_acc, i) if acc_opt < val_acc else (acc_opt, max_depth_opt)
        graph_vals.append([i, train_acc, val_acc])

    plt.scatter(
        [x[0] for x in graph_vals],
        [x[1] for x in graph_vals]
    )
    plt.scatter(
        [x[0] for x in graph_vals],
        [x[2] for x in graph_vals]
    )
    plt.savefig('depth_plot')
    print acc_opt, max_depth_opt
    acc_opt = -1
    plt.clf()
    graph_vals = []
    for j in min_samples_leaf:
        clf = tree.DecisionTreeClassifier(min_samples_leaf = j).fit(train_x, train_y)
        train_acc = DT_Test(data, clf, 'train_x', 'train_y')
        val_acc = DT_Test(data, clf, 'val_x', 'val_y')
        acc_opt, min_samples_leaf_opt = (val_acc, j) if acc_opt < val_acc else (acc_opt, min_samples_leaf_opt)
        graph_vals.append([j, train_acc, val_acc])


    plt.scatter(
        [x[0] for x in graph_vals],
        [x[1] for x in graph_vals]
    )
    plt.scatter(
        [x[0] for x in graph_vals],
        [x[2] for x in graph_vals]
    )
    plt.savefig('leaf_plot')
    print acc_opt, min_samples_leaf_opt
    return

def main():
    data = {}
    data['train_x'], data['train_y'] = feature_transform(
        read_file(
            'adult_train.txt'
        )
    )

    data['test_x'], data['test_y'] = feature_transform(
        read_file(
        'adult_test.txt'
        )
    )

    build_feature_dict(data)
    make_numeric(
        data=data,
        f_name='features.txt',
        x='train_x',
        y='train_y'
    )
    make_numeric(
        data=data,
        f_name='features.txt',
        x='test_x',
        y='test_y'
    )
    # sample_split(data)
    # optimize_params(data)
    train_x = data['train_x']
    train_y = data['train_y']
    clf = tree.DecisionTreeClassifier(max_depth = 10, min_samples_leaf = 32, criterion = 'entropy').fit(train_x, train_y)


    dot_data = tree.export_graphviz(clf, out_file=None, max_depth=3, label="all", filled=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("tree.pdf")
    # # clf = DT_Train(data, 10, 32)
    # # clf.fit(data['train_x'], data['train_y'])
    # # print DT_Test(data, clf, 'train_x', 'train_y')
    # print DT_Test(data, clf, 'test_x', 'test_y')
    # # print data['test_y']

if __name__ == '__main__':
    main()
