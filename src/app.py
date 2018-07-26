#!/usr/bin/env python

from os.path import join
import scipy as sp
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import tree

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


def expand_sample(elements, new_sample, target):
    for element in elements:
        if element == target:
            new_sample.append(1)
        else:
            new_sample.append(0)

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
        data_y.append(raw_data[i][12:][0].strip())

    return data_x, data_y

def sample_split(data):
    data_x = data['train_x']
    data_y = data['train_y']

    split_data = train_test_split(data_x, data_y, train_size = 0.3)
    data['train_x'], data['train_y'] = split_data[0], split_data[2]
    data['val_x'], data['val_y'] = split_data[1], split_data[3]

def DT_Train(data, m_depth, min_samples):
    train_x, train_y = data['train_x'], data['train_y']

    clf = tree.DecisionTreeClassifier(max_depth = m_depth, min_samples_leaf = min_samples)
    clf.fit(train_x, train_y)
    return clf

def DT_Test(data, clf, x, y):
    test_x = data[x]
    test_y = data[y]

    return clf.score(test_x, test_y)

def optimize_params(data):
    max_depth = range(1,31)
    min_samples_leaf = range(1,51)

    results = []

    max_acc = -1
    args_max = []
    for i in max_depth:
        for j in min_samples_leaf:
            clf = DT_Train(data, i, j)
            train_acc = DT_Test(data, clf, 'train_x', 'train_y')
            val_acc = DT_Test(data, clf, 'val_x', 'val_y')
            max_acc, args_max = (val_acc, [i,j]) if max_acc < val_acc else (max_acc, args_max)
            print max_acc, args_max
            results.append([i, j, train_acc, val_acc])
    print max_acc, args_max
    return args_max



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
    make_numeric(data)
    make_numeric(data, 'features.txt', 'test_x', 'test_y')
    # sample_split(data)
    #
    # optimize_params(data)
    clf = DT_Train(data, 10, 32)
    clf.fit(data['train_x'], data['train_y'])
    print DT_Test(data, clf, 'train_x', 'train_y')
    print DT_Test(data, clf, 'test_x', 'test_y')

if __name__ == '__main__':
    main()
