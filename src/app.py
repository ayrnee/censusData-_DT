#!/usr/bin/env python

from os.path import join
import scipy as sp
import numpy as np
from collections import Counter

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


def expand_sample(elements, raw_sample, target):
    for element in elements:
        if element == target:
            new_sample.append(1)
        else:
            new_sample.append(0)

def make_numeric(data, f_name = 'features.txt'):
    features = {}
    with open(join(DATA_DIR, f_name)) as file:
        index = 0
        for line in file:
            feature_values = line.strip().split(':')[1][:-1].split(',')
            features[index] = feature_values
            index += 1

    train = data['train_x']
    new_samples = []
    for sample in train:
        new_sample = []
        for i in range(len(sample)):
            if type(sample[i]) == int:
                new_sample.append(sample[i])
            else:
                new_sample.append(expand_sample(features[i], new_sample, sample[i]))
        new_samples.append(new_sample)

    data['train_x'] = new_samples
    print len(new_samples[0])






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
        data_y.append(raw_data[i][12:])
    return data_x, data_y


def main():
    data = {}
    data['train_x'], data['train_y'] = feature_transform(read_file('adult_train.txt'))
    data['test_x'], data['test_y'] = feature_transform(read_file('adult_test.txt'))
    # print train
    # print type(train)
    # print len(train) + len(test)
    build_feature_dict(data)
    make_numeric(data)





if __name__ == '__main__':
    main()
