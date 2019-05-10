import os
import numpy as np
import matplotlib.pyplot as plt
from model.LogisticRegression import LogisticRegression


def load_meta_data(path, filename, shuffle=False):
    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    raw_data = lines[1:]
    data = []
    namelist = []
    for d in raw_data:
        line = []
        namelist.append(d[0])
        for i in range(1, len(d)-1):
            line.append(float(d[i]))
        data.append(line)

    namelist = np.array(namelist, dtype=np.unicode_)
    data = np.array(data, dtype=np.float32)

    folds, classid = data[:, -2].astype(np.int32), data[:, -1].astype(np.int32)

    num_data = folds.shape[0]
    if shuffle:
        perm = np.random.permutation(num_data)
        folds = folds[perm]
        classid = classid[perm]

    return namelist, folds, classid

def UrbanSound8KData(path, filename, test_fold):
    namelist, folds, classid = load_meta_data(path, filename, shuffle=False)
    folds = folds.squeeze()
    train_namelist = namelist[folds[:] != test_fold]
    test_namelist = namelist[folds[:] == test_fold]

    charar = np.empty_like(train_namelist, dtype='U10')
    charar[:] = "audio/fold"
    train_dir = folds[folds[:] != test_fold]
    train_dir = train_dir.astype(np.unicode_)
    train_dir = np.core.defchararray.add(charar, train_dir)
    charar[:] = '/'
    train_dir = np.core.defchararray.add(train_dir, charar)
    train_dir = np.core.defchararray.add(train_dir, train_namelist)

    charar = np.array(test_namelist.shape, dtype='U10')
    charar[:] = 'audio/fold'
    test_dir = folds[folds[:] == test_fold]
    test_dir = test_dir.astype(np.unicode_)
    test_dir = np.core.defchararray.add(charar, test_dir)
    charar[:] = '/'
    test_dir = np.core.defchararray.add(test_dir, charar)
    test_dir = np.core.defchararray.add(test_dir, test_namelist)

    train_classid = classid[folds[:] != test_fold]
    test_classid = classid[folds[:] == test_fold]

    return (train_dir, train_classid), (test_dir, test_classid)

def RMSE(h, y):
    if len(h.shape) > 1:
        h = h.squeeze()
    se = np.square(h - y)
    mse = np.mean(se)
    rmse = np.sqrt(mse)
    return rmse

def Accuracy(h, y):
    """
    h : (N, ), predicted label
    y : (N, ), correct label
    """
    if len(h.shape) == 1:
        h = np.expand_dims(h, 1)
    if len(y.shape) == 1:
        y = np.expand_dims(y, 1)

    total = h.shape[0]
    correct = len(np.where(h==y)[0])
    accuracy = correct / total

    return accuracy

config = {
    'UrbanSound8K': ('metadata', 'audio', LogisticRegression, Accuracy)
}

def _initialize(data_name, test_fold):
    dir_name_meta, _, model, metric = config[data_name]
    path = os.path.join('./data', dir_name_meta)

    if data_name == 'UrbanSound8K':
        train_meta_data, test_meta_data = UrbanSound8KData(path, 'UrbanSound8K.csv', test_fold)
    else:
        raise NotImplementedError

    return train_meta_data, test_meta_data, model, metric
