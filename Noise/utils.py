import os
import numpy as np
from model.CNN import CNN

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

    # Classify the class id with dangerous one(1) and the others(0)
    for idx, y in enumerate(classid):
        if y in [0, 2, 5, 9]:
            classid[idx] = 0
        else:
            classid[idx] = 1

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

config_D = {
    'UrbanSound8K': ('metadata', 'audio')
}
config_C = {
    'CNN': CNN
}

def _initialize(data_name, classifier_name, test_fold):
    dir_name_meta, _ = config_D[data_name]
    model = config_C[classifier_name]
    path = os.path.join('./data', dir_name_meta)

    if data_name == 'UrbanSound8K':
        train_meta_data, test_meta_data = UrbanSound8KData(path, 'UrbanSound8K.csv', test_fold)
    else:
        raise NotImplementedError

    return train_meta_data, test_meta_data, model
