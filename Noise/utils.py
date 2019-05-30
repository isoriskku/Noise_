import os
import numpy as np
import librosa
from model.CNN import CNN
from keras.utils.np_utils import to_categorical


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
        for i in range(1, len(d)):
            line.append(int(d[i]))
        data.append(line)

    namelist = np.array(namelist, dtype=np.unicode_)
    data = np.array(data, dtype=np.int32)

    folds, classid = data[:, 0], data[:, 1]

    """
    # Classify the class id with dangerous one(1) and the others(0)
    for idx, y in enumerate(classid):
        if y in [0, 2, 5, 9]:
            classid[idx] = 0
        else:
            classid[idx] = 1
    """

    num_data = folds.shape[0]
    if shuffle:
        perm = np.random.permutation(num_data)
        folds = folds[perm]
        classid = classid[perm]

    return namelist, folds, classid

def UrbanSound8KData(path, validation_fold, test_fold, b_preprocess):
    filename = ['fold1', 'fold2', 'fold3', 'fold4', 'fold5',
                'fold6', 'fold7', 'fold8', 'fold9', 'fold10']
    train_x = None
    train_class = None
    validation_x = None
    validation_class = None
    test_x = None
    test_class = None

    if b_preprocess:
        print('========== PREPROCESSING DATA ==========')
        for i in range(10):
            namelist, folds, classid = load_meta_data(path, filename[i]+".csv", shuffle=False)
            charar = np.empty_like(folds, dtype='U10')
            charar[:] = "audio/fold"
            audio_dir = folds.astype(np.unicode_)
            audio_dir = np.core.defchararray.add(charar, audio_dir)
            charar[:] = '/'
            audio_dir = np.core.defchararray.add(audio_dir, charar)
            audio_dir = np.core.defchararray.add(audio_dir, namelist)

            print('========== Fold', (i+1), '==========')
            x = []
            for idx, s in enumerate(audio_dir):
                audio_path = os.path.join('./data', s)
                y, sr = librosa.load(audio_path)
                mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
                melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
                features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
                x.append(features)
                if idx % 50 == 0:
                    print('#', idx + 1, 'th data is processed')

            x = np.array(x, dtype=np.float32)

            # reshaping into 2d to save in csv format
            x_2d = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
            np.savetxt(filename[i]+"_data.csv", x_2d, delimiter=",")
            np.savetxt(filename[i]+"_labels.csv", classid, delimiter=",")

            # Generate the train, validation, test sets
            if (i+1) in validation_fold:
                if validation_x is None:
                    validation_x = x
                    validation_class = classid
                else:
                    validation_x = np.append(validation_x, x, axis=0)
                    validation_class = np.append(validation_class, classid, axis=0)
            elif (i+1) == test_fold:
                test_x = x
                test_class = classid
            else:
                if train_x is None:
                    train_x = x
                    train_class = classid
                else:
                    train_x = np.append(train_x, x, axis=0)
                    train_class = np.append(train_class, classid, axis=0)
    else:
        for i in range(10):
            x = np.genfromtxt(filename[i] + "_data.csv", delimiter=',')
            label = np.genfromtxt(filename[i] + "_labels.csv", delimiter=',')

            if (i+1) in validation_fold:
                if validation_x is None:
                    validation_x = x
                    validation_class = label
                else:
                    validation_x = np.append(validation_x, x, axis=0)
                    validation_class = np.append(validation_class, label, axis=0)
            elif (i+1) == test_fold:
                test_x = x
                test_class = label
            else:
                if train_x is None:
                    train_x = x
                    train_class = label
                else:
                    train_x = np.append(train_x, x, axis=0)
                    train_class = np.append(train_class, label, axis=0)

    # converting to one hot
    train_class = to_categorical(train_class, num_classes=10)
    validation_class = to_categorical(validation_class, num_classes=10)
    # test_class = to_categorical(test_class, num_classes=10)

    # reshaping to 2D
    train_x = np.reshape(train_x, (train_x.shape[0], 40, 5))
    validation_x = np.reshape(validation_x, (validation_x.shape[0], 40, 5))
    test_x = np.reshape(test_x, (test_x.shape[0], 40, 5))

    # reshaping to shape required by CNN
    train_x = np.reshape(train_x, (train_x.shape[0], 40, 5, 1))
    validation_x = np.reshape(validation_x, (validation_x.shape[0], 40, 5, 1))
    test_x = np.reshape(test_x, (test_x.shape[0], 40, 5, 1))

    return (train_x, train_class), (validation_x, validation_class), (test_x, test_class)

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

config_D = {
    'UrbanSound8K': ('metadata', 'audio', Accuracy)
}
config_C = {
    'CNN': CNN
}

def _initialize(data_name, classifier_name, validation_fold, test_fold, b_preprocess):
    dir_name_meta, _, metric = config_D[data_name]
    model = config_C[classifier_name]
    path = os.path.join('./data', dir_name_meta)

    if data_name == 'UrbanSound8K':
        training_set, validation_set, test_set = UrbanSound8KData(path, validation_fold, test_fold, b_preprocess)
    else:
        raise NotImplementedError

    return training_set, validation_set, test_set, model, metric
