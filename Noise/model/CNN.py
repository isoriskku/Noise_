import os
import numpy as np
import librosa
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class CNN:
    def __init__(self):
        self.model = None

    def preprocess(self, train_dir, train_class, test_dir, test_class, b_preprocess):
        train_x = None
        test_x = None

        if b_preprocess:
            # ================= train_x, train_y ===================
            print('========== PREPROCESSING DATA ==========')
            train_x = []
            for i, s in enumerate(train_dir):
                audio_path = os.path.join('./data', s)
                y, sr = librosa.load(audio_path)
                mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
                melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
                features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
                train_x.append(features)
                if i % 50 == 0:
                    print('#', i+1, 'th data is processed')

            train_x = np.array(train_x, dtype=np.float32)

            # ===================== test_x =======================
            test_x = []
            for i, s in enumerate(test_dir):
                audio_path = os.path.join('./data', s)
                y, sr = librosa.load(audio_path)
                mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
                melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
                chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
                features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
                test_x.append(features)
                if i % 50 == 0:
                    print('#', i + 1, 'th data is processed')

            test_x = np.array(test_x, dtype=np.float32)

            # reshaping into 2d to save in csv format
            train_x_2d = np.reshape(train_x, (train_x.shape[0], train_x.shape[1] * train_x.shape[2]))
            test_x_2d = np.reshape(test_x, (test_x.shape[0], test_x.shape[1] * test_x.shape[2]))
            np.savetxt("train_data.csv", train_x_2d, delimiter=",")
            np.savetxt("test_data.csv", test_x_2d, delimiter=",")
            np.savetxt("train_labels.csv", train_class, delimiter=",")
            np.savetxt("test_labels.csv", test_class, delimiter=",")
        else:
            train_x = np.genfromtxt('train_data.csv', delimiter=',')
            train_class = np.genfromtxt('train_labels.csv', delimiter=',')
            test_x = np.genfromtxt('test_data.csv', delimiter=',')
            test_class = np.genfromtxt('test_labels.csv', delimiter=',')

        # converting to one hot
        train_class = to_categorical(train_class, num_classes=10)
        test_class = to_categorical(test_class, num_classes=10)

        # reshaping to 2D
        train_x = np.reshape(train_x, (train_x.shape[0], 40, 5))
        test_x = np.reshape(test_x, (test_x.shape[0], 40, 5))

        # reshaping to shape required by CNN
        train_x = np.reshape(train_x, (train_x.shape[0], 40, 5, 1))
        test_x = np.reshape(test_x, (test_x.shape[0], 40, 5, 1))

        return train_x, train_class, test_x, test_class

    def preproc_test(self, path):
        print('========== PREPROCESSING DATA ==========')
        y, sr = librosa.load(path)
        mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
        features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))

        return features

    def train(self, train_x, train_y, test_x, test_y):
        # forming model
        self.model = Sequential()

        # adding layers and forming the model
        self.model.add(Conv2D(64, kernel_size=5, strides=1, padding="Same", activation="relu", input_shape=(40, 5, 1)))
        self.model.add(MaxPooling2D(padding="same"))

        self.model.add(Conv2D(128, kernel_size=5, strides=1, padding="same", activation="relu"))
        self.model.add(MaxPooling2D(padding="same"))
        self.model.add(Dropout(0.3))

        self.model.add(Flatten())

        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dropout(0.3))

        self.model.add(Dense(10, activation="softmax"))

        # compiling
        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        # training the model
        print('========== TRAINING START ==========')
        self.model.fit(train_x, train_y, batch_size=50, epochs=30, validation_data=(test_x, test_y))
        print('========== TRAINING FINISH ==========')

        # model save
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("model.h5")

        return

    def load_model(self):
        # model load
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("model.h5")

        self.model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return

    def eval(self, train_x, train_y, test_x, test_y):
        print('========== EVALUATION START ==========')
        # train and test loss and scores respectively
        train_loss_score = self.model.evaluate(train_x, train_y)
        test_loss_score = self.model.evaluate(test_x, test_y)
        print(train_loss_score)
        print(test_loss_score)

        return

    def predict(self, x):
        print('========== PREDICTION START ==========')
        y = self.model.predict(x)[0]
        return y