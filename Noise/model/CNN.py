import os
import numpy as np
import librosa
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class CNN:
    def __init__(self):
        self.model = None

    def preprocess(self, path):
        print('========== PREPROCESSING DATA ==========')
        y, sr = librosa.load(path)
        mfccs = np.mean(librosa.feature.mfcc(y, sr, n_mfcc=40).T, axis=0)
        melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, fmax=8000).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cq = np.mean(librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=40).T, axis=0)
        chroma_cens = np.mean(librosa.feature.chroma_cens(y=y, sr=sr, n_chroma=40).T, axis=0)
        features = np.reshape(np.vstack((mfccs, melspectrogram, chroma_stft, chroma_cq, chroma_cens)), (40, 5))
        features = features[np.newaxis, :, :, np.newaxis]

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
        y_pred = self.model.predict_classes(x)
        return y_pred[0]
