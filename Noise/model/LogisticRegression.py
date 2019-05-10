import os
import numpy as np
import librosa
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
from sklearn.preprocessing import maxabs_scale

class LogisticRegression:
    def __init__(self, num_features, num_label):
        self.num_features = num_features
        self.num_label = num_label
        self.W = np.zeros((self.num_features, self.num_label))
        self.logreg = linear_model.LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', max_iter=1000)

    def preprocess(self, train_dir, classid, test_dir):
        # ================= train_x, train_y ===================
        print('========== PREPROCESSING DATA ==========')
        train_x = []
        for i, s in enumerate(train_dir):
            audio_path = os.path.join('./data', s)
            y, sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            """
            오디오를 1차원 NumPy 부동 소수점 배열로 표현되는 시계열 y로 로드 및 디코딩한다. 변수 sr에는 y의 샘플링속도,
            즉 오디오의 초당 샘플 수가 포함된다. 기본적으로 모든 오디오는 모노로 혼합되어 로드 할 때 22050Hzㅀ 재샘플링된다.

            arguments 설명
                sr=22050 :	input 샘플링 주파수입니다. 아마도 갖고있는 오디오 파일의 샘플링 주파수는 22050이 아닐 확률이 큽니다. 
                            이렇게 값을 설정해주는 것은 11025 Hz 까지의 값만 써도 된다는 가정을 한 것이죠. 잘 모르시면 그냥 두세요.
                mono=True : 스테레오 음원일경우 모노로 바꿔준다는 말입니다. 역시 그냥 두시면 됩니다. 대부분의 경우 모노면 충분합니다.
                            이 글의 타겟이시면 스테레오 음원이 필요한 경우가 아닐거에요.
                offset, duration :	오디오 파일의 특정 구간만 쓰실경우 설정하시면 됩니다. 
                                    그러나, 초심자라면 이걸 쓰지 마시구요, 갖고있는 오디오 파일에서 의미있는 구간만 미리 잘라놓으세요.
                                    예를들어 음원이 60초인데 아기 우는 소리가 20~35초에 있다면 그 부분만 남기고 나머지는 버려서 15초로 만들어놓고 쓰시면 됩니다.
            """
            mfcc_mean = np.mean(mfcc, axis=1).squeeze()
            train_x.append(maxabs_scale(mfcc_mean))
            if i % 10 == 0:
                print('#', i+1, 'th data is processed')
        train_x = np.array(train_x, dtype=np.float32)

        pickle.dump(train_x, open('train_X.pkl', 'wb'))
        pickle.dump(classid, open('train_Y.pkl', 'wb'))

        # ===================== test_x =======================
        test_x = []
        for i, s in enumerate(test_dir):
            audio_path = os.path.join('./data', s)
            y, sr = librosa.load(audio_path)
            mfcc = librosa.feature.mfcc(y=y, sr=sr)
            mfcc_mean = np.mean(mfcc, axis=1).squeeze()
            test_x.append(maxabs_scale(mfcc_mean))
            if i % 10 == 0:
                print('#', i + 1, 'th data is processed')
        test_x = np.array(test_x, dtype=np.float32)

        pickle.dump(test_x, open('test_X.pkl', 'wb'))

        return train_x, classid, test_x

    def train(self, x, y):
        print('========== TRAINING START ==========')
        self.logreg.fit(x, y)

        joblib.dump(self.logreg, 'saved_model.pkl')
        return

    def load_model(self):
        self.logreg = joblib.load('saved_model.pkl')

        return

    def eval(self, x):
        print('========== EVALUATION START ==========')
        pred = self.logreg.predict(x)

        return pred
