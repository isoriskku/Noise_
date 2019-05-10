import numpy as np
import pickle
from utils import _initialize


# =================================================================
# 1. Choose DATA : UrbanSound8K
# 2. Choose Test fold : 1 ~ 10

DATA_NAME = 'UrbanSound8K'
TEST_FOLD = 10
TRAINING = True
PREPROCESS = False
# =================================================================
assert DATA_NAME in ['UrbanSound8K']
assert TEST_FOLD in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load dataset
train_meta_data, test_meta_data, logistic_regression, accuracy = _initialize(DATA_NAME, TEST_FOLD)
train_dir, train_classid = train_meta_data
test_dir, test_classid = test_meta_data

num_data = train_dir.shape
num_label = int(train_classid.max()) + 1
print('# of Training data : %d \n' % num_data)

# Make model
model = logistic_regression(20, num_label)

if PREPROCESS:
    # preprocessing
    train_x, train_y, test_x = model.preprocess(train_dir, train_classid, test_dir)
else:
    train_x = pickle.load(open('train_X.pkl', 'rb'))
    train_y = pickle.load(open('train_Y.pkl', 'rb'))
    test_x = pickle.load(open('test_X.pkl', 'rb'))

if TRAINING:
    # TRAIN
    model.train(train_x, train_y)
    print('Training finished\n')
else:
    # Load the model
    model.load_model()

# EVALUATION
pred = model.eval(test_x)
print('pred : ', pred, '\ntest_y : ', test_classid)

acc = accuracy(pred, test_classid)
print(' Accuracy on Test Data : %.2f' % acc)
