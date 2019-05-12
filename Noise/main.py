import numpy as np
from utils import _initialize


# =================================================================
# 1. Choose DATA : UrbanSound8K
# 2. Choose Classifier model : CNN
# 3. Choose Test fold : 1 ~ 10

# DATA
DATA_NAME = 'UrbanSound8K'

# CLASSIFIER
CLASSIFIER_NAME = 'CNN'

TEST_FOLD = 10
TRAINING = False
PREPROCESS = False
# =================================================================
assert DATA_NAME in ['UrbanSound8K']
assert CLASSIFIER_NAME in ['CNN']
assert TEST_FOLD in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load dataset
train_meta_data, test_meta_data, classifier = _initialize(DATA_NAME, CLASSIFIER_NAME, TEST_FOLD)
train_dir, train_classid = train_meta_data
test_dir, test_classid = test_meta_data

num_data = train_dir.shape
num_label = int(train_classid.max()) + 1
print('# of Training data : %d \n' % num_data)

# Make model
model = classifier()

# preprocessing
train_x, train_y, test_x, test_y = model.preprocess(train_dir, train_classid, test_dir, test_classid, PREPROCESS)

if TRAINING:
    # TRAIN
    model.train(train_x, train_y, test_x, test_y)
    print('Training finished\n')
else:
    # Load the model
    model.load_model()

# EVALUATION
model.eval(train_x, train_y, test_x, test_y)
