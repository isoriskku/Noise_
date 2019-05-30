import numpy as np
from utils import _initialize



# =================================================================
# 1. Choose DATA : UrbanSound8K
# 2. Choose Classifier model : UrbanSound8K, CNN
# 3. Choose Test fold : 1 ~ 10

# DATA
DATA_NAME = 'UrbanSound8K'

# CLASSIFIER
CLASSIFIER_NAME = 'CNN'

VALIDATION_FOLD = [8, 9]
TEST_FOLD = 10
TRAINING = False
PREPROCESS = False
# =================================================================
# Return the label from 1 to 10
assert DATA_NAME in ['UrbanSound8K']
assert CLASSIFIER_NAME in ['CNN']
assert TEST_FOLD in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Load dataset
training_set, validation_set, test_set, classifier, accuracy = _initialize(DATA_NAME, CLASSIFIER_NAME,
                                                                 VALIDATION_FOLD, TEST_FOLD, PREPROCESS)

train_x, train_class = training_set
validation_x, validation_class = validation_set
test_x, test_class = test_set

num_data = train_x.shape[0]
num_label = int(train_class.max()) + 1
print('# of Training data : %d \n' % num_data)

# Make model
model = classifier()

if TRAINING:
    # TRAIN
    model.train(train_x, train_class, validation_x, validation_class)
    print('Training finished\n')
else:
    # Load the model
    model.load_model()

"""
# EVALUATION
model.eval(train_x, train_class, validation_x, validation_class)
"""

# Determine weather it is dangerous sound or not
y_pred = model.predict(test_x)
print('y_pred.shape :', y_pred.shape, 'test_class.shape : ', test_class.shape)
acc = accuracy(y_pred, test_class)
print('Class accuracy on Test Data : %.2f' % acc)

for idx, y in enumerate(test_class):
    if y in [0, 2, 5, 9]:
        test_class[idx] = 0
    else:
        test_class[idx] = 1
for idx, y in enumerate(y_pred):
    if y in [0, 2, 5, 9]:
        y_pred[idx] = 0
    else:
        y_pred[idx] = 1
acc = accuracy(y_pred, test_class)
print('Danger accuracy on Test Data : %.2f' % acc)


