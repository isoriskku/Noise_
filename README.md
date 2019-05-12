# Noise_
Noise analysis

### 사용법
main.py의 다음 부분에서 전처리를 할지, 학습을 할지 정한다.
```
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
```

#### PREPROCESS
10개의 데이터 그룹중 10번째 fold를 test data로 정해 미리 전처리 해둔 파일이 CSV로 저장되어있습니다.  
그대로 사용할거면 PREPROCESS = False, 교차검증을 위해 다른 fold의 data를 test data로 다시 학습하고 싶으면 PREPROCESS = True.

#### TRAINING
미리 전처리 해둔 파일로 학습한 모델을 사용할거면 TRAINING = False.  
새로운 data로 다시 학습할거면 True.

#### 주의
전처리나 학습을 새로 하게 될 경우 model.json, model.h5에 저장되어있는 기학습된 모델 백업데이터에 새로운 모델을 덮어씌우므로
보존하고 싶을시 따로 저장해주세요.
