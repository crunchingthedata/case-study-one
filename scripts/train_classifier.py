import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from bank_deposit_classifier.sample import upsample_minority_class
from bank_deposit_classifier.prep_data import DATA_DIR


outcome = 'y'
n_estimators = 100
max_features = 6

test_path = os.path.join(DATA_DIR, 'intermediate/test.csv')
train_path = os.path.join(DATA_DIR, 'intermediate/train.csv')
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)
train_resampled = upsample_minority_class(train, outcome, 0.5)

rf = RandomForestClassifier(
    n_estimators = n_estimators,
    max_features = max_features,
    random_state = 123
    )
rf.fit(train_resampled.drop(outcome, axis=1), train_resampled[outcome])

train_predictions = rf.predict(train_resampled.drop(outcome, axis=1))
test_predictions = rf.predict(test.drop(outcome, axis=1))
train_auc = roc_auc_score(train_resampled[outcome], train_predictions)
test_auc = roc_auc_score(test[outcome], test_predictions)
