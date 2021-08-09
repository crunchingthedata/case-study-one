import os
import yaml

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from bank_deposit_classifier.sample import upsample_minority_class
from bank_deposit_classifier.prep_data import DATA_DIR, CONFIG_DIR


config_path = 'train_model.yaml'
test_path = 'intermediate/test.csv'
train_path = 'intermediate/train.csv'

# get parameters from config
config_path_full = os.path.join(CONFIG_DIR, config_path)
with open(config_path_full, 'r') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
outcome = config.get('outcome')
n_estimators = config.get('n_estimators')
max_features = config.get('max_features')

# get test and train data
test_path_full = os.path.join(DATA_DIR, test_path)
train_path_full = os.path.join(DATA_DIR, train_path)
test = pd.read_csv(test_path_full)
train = pd.read_csv(train_path_full)
train_resampled = upsample_minority_class(train, outcome, 0.5)

# train model
rf = RandomForestClassifier(
    n_estimators = n_estimators,
    max_features = max_features,
    random_state = 123
    )
rf.fit(train_resampled.drop(outcome, axis=1), train_resampled[outcome])

# evaluate model
train_predictions = rf.predict(train_resampled.drop(outcome, axis=1))
test_predictions = rf.predict(test.drop(outcome, axis=1))
train_auc = roc_auc_score(train_resampled[outcome], train_predictions)
test_auc = roc_auc_score(test[outcome], test_predictions)

# log data
import mlflow
import tempfile

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('case-study-one')
mlflow.start_run()

mlflow.log_param('n_estimators', n_estimators)
mlflow.log_param('max_features', max_features)
mlflow.log_metric('train_auc', train_auc)
mlflow.log_metric('test_auc', test_auc)

with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, 'train.csv')
    train.to_csv(path)
    mlflow.log_artifacts(tmp)

mlflow.end_run()
