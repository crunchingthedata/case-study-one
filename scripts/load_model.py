import os

import mlflow
import pandas as pd

from bank_deposit_classifier.prep_data import DATA_DIR

# load a model via the model registry
mlflow.set_tracking_uri('http://localhost:5000')
path = "models:/bank-deposit-classifier-test/1"
model = mlflow.sklearn.load_model(path)
print(model)

# load a model via the model registry
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('case-study-one-test')
path = "runs:/427e70ff5e8c48aeb741f98e7dba42b4/model"
model = mlflow.sklearn.load_model(path)
print(model)

# make predictions 
test_path = 'intermediate/test.csv'
test_path_full = os.path.join(DATA_DIR, test_path)
test = pd.read_csv(test_path_full) \
    .drop(['y'], axis=1) \
    .head(5)
predictions = model.predict(test)
print(predictions)
