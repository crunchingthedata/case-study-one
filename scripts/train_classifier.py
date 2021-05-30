import os
from pathlib import Path

from bank_deposit_classifier.prep_data import *

INPUT_DATA_DIR = os.path.join(
    Path(__file__).parents[1],
    'data/input'
    )


# TODO: preserve base outcome name after one hot encoding
outcome = 'y_yes'
data_path = os.path.join(INPUT_DATA_DIR, 'bank-additional-full.csv')
features = None
categorical_features = None

dp = DataPrep(
    outcome=outcome,
    features=features,
    categorical_features=categorical_features
    )
data = dp.get_data(data_path)
data, encoder = dp.encode_data(data)
train, test = dp.split_data(data)
dp.save_data(train, 'train.csv')
dp.save_data(test, 'test.csv')
