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

data = get_data(data_path, features = features)
data, encoder = encode_data(data, categorical_features)
train, test = split_data(data, outcome)
save_data(train, 'train.csv')
save_data(test, 'test.csv')
