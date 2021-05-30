import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from bank_deposit_classifier.sample import upsample_minority_class


INTERMEDIATE_DATA_DIR = os.path.join(
    Path(__file__).parents[2],
    'data/intermediate'
    )

def get_data(data_path, features = None):
    data = pd.read_csv(data_path, sep=';')
    if features:
        data = data[features]
    print(f'Data read from {data_path}')
    return data

def encode_data(data, one_hot_encoder = None, categorical_features = None):
    if categorical_features:
        categorial = data[categorical_features]
    else:
        categorical = data.select_dtypes(exclude=np.number)
        categorical_features = categorical.columns
    if not one_hot_encoder:
        one_hot_encoder = preprocessing.OneHotEncoder(
            sparse=False,
            drop='first'
            )
    categorical = one_hot_encoder.fit_transform(categorical)
    categorical = pd.DataFrame(categorical, columns=one_hot_encoder.get_feature_names(categorical_features))
    continuous = data.select_dtypes(include=np.number)
    data = pd.concat([categorical, continuous], axis=1)
    return data, one_hot_encoder

def split_data(data, outcome):
    sampled_data = upsample_minority_class(data, outcome, 0.5)
    train, test = train_test_split(data, random_state=123, train_size=0.8)
    return train, test

def save_data(data, file_name = 'data.csv'):
    path = os.path.join(INTERMEDIATE_DATA_DIR, file_name)
    data.to_csv(path, index=False)
    print(f'Data written to {path}')
