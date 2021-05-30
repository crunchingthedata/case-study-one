import os
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from bank_deposit_classifier.sample import upsample_minority_class


DATA_DIR = os.path.join(Path(__file__).parents[2], 'data')

class DataPrep:
    def __init__(self, outcome, features=None, categorical_features=None, one_hot_encoder=None):
        self._outcome = outcome
        self._features = features
        self._categorical_features = categorical_features
        self._one_hot_encoder = one_hot_encoder

    def get_data(self, path):
        path = os.path.join(DATA_DIR, path)
        data = pd.read_csv(path, sep=';')
        if self._features:
            data = data[self._features]
        print(f'Data read from {path}')
        return data

    def encode_data(self, data):
        if self._categorical_features:
            categorial = data[self._categorical_features]
        else:
            categorical = data.select_dtypes(exclude=np.number)
            categorical_features = categorical.columns
        if self._one_hot_encoder:
            one_hot_encoder = self._one_hot_encoder
        else:
            one_hot_encoder = preprocessing.OneHotEncoder(
                sparse=False,
                drop='first'
                )
        categorical = one_hot_encoder.fit_transform(categorical)
        categorical_columns = one_hot_encoder.get_feature_names(categorical_features)
        categorical = pd.DataFrame(categorical, columns=categorical_columns)
        continuous = data.select_dtypes(include=np.number)
        data = pd.concat([categorical, continuous], axis=1)
        return data, one_hot_encoder

    def split_data(self, data):
        sampled_data = upsample_minority_class(data, self._outcome, 0.5)
        train, test = train_test_split(data, random_state=123, train_size=0.8)
        return train, test

    def save_data(self, data, path):
        path = os.path.join(DATA_DIR, path)
        data.to_csv(path, index=False)
        print(f'Data written to {path}')
