import os
from pathlib import Path
import re
import yaml

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing

from bank_deposit_classifier.sample import upsample_minority_class

BASE_DIR = Path(__file__).parents[2]
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_DIR = os.path.join(BASE_DIR, 'config')

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
            categorical = data[self._categorical_features]
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
        data = self.fix_column_names(data)
        return data, one_hot_encoder

    def split_data(self, data):
        sampled_data = upsample_minority_class(data, self._outcome, 0.5)
        train, test = train_test_split(data, random_state=123, train_size=0.8)
        return train, test

    def save_data(self, data, path):
        path = os.path.join(DATA_DIR, path)
        data.to_csv(path, index=False)
        print(f'Data written to {path}')

    def fix_column_names(self, data):
        data.columns = [self.standardize_names(x) for x in data.columns]
        pattern = self._outcome + '_'
        matches = [x for x in data.columns if re.match(pattern, x)]
        if len(matches) != 1:
            raise Exception('Cannot uniquely identify outcome column!')
        data = data.rename(columns={matches[0]: self._outcome})
        return data

    @staticmethod
    def standardize_names(name):
        return re.sub('\W', '_', name).lower()

    @classmethod
    def from_yaml(cls, path):
        path = os.path.join(CONFIG_DIR, path)
        with open(path, 'r') as file:
            params = yaml.load(file, Loader=yaml.FullLoader)
        return cls(**params)
