import pandas as pd
import pytest

import bank_deposit_classifier.sample as sample

@pytest.fixture
def data():
    data = pd.DataFrame({
        'y': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'x1': [1, 2, 1, 2, 1, 2, 1, 5, 6, 5],
        })
    return data


def test_upsample_minority_class(data):
    data_ = sample.upsample_minority_class(data, 'y', 0.5)
    assert isinstance(data_, pd.DataFrame)
