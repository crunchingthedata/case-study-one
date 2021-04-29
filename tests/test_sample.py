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

def test_upsample_minority_class_high_p(data):
    with pytest.raises(ValueError) as e:
        sample.upsample_minority_class(data, 'y', 1.5)
    assert "Proportion out of bounds" in str(e.value)

def test_upsample_minority_class_binary(data):
    with pytest.raises(ValueError) as e:
        data_ = data.loc[data['y'] == 1]
        sample.upsample_minority_class(data_, 'y', 0.5)
    assert "Binary outcome expected" in str(e.value)
