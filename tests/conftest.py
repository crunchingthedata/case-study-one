import pandas as pd
import pytest


@pytest.fixture
def data():
    data = pd.DataFrame({
        'y': [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        'x1': [1, 2, 1, 2, 1, 2, 1, 5, 6, 5],
        })
    return data
