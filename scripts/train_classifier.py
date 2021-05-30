from bank_deposit_classifier.prep_data import *


# TODO: preserve base outcome name after one hot encoding
outcome = 'y_yes'
input_path = 'input/bank-additional-full.csv'
train_path = 'intermediate/train.csv'
test_path = 'intermediate/test.csv'
features = None
categorical_features = None

dp = DataPrep(
    outcome=outcome,
    features=features,
    categorical_features=categorical_features
    )
data = dp.get_data(input_path)
data, encoder = dp.encode_data(data)
train, test = dp.split_data(data)
dp.save_data(train, train_path)
dp.save_data(test, test_path)
