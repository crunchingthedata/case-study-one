from bank_deposit_classifier.prep_data import *


input_path = 'input/bank-additional-full.csv'
train_path = 'intermediate/train.csv'
test_path = 'intermediate/test.csv'
config_path= 'data_prep.yaml'

dp = DataPrep.from_yaml(config_path)
data = dp.get_data(input_path)
data, encoder = dp.encode_data(data)
train, test = dp.split_data(data)
dp.save_data(train, train_path)
dp.save_data(test, test_path)
