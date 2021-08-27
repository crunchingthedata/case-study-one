import mlflow
import pandas as pd


COLUMN_ORDER = [
    'job_blue_collar', 'job_entrepreneur', 'job_housemaid',
    'job_management', 'job_retired', 'job_self_employed', 'job_services',
    'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
    'marital_married', 'marital_single', 'marital_unknown',
    'education_basic_6y', 'education_basic_9y', 'education_high_school',
    'education_illiterate', 'education_professional_course',
    'education_university_degree', 'education_unknown', 'default_unknown',
    'default_yes', 'housing_unknown', 'housing_yes', 'loan_unknown',
    'loan_yes', 'contact_telephone', 'month_aug', 'month_dec', 'month_jul',
    'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct',
    'month_sep', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue',
    'day_of_week_wed', 'poutcome_nonexistent', 'poutcome_success', 'age',
    'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'
    ]
NUMERIC_COLUMNS = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate',
    'cons_price_idx', 'cons_conf_idx', 'euribor3m', 'nr_employed'
    ]
CATEGORICAL_COLUMNS = [x for x in COLUMN_ORDER if x not in NUMERIC_COLUMNS]
COLUMN_DEFAULTS = {x: [0] for x in COLUMN_ORDER}
DEFAULT_DATA = pd.DataFrame(COLUMN_DEFAULTS)[COLUMN_ORDER]

def load_model(model='bank-deposit-classifier-test', version='1'):
    mlflow.set_tracking_uri('http://localhost:5000')
    path = f'models:/{model}/{version}'
    model = mlflow.sklearn.load_model(path)
    return model

def predict_with_defaults(model, categoricals=[], numerics={}):
    data = DEFAULT_DATA.copy()
    for c in categoricals:
        data[c] = 1
    for k, v in numerics.items():
        data[k] = v
    prediction = model.predict(data)[0]
    return prediction
