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
COLUMN_DEFAULTS = {
    'job_blue_collar': [0],
    'job_entrepreneur': [0],
    'job_housemaid': [0],
    'job_management': [0],
    'job_retired': [0],
    'job_self_employed': [0],
    'job_services': [0],
    'job_student': [0],
    'job_technician': [0],
    'job_unemployed': [0],
    'job_unknown': [0],
    'marital_married': [0],
    'marital_single': [0],
    'marital_unknown': [0],
    'education_basic_6y': [0],
    'education_basic_9y': [0],
    'education_high_school': [0],
    'education_illiterate': [0],
    'education_professional_course': [0],
    'education_university_degree': [0],
    'education_unknown': [0],
    'default_unknown': [0],
    'default_yes': [0],
    'housing_unknown': [0],
    'housing_yes': [0],
    'loan_unknown': [0],
    'loan_yes': [0],
    'contact_telephone': [0],
    'month_aug': [0],
    'month_dec': [0],
    'month_jul': [0],
    'month_jun': [0],
    'month_mar': [0],
    'month_may': [0],
    'month_nov': [0],
    'month_oct': [0],
    'month_sep': [0],
    'day_of_week_mon': [0],
    'day_of_week_thu': [0],
    'day_of_week_tue': [0],
    'day_of_week_wed': [0],
    'poutcome_nonexistent': [0],
    'poutcome_success': [0],
    'age': [40],
    'duration': [258],
    'campaign': [2.6],
    'pdays': [962],
    'previous': [0.17],
    'emp_var_rate': [0.09],
    'cons_price_idx': [93.6],
    'cons_conf_idx': [-40.5],
    'euribor3m': [3.6],
    'nr_employed': [5168]
    }
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
    prediction = model.predict(data)[0]
    return prediction
