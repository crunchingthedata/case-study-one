import mlflow

# load a model via the model registry
mlflow.set_tracking_uri('http://localhost:5000')
path = "models:/bank-deposit-classifier-test/1"
model = mlflow.sklearn.load_model(path)
print(model)

# load a model via the model registry
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('case-study-one-test')
path = "runs:/427e70ff5e8c48aeb741f98e7dba42b4/model"
model = mlflow.sklearn.load_model(path) 
print(model)
