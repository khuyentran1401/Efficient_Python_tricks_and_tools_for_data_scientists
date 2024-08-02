import mlflow
import numpy as np

model_uri = "runs:/1e20d72afccf450faa3b8a9806a97e83/model"
sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_uri)

data = np.array([-4, 1, 0, 10, -2, 1]).reshape(-1, 1)

predictions = sklearn_pyfunc.predict(data)
