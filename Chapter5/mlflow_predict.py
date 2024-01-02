import mlflow.pyfunc
import joblib

X_test = joblib.load("data/X_test.pkl")
y_test = joblib.load("data/y_test.pkl")

model_name = "sk-learn-random-forest-reg-model"
alias = "champion"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}@{alias}")

model.predict(X_test)
