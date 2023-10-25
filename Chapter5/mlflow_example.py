import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor


import itertools

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Define a list of hyperparameters to try out
n_estimators_list = [50, 100, 150]
max_depth_list = [4, 6, 8]
max_features_list = [2, 3, 4]

for combination in itertools.product(
    n_estimators_list, max_depth_list, max_features_list
):
    # Create and train models.
    rf = RandomForestRegressor(
        n_estimators=combination[0],
        max_depth=combination[1],
        max_features=combination[2],
    )
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
