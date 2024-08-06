from fastapi import FastAPI
import joblib
import pandas as pd 

# Create a FastAPI application instance
app = FastAPI()

# Load the pre-trained machine learning model
model = joblib.load("lr.joblib")

# Define a POST endpoint for making predictions
@app.post("/predict/")
def predict(data: list[float]):
    # Define the column names for the input features
    columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]
    
    # Create a pandas DataFrame from the input data
    features = pd.DataFrame([data], columns=columns)
    
    # Use the model to make a prediction
    prediction = model.predict(features)[0]
    
    # Return the prediction as a JSON object, rounding to 2 decimal places
    return {"price": round(prediction, 2)}
