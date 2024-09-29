from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from typing import List
from datetime import timedelta
from model import make_prediction 
from mongodb.save_results import save_prediction, get_all_predictions 
from auth import create_access_token, authenticate_user, get_current_user  # Authentication functions

app = FastAPI()

# Example model metadata (can be dynamically updated)
MODEL_METADATA = {
    "model_name": "Text Classifier",
    "version": "1.0.0",
    "description": "A simple text classifier for predicting categories.",
    "last_trained": "2024-09-28",
}

# Define the request models
class TextInput(BaseModel):
    text: str

class BatchInput(BaseModel):
    texts: List[str]

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/token")
def login_for_access_token(form_data: UserLogin):
    """
    Endpoint to authenticate a user and generate a JWT token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", dependencies=[Depends(get_current_user)])
def predict_text(input: TextInput):
    """
    Endpoint to make a single prediction.
    Requires user authentication.
    """
    prediction = make_prediction(input.text)
    save_prediction(input.text, prediction)  # Save the prediction to MongoDB
    return {"prediction": prediction}

@app.post("/batch_predict", dependencies=[Depends(get_current_user)])
def batch_predict_text(batch_input: BatchInput):
    """
    Endpoint to process multiple texts and return predictions for each.
    Requires user authentication.
    """
    results = [{"text": text, "prediction": make_prediction(text)} for text in batch_input.texts]
    for result in results:
        save_prediction(result["text"], result["prediction"])  # Save each prediction to MongoDB
    return results

@app.get("/predictions", dependencies=[Depends(get_current_user)])
def get_predictions():
    """
    Endpoint to retrieve all past predictions from the database.
    Requires user authentication.
    """
    predictions = get_all_predictions()
    if not predictions:
        raise HTTPException(status_code=404, detail="No predictions found.")
    return predictions

@app.get("/model_metadata")
def model_metadata():
    """
    Endpoint to retrieve metadata about the model.
    """
    return MODEL_METADATA
