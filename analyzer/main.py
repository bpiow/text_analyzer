# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from preprocess import preprocess_text

# Load the saved model and vectorizer
model = load("Results/text_classifier.joblib")
vectorizer = load("Results/tfidf_vectorizer.joblib")

# Initialize FastAPI app
app = FastAPI()

# Define the root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Text Classifier API!"}

# Define the input schema
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_text(input: TextInput):
    """
    API endpoint to classify input text.
    """
    # Preprocess the input text
    clean_text = preprocess_text(input.text)

    # Convert the cleaned text to a vector
    text_vector = vectorizer.transform([clean_text])  # Ensure vectorizer is accessible here
 
    # Predict the category using the trained model
    prediction = model.predict(text_vector)

    # Get the human-readable label
    category = ['rec.sport.hockey', 'sci.space'][prediction[0]]
 
    return {"prediction": category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
