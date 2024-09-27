# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from preprocess import preprocess_text
from mongodb.save_results import save_prediction

model = load("Results/text_classifier.joblib")
vectorizer = load("Results/tfidf_vectorizer.joblib")
app = FastAPI()

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
    text_vector = vectorizer.transform([clean_text])
 
    # Predict the category using the trained model
    prediction = model.predict(text_vector)

    # Get the human-readable label
    category = ['rec.sport.hockey', 'sci.space'][prediction[0]]

    # Save the prediction to MongoDB
    save_prediction(input.text, category)
 
    return {"prediction": category}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
