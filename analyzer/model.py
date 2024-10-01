import joblib
import os
from preprocess import preprocess_text 

MODEL_PATH = os.path.join("results", "text_classifier.joblib")
VECTORIZER_PATH = os.path.join("results", "tfidf_vectorizer.joblib")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

CATEGORIES = ["rec.sport.hockey", "sci.space"]

def make_prediction(text):
    """
    Preprocess the input text, transform it using the vectorizer,
    and make a prediction using the loaded model.

    :param text: Input text string to classify.
    :return: The predicted category label as a string.
    """
    # Preprocess the input text
    clean_text = preprocess_text(text)
    
    # Transform the preprocessed text using the vectorizer
    text_vector = vectorizer.transform([clean_text])
    
    # Make a prediction using the loaded model
    prediction_index = model.predict(text_vector)
    
    # Map the prediction index to the corresponding category label
    prediction_label = CATEGORIES[prediction_index[0]]

    return prediction_label
