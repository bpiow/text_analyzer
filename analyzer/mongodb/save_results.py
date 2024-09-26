# database/save_results.py
from mongodb.db import predictions_collection
from datetime import datetime

def save_prediction(text, prediction):
    """
    Saves the prediction result into the MongoDB collection.
    :param text: The input text that was classified.
    :param prediction: The prediction result from the model.
    """

    data = {
        "text": text,
        "prediction": prediction,
        "timestamp": datetime.now()
    }

    # Inserting the data into the MongoDB collection
    try:
        predictions_collection.insert_one(data)
        print(f"Prediction saved: {data}")
    except Exception as error:
        print(f"Error saving prediction to MongoDB: {error}")
