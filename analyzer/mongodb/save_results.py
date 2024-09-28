from mongodb.db import predictions_collection
from datetime import datetime

def save_prediction(text, prediction):
    data = {
        "text": text,
        "prediction": prediction,
        "timestamp": datetime.now()
    }
    predictions_collection.insert_one(data)
    print(f"Saved prediction: {data}")

def get_all_predictions():
    return list(predictions_collection.find({}, {"_id": 0, "text": 1, "prediction": 1, "timestamp": 1}))
