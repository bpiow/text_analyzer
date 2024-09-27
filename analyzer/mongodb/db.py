from pymongo import MongoClient

client = MongoClient('mongodb://localhost:27017/')

# Create a database named 'text_classification'
db = client.text_classification

# Create a collection (similar to a table in SQL) named 'predictions'
predictions_collection = db.predictions
