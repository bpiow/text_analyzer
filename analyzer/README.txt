Step 1: Train the model:
	•	First, run model.py to train the Logistic Regression model and save it to disk.
	•	Execute: python model.py

Step 2: Run the FastAPI app:
	•	After the model is saved, run the FastAPI app with uvicorn.
	•	Execute: uvicorn main:app --reload

Step 3: Test the API:
	•	Once the FastAPI server is running, you can test the API by sending a POST request to http://127.0.0.1:8000/predict with a JSON payload containing a text field.
For example, using curl: 
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{"text": "I love playing hockey!"}':
