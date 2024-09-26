from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump
from preprocess import preprocess_text
import os

# Load dataset (using only 2 categories for simplicity)
newsgroups = fetch_20newsgroups(subset='train', categories=['rec.sport.hockey', 'sci.space'])

# Preprocess text data
texts = [preprocess_text(text) for text in newsgroups.data]
labels = newsgroups.target

# Convert the text data into TF-IDF vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
y = labels

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model (optional step)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Ensure the 'Results' directory exists
os.makedirs('Results', exist_ok=True)

# Save the trained model and vectorizer
dump(model, 'Results/text_classifier.joblib')
dump(vectorizer, 'Results/tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully as 'text_classifier.joblib' and 'tfidf_vectorizer.joblib'.")
