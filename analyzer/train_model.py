import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load the dataset
categories = ['rec.sport.hockey', 'sci.space']
newsgroups = fetch_20newsgroups(subset='all', categories=categories)

# Extract data and labels
texts = newsgroups.data
labels = newsgroups.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Define a pipeline that includes both the TF-IDF vectorizer and Logistic Regression model
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(stop_words='english', max_df=0.7)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained vectorizer and model
# Ensure 'results' directory exists or adjust the path as needed
joblib.dump(pipeline.named_steps['vectorizer'], 'results/tfidf_vectorizer.joblib')
joblib.dump(pipeline.named_steps['classifier'], 'results/text_classifier.joblib')

print("Model and vectorizer saved successfully.")
