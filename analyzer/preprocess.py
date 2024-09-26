import spacy

# Load a pre-trainer spacy model for text preprocessing
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """
    Function to preprocess the input text by tokenizing, 
    removing stop words, and lemmatizing.
    """
    # Tokenize and process the text us spacy
    doc = nlp(text)

    # Filter tokens: Remove stopwords and puctiation, then lemmatize
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    return ' '.join(clean_tokens)

# Function test
example_text = "This is an example sentence to test our preprocessing function."
clean_text = preprocess_text(example_text)
print(f"Cleaned text: {clean_text}")
