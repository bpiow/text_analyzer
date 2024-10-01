import spacy

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    # Tokenize and process the text using spacy
    doc = nlp(text)
    # Filter tokens: Remove stopwords and puctiation, then lemmatize
    clean_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(clean_tokens)
