import numpy as np

def score_comment(model, vectorizer, comment):
    vectorized_comment = vectorizer.transform([comment])
    results = model.predict(vectorized_comment)
    return results
