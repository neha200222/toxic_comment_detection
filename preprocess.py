import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df['comment_text']
    y = df[df.columns[2:]].values
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X = vectorizer.fit_transform(X)
    return X, y, vectorizer

def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
