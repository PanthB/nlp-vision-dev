import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from collections import Counter

# Module-level cache for model and data
_model = None
_data = None
_database_data = None
_database_embeddings = None

def _load_model_and_data():
    global _model, _data, _database_data, _database_embeddings
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if _data is None:
        with open("final_sentences.json", 'r', encoding='utf-8') as f:
            _data = json.load(f)
        _database_data = _data
        _database_embeddings = np.array([item['embedding'] for item in _database_data])

def process_user_input(user_input: str) -> str:
    """Given a user input string, return the label of the most similar sentence from the database, or a fallback if no good match."""
    _load_model_and_data()
    input_embedding = _model.encode([user_input])
    similarities = cosine_similarity(input_embedding, _database_embeddings)[0]
    best_index = np.argmax(similarities)
    best_score = similarities[best_index]
    best_match = _database_data[best_index]
    threshold = 0.7  # You can adjust this value as needed
    if best_score < threshold:
        return "we couldn't map that to anything"
    return best_match['label']

def process_user_input_knn(user_input: str, k: int = 3) -> str:
    """Given a user input string, return the most common label among k nearest neighbors using cosine similarity, or a fallback if confidence is low."""
    _load_model_and_data()
    database_data = _database_data
    database_embeddings = _database_embeddings
    if len(database_embeddings) == 0:
        return "we couldn't map that to anything"
    n_neighbors_actual = min(k, len(database_embeddings))
    knn = NearestNeighbors(n_neighbors=n_neighbors_actual, metric='cosine')
    knn.fit(database_embeddings)
    input_embedding = _model.encode([user_input])
    distances, indices = knn.kneighbors(input_embedding)
    neighbor_labels = [database_data[idx]['label'] for idx in indices[0]]
    label_counts = Counter(neighbor_labels)
    predicted_label, count = label_counts.most_common(1)[0]
    confidence = count / n_neighbors_actual
    confidence_threshold = 0.6  # You can adjust this as needed
    if confidence < confidence_threshold:
        return "we couldn't map that to anything"
    return predicted_label