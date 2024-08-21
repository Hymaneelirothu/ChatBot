from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved TF-IDF matrix and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

tfidf_matrix = np.load('tfidf_matrix.npy')

# Your dataset (Ensure 'df' is the same DataFrame used when creating the TF-IDF matrix)
df = pd.read_csv("C:\\Users\\neeli\\ChatBotDataSet.csv")

# Preprocessing function (adjust as needed)
import re

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove unwanted special characters (e.g., underscores or excess punctuation)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    return text


# Function to get the best response based on user query
def get_response(user_query):
    # Preprocess the user query
    query = preprocess_text(user_query)
    
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    
    # Check similarity score
    best_match_index = similarity.argmax()
    best_similarity_score = similarity.max()
    
    # If the similarity score is too low, return a default response
    if best_similarity_score < 0.02:  # Adjust the threshold as needed
        return "I'm not sure about that. Could you ask something else?"
    
    # Return the best match
    return df['Response'].iloc[best_match_index]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query', '')

    # Get a response based on the query
    response = get_response(user_query)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
