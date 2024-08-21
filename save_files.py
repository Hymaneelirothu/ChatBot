import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    return text

# Load the cleaned dataset
def load_data():
    df = pd.read_csv("C:\\Users\\neeli\\ChatBotDataSet.csv")
    return df

# Train the TF-IDF vectorizer and save it
def train_vectorizer(df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Prompt'])
    
    # Save vectorizer and TF-IDF matrix
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save the TF-IDF matrix as a dense array
    np.save('tfidf_matrix.npy', tfidf_matrix.toarray())
    
    return vectorizer, tfidf_matrix

# Function to get the best response based on user query
def get_response(user_query, df, vectorizer, tfidf_matrix):
    # Preprocess the user query
    query = preprocess_text(user_query)
    
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Compute the cosine similarity between the query and the TF-IDF matrix
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    
    # Find the index of the most similar document
    best_match_index = similarity.argmax()
    
    # Get the most relevant response
    best_response = df['Response'].iloc[best_match_index]
    
    return best_response

# Main function to load data, train model, and run the chatbot
def main():
    # Load the cleaned dataset
    df = load_data()
    
    # Train the vectorizer and save the model
    vectorizer, tfidf_matrix = train_vectorizer(df)
    
    # Test the chatbot with a sample query
    user_query = "What is the jurisdiction of Commercial Courts?"
    response = get_response(user_query, df, vectorizer, tfidf_matrix)
    print("Chatbot Response:", response)

if __name__ == "__main__":
    main()
