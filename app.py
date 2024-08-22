from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import re
import easyocr

app = Flask(__name__)

# Load the saved TF-IDF matrix and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

tfidf_matrix = np.load('tfidf_matrix.npy')

# Your dataset
df = pd.read_csv("C:\\Users\\neeli\\ChatBotDataSet.csv")

# Initialize EasyOCR reader (use GPU if available)
reader = easyocr.Reader(['en'], gpu=True)  # Set gpu=False if you don't have a GPU

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def keyword_filter(text):
    keywords = ['court', 'judge', 'litigation', 'commercial law', 'appeal', 'tribunal']
    return any(keyword in text for keyword in keywords)

def analyze_text(text):
    if not keyword_filter(text.lower()):
        return "This text is not related to commercial courts."
    
    query = preprocess_text(text)
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, tfidf_matrix)
    best_similarity_score = similarity.max()

    if best_similarity_score < 0.1:
        return "This text is not related to commercial courts."
    
    best_match_index = similarity.argmax()
    return df['Response'].iloc[best_match_index]


def extract_text_from_image(image_data):
    try:
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        results = reader.readtext(image_np)
        text = ' '.join([res[1] for res in results])
        print(f"Extracted text from image: {text}")  # Debug print
        return text.strip()
    except Exception as e:
        print(f"Error in OCR processing: {e}")  # Debug print
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query', '')
    if not user_query:
        return jsonify({"response": "No query provided."})
    
    response = analyze_text(user_query)
    return jsonify({"response": response})

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    print(f"Received file: {file.filename}, type: {file.content_type}")  # Debug print

    try:
        if file.content_type.startswith('image/'):
            image_data = file.read()
            extracted_text = extract_text_from_image(image_data)

            if extracted_text:
                analysis_result = analyze_text(extracted_text)
                return jsonify({"response": f"Extracted text: {extracted_text}\n\nAnalysis: {analysis_result}"})
            else:
                return jsonify({"response": "No text found in the image."})

        elif file.content_type == 'text/plain':
            text_content = file.read().decode('utf-8')
            analysis_result = analyze_text(text_content)
            return jsonify({"response": analysis_result})
        
        else:
            return jsonify({"response": "Unsupported file type. Please upload an image or a text file."})

    except Exception as e:
        print(f"Error processing file: {e}")  # More detailed error logging
        return jsonify({"error": f"Failed to process the file: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
