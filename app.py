from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import gdown
import os

app = Flask(__name__)
CORS(app)

# URL chia sẻ của các tệp trên Google Drive
url_vectorizer = 'https://drive.google.com/uc?id=19v7W6fafUWyUJYBjdyqL72jMxBkK5L-4'
url_matrix = 'https://drive.google.com/uc?id=1-WNsrUwKUJlt_RsLnBGlKkmRoMzTGv50'

# Đường dẫn lưu tệp trong Persistent Storage
persistent_dir = '/persistent'
vectorizer_path = os.path.join(persistent_dir, 'tfidf_vectorizer.pkl')
matrix_path = os.path.join(persistent_dir, 'tfidf_matrix.pkl')

# Tải các tệp từ Google Drive nếu chưa tồn tại
def download_files():
    if not os.path.exists(persistent_dir):
        os.makedirs(persistent_dir)
    if not os.path.exists(vectorizer_path):
        gdown.download(url_vectorizer, vectorizer_path, quiet=False)
    if not os.path.exists(matrix_path):
        gdown.download(url_matrix, matrix_path, quiet=False)

# Tải TF-IDF vectorizer và ma trận TF-IDF
def load_tfidf():
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    with open(matrix_path, 'rb') as f:
        reduced_matrix = pickle.load(f)
    return vectorizer, reduced_matrix

# Đọc câu từ tệp theo chỉ mục
def get_sentence_by_index(file_path, index):
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == index:
                return line.strip()
    return None

# Tìm kiếm các câu tương tự
def find_similar_sentences(input_sentence, vectorizer, reduced_matrix, top_n=5):
    input_vec = vectorizer.transform([input_sentence])
    cosine_similarities = cosine_similarity(input_vec, reduced_matrix).flatten()
    similar_indices = cosine_similarities.argsort()[-top_n:][::-1]
    return similar_indices

# Hàm tính toán độ tương đồng giữa hai câu
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.route('/search', methods=['POST'])
def search_sentence():
    input_sentence = request.json['text']
    similar_indices = find_similar_sentences(input_sentence, vectorizer, reduced_matrix)
    
    found = False
    for idx in similar_indices:
        sentence = get_sentence_by_index(output, idx)
        similarity = similar(input_sentence, sentence)
        if similarity >= 0.8:
            return jsonify({"input_sentence": input_sentence, "results": f'"{sentence}" (Similarity: {similarity})'})
            found = True

    if not found:
        return jsonify({"input_sentence": input_sentence, "results": "No similar sentence found."})

if __name__ == '__main__':
    # Tải tệp từ Google Drive nếu cần
    download_files()
    
    # Tải TF-IDF index từ các tệp .pkl
    vectorizer, reduced_matrix = load_tfidf()
    
    # Chạy server Flask
    port = int(os.environ.get("PORT", 5001))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
