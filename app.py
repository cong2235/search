from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import gdown

app = Flask(__name__)

# URL chia sẻ của tệp trên Google Drive
url = 'https://drive.google.com/uc?id=1mMznGAmltidr2RHg-lZiIk8UWkJVKVwu'
output = 't1.txt'
gdown.download(url, output, quiet=False)

# Tải TF-IDF vectorizer và ma trận TF-IDF đã giảm chiều
def load_tfidf(vectorizer_path='tfidf_vectorizer.pkl', matrix_path='tfidf_matrix.pkl'):
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
    # Tải TF-IDF index từ các tệp .pkl
    vectorizer, reduced_matrix = load_tfidf()
    app.run(debug=True)
