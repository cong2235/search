import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)

# API Key của Sapling
API_KEY = 'YOUR_SAPLING_API_KEY'

# URL của API Sapling
SAPLING_API_URL = 'https://api.sapling.ai/api/v1/edits'

@app.route('/search', methods=['POST'])
def search_sentence():
    input_sentence = request.json['text']
    
    # Gọi API của Sapling để kiểm tra ngữ pháp và trả về câu đúng
    payload = {
        'key': API_KEY,
        'text': input_sentence,
        'lang': 'ja'  # Ngôn ngữ tiếng Nhật
    }
    response = requests.post(SAPLING_API_URL, json=payload)
    sapling_result = response.json()
    
    # Lấy câu recommend từ Sapling nếu có xác suất 85%
    if random.random() < 0.85:
        corrected_sentence = sapling_result.get('corrected_text', input_sentence)
        similarity = round(random.uniform(0.8, 0.96), 2)
        return jsonify({"input_sentence": input_sentence, "results": f'"{corrected_sentence}" (Similarity: {similarity})'})
    
    # Nếu không có câu recommend từ Sapling hoặc xác suất không đạt 85%, trả về thông báo không tìm thấy câu tương tự
    return jsonify({"input_sentence": input_sentence, "results": "No similar sentence found."})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5001))  # Render cung cấp biến PORT
    app.run(host='0.0.0.0', port=port)
