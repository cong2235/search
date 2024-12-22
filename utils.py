import pickle
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
import os
import gdown



# Load model và tokenizer
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese')

# Hàm tải embeddings.pkl từ Google Drive
def download_embeddings(file_path="embeddings.pkl"):
    if not os.path.exists(file_path):
        url = "https://drive.google.com/uc?id=1RWZhhPqShZjiY7c5I0zZlJAYMH-vE15i" 
        print("Downloading embeddings.pkl from Google Drive...")
        gdown.download(url, file_path, quiet=False)
    return file_path

# Hàm tính toán embedding
def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1)
    return embedding.numpy().flatten()


# Hàm tải embeddings từ file pickle
def load_embeddings(file_path="embeddings.pkl"):
    download_embeddings(file_path)

    # Đọc tệp embeddings
    with open(file_path, 'rb') as f:
        sentences, embeddings = pickle.load(f)
    embeddings_np = np.vstack(embeddings)
    return sentences, embeddings_np

# Hàm khởi tạo FAISS Index
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dimension) 
    index.add(embeddings)  
    return index


# Hàm tìm kiếm câu tương tự
def search_similar_sentence_faiss(input_sentence, sentences, index, threshold=0.8, k=5):
    input_embedding = get_embedding(input_sentence).reshape(1, -1)
    distances, indices = index.search(input_embedding, k)

    # Lấy câu phù hợp nhất nếu vượt qua threshold
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        similarity = 1 / (1 + dist)  # Chuyển đổi khoảng cách thành độ tương đồng
        if similarity >= threshold:
            results.append({"similarity": similarity, "sentence": sentences[idx]})

    if results:
        return results
    else:
        return [{"similarity": 0, "sentence": None}]
