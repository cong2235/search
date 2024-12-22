from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_embeddings, create_faiss_index, search_similar_sentence_faiss
import os
import uvicorn
# Tạo ứng dụng FastAPI
app = FastAPI()

# Load sentences và embeddings
EMBEDDINGS_FILE = "app/embeddings.pkl"
sentences, embeddings = load_embeddings(EMBEDDINGS_FILE)

# Tạo FAISS Index
faiss_index = create_faiss_index(embeddings)

# Định nghĩa schema đầu vào
class SearchRequest(BaseModel):
    input_sentence: str
    threshold: float = 0.8
    top_k: int = 5

# API gốc
@app.get("/")
def root():
    return {"message": "Welcome to the Sentence Similarity API!"}

# Endpoint tìm kiếm câu tương tự
@app.post("/search")
def search_sentence(request: SearchRequest):
    results = search_similar_sentence_faiss(request.input_sentence, sentences, faiss_index, threshold=request.threshold, k=request.top_k)
    
    if results and results[0]["sentence"]:
        return {"input_sentence": request.input_sentence, "results": results}
    else:
        return {"input_sentence": request.input_sentence, "message": "No similar sentence found."}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)

