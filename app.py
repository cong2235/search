from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from difflib import SequenceMatcher
from waitress import serve

app = Flask(__name__)
CORS(app)

es = Elasticsearch(
    hosts=["https://d93143eb81aa40ae9b186eeee81a1adc.us-central1.gcp.cloud.es.io"],
    basic_auth=("elastic", "YgZyYW1VLobvLzpWvVup0ZwE"),
    request_timeout=60
)

index_name = "japanese_sentences"

def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

@app.route('/search', methods=['POST'])
def search_and_compare():
    query = request.json.get('query')
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "sentence": query
                }
            },
            "size": 10
        }
    )

    results = []
    for hit in response['hits']['hits']:
        sentence = hit['_source']['sentence']
        similarity = calculate_similarity(query, sentence)
        results.append({"sentence": sentence, "similarity": similarity})

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)[:2]

    if results and results[0]['similarity'] > 0.8:
        return jsonify({"text": results[0]['sentence']})
    else:
        return jsonify({"text": "no similar sentence found"})

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
