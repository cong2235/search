from flask import Flask, request, jsonify
from flask_cors import CORS
from elasticsearch import Elasticsearch
from waitress import serve
import logging

app = Flask(__name__)
CORS(app)

es = Elasticsearch(
    hosts=["https://d93143eb81aa40ae9b186eeee81a1adc.us-central1.gcp.cloud.es.io"],
    basic_auth=("elastic", "YgZyYW1VLobvLzpWvVup0ZwE"),
    request_timeout=60
)

index_name = "japanese_sentences"

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/search', methods=['POST'])
def search_and_compare():
    try:
        query = request.json.get('query')
        if not query:
            return jsonify({"error": "Query not provided"}), 400

        search_query = {
            "query": {
                "more_like_this": {
                    "fields": ["sentence"],
                    "like": query,
                    "min_term_freq": 1,
                    "max_query_terms": 12
                }
            },
            "size": 10  # Limit the number of documents retrieved
        }

        # Send search request to Elasticsearch
        response = es.search(index=index_name, body=search_query)

        results = []
        for hit in response['hits']['hits']:
            sentence = hit['_source']['sentence']
            score = hit['_score']
            results.append({"sentence": sentence, "score": score})

        # Sort results by score and get the highest score
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        if results and results[0]['score'] > 35:
            return jsonify({"text": results[0]['sentence']})
        else:
            return jsonify({"text": "no similar sentence found"})
    except Exception as e:
        app.logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5001)
