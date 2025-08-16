from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import os
import sys

# Add current directory to path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from specialization_suggestion_realtime import load_models, start_auto_retrain, get_suggestions

app = Flask(__name__)
CORS(app)

# Load models at startup
load_models()
start_auto_retrain()

@app.route('/api/spec_suggest', methods=['GET'])
def spec_suggest():
    start_time = time.time()

    # Params from frontend
    query = request.args.get('query', '').strip()
    course_id = request.args.get('course_id', type=int)

    if not query or not course_id:
        return jsonify({
            'error': 'query and course_id are required',
            'suggestions': [],
            'count': 0
        })

    # Dynamic limit based on query length (like course system)
    qlen = len(query)
    if qlen <= 1:
        limit = 30
    elif qlen <= 3:
        limit = 20
    elif qlen <= 6:
        limit = 10
    else:
        limit = 8
    
    # Use hybrid ML + Rule-based suggestions with dynamic limit
    suggestions = get_suggestions(query, course_id=course_id, limit=limit)

    response_time = round((time.time() - start_time) * 1000, 2)

    return jsonify({
        'query': query,
        'course_id': course_id,
        'suggestions': [
            {
                'spec_id': spec_id,
                'specialization_name': name,
                'confidence_percentage': score
            }
            for name, spec_id, course_id, score in suggestions
        ],
        'count': len(suggestions),
        'response_time_ms': response_time
    })

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'title': 'Specialization Suggestion API',
        'usage': '/api/spec_suggest?query=TEXT&course_id=12'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
