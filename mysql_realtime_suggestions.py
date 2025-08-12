import pickle
import mysql.connector
import pandas as pd
import msvcrt
import time
from pathlib import Path

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',  # Change this
    'database': 'specializations_db'
}

HARDCODED_COURSE_ID = 12

BASE = Path("ml_specialization_module")
VEC_FILE = BASE / "spec_vectorizer.pkl"
MODEL_FILE = BASE / "spec_nn_model.pkl"
LOOKUP_FILE = BASE / "specializations_lookup.csv"

SPEC_COL = "specilization_name"

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_models():
    """Load ML models and lookup data"""
    vectorizer = pickle.load(open(VEC_FILE, "rb"))
    nn = pickle.load(open(MODEL_FILE, "rb"))
    df = pd.read_csv(LOOKUP_FILE)
    return vectorizer, nn, df

def get_suggestions_from_db(query, course_id=None):
    """Get suggestions using database + ML model"""
    if not query.strip():
        return []
    
    vectorizer, nn, df = load_models()
    
    query_normalized = " ".join(str(query).lower().replace(".", "").replace(",", "").strip().split())
    X_query = vectorizer.transform([query_normalized])
    
    # Dynamic threshold
    query_len = len(query_normalized.replace(" ", ""))
    if query_len <= 2:
        threshold = 3
    elif query_len <= 5:
        threshold = 20
    else:
        threshold = 40
    
    distances, indices = nn.kneighbors(X_query, n_neighbors=min(len(df), 100))
    
    suggestions = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        
        if course_id and row['course_id'] != course_id:
            continue
            
        spec_name = row[SPEC_COL]
        spec_id = int(row['id'])
        score = max(0, int((1 - dist) * 100))
        
        if score >= threshold:
            suggestions.append((spec_name, spec_id, score))
    
    suggestions.sort(key=lambda x: -x[2])
    return suggestions

def realtime_mode():
    """Real-time suggestion mode with MySQL backend"""
    title = f"MySQL Real-time Suggestions (Course ID: {HARDCODED_COURSE_ID})"
    print(title)
    print("Start typing (ESC to exit)\n")
    current_input = ""

    while True:
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\x1b':
                break
            elif key == b'\r':
                print("\n")
                continue
            elif key == b'\x08':
                if current_input:
                    current_input = current_input[:-1]
            elif 32 <= ord(key) <= 126:
                current_input += key.decode('utf-8')

            print("\033[2J\033[H", end="")
            print(title)
            print("Start typing (ESC to exit)\n")
            print(f"Query: {current_input}")

            if current_input:
                suggestions = get_suggestions_from_db(current_input, HARDCODED_COURSE_ID)
                if suggestions:
                    print(f"\nSuggestions ({len(suggestions)}):")
                    for name, spec_id, score in suggestions:
                        print(f"  - {name} ({score}%) [ID: {spec_id}]")
                else:
                    print("\nNo matches found")
            print("\n" + "-" * 50)

if __name__ == "__main__":
    print(f"=== MYSQL REAL-TIME SUGGESTIONS - COURSE_ID: {HARDCODED_COURSE_ID} ===")
    print("Auto-retrains every 5 minutes if database is updated")
    print("Change HARDCODED_COURSE_ID to test different courses")
    print()
    
    realtime_mode()