import pickle
import pandas as pd
import msvcrt
import time
import threading
import mysql.connector
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

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
LAST_UPDATE_FILE = BASE / "last_update.txt"

SPEC_COL = "specilization_name"

# Global variables for models
vectorizer = None
nn = None
df = None

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_data_from_db():
    """Load specialization data from MySQL"""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM specializations", conn)
    conn.close()
    return df

def generate_abbreviation_variants(text):
    """Generate abbreviation variants"""
    variants = [text]
    
    words = text.split()
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    short_forms = {
        'computer science': ['cs'], 'artificial intelligence': ['ai'],
        'machine learning': ['ml'], 'information technology': ['it'],
        'electronics and communication': ['ece'], 'electronics and telecommunications': ['extc'],
        'master of business administration': ['mba'], 'bachelor of business administration': ['bba'],
        'bachelor of medicine bachelor of surgery': ['mbbs'], 'bachelor of dental surgery': ['bds'],
        'bachelor of pharmacy': ['bpharm'], 'master of pharmacy': ['mpharm'],
        'bachelor of commerce': ['bcom'], 'master of commerce': ['mcom'],
        'bachelor of computer applications': ['bca'], 'master of computer applications': ['mca'],
        'bachelor of arts': ['ba'], 'master of arts': ['ma'],
        'bachelor of science': ['bsc'], 'master of science': ['msc'],
        'bachelor of laws': ['llb'], 'master of laws': ['llm']
    }
    
    text_lower = text.lower()
    for full_form, abbrevs in short_forms.items():
        if full_form in text_lower:
            variants.extend(abbrevs)
    
    return list(set(variants))

def train_model():
    """Train ML model from MySQL database"""
    global vectorizer, nn, df
    
    print("Training from MySQL database...")
    source_df = load_data_from_db()
    
    # Generate expanded training data
    expanded_data = []
    for _, row in source_df.iterrows():
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        variants = generate_abbreviation_variants(spec)
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    
    # Train model
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    X = vectorizer.fit_transform(specializations)
    nn = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
    nn.fit(X)
    
    # Save artifacts
    BASE.mkdir(parents=True, exist_ok=True)
    with open(VEC_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn, f)
    
    df = expanded_df
    
    # Save last update timestamp
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(updated_at) FROM specializations")
    last_update = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    LAST_UPDATE_FILE.write_text(str(last_update))
    
    print(f"Model trained on {len(specializations)} entries (expanded from {len(source_df)})")

def load_models():
    """Load existing models or train new ones"""
    global vectorizer, nn, df
    
    try:
        vectorizer = pickle.load(open(VEC_FILE, "rb"))
        nn = pickle.load(open(MODEL_FILE, "rb"))
        # Load data from database for current session
        source_df = load_data_from_db()
        expanded_data = []
        for _, row in source_df.iterrows():
            spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
            variants = generate_abbreviation_variants(spec)
            for variant in variants:
                new_row = row.copy()
                new_row['spec_normalized'] = variant
                expanded_data.append(new_row)
        df = pd.DataFrame(expanded_data)
        print("Loaded existing models with current database data")
    except:
        print("No existing models found, training new ones...")
        train_model()

def auto_retrain_daemon():
    """Background daemon to check for updates every 5 minutes"""
    while True:
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(updated_at) FROM specializations")
            db_update = str(cursor.fetchone()[0])
            cursor.close()
            conn.close()
            
            model_update = "1970-01-01 00:00:00"
            if LAST_UPDATE_FILE.exists():
                model_update = LAST_UPDATE_FILE.read_text().strip()
            
            if db_update > model_update:
                print(f"Database updated: {db_update} > {model_update}")
                train_model()
                print("Auto-retrain completed!")
                
        except Exception as e:
            print(f"Auto-retrain error: {e}")
        
        time.sleep(300)  # 5 minutes

def start_auto_retrain():
    """Start auto-retrain in background thread"""
    daemon = threading.Thread(target=auto_retrain_daemon, daemon=True)
    daemon.start()
    print("Auto-retrain daemon started (checks every 5 minutes)")

def get_suggestions(query):
    """Get suggestions from MySQL database"""
    if not query.strip():
        return []
    
    query_normalized = " ".join(str(query).lower().replace(".", "").replace(",", "").strip().split())
    X_query = vectorizer.transform([query_normalized])
    
    # Dynamic threshold based on query length
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
        
        if row['course_id'] != HARDCODED_COURSE_ID:
            continue
            
        spec_name = row[SPEC_COL]
        spec_id = int(row['id'])
        score = max(0, int((1 - dist) * 100))
        
        if score >= threshold:
            suggestions.append((spec_name, spec_id, score))
    
    suggestions.sort(key=lambda x: -x[2])
    return suggestions

def realtime_mode():
    """Real-time suggestion mode"""
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
                suggestions = get_suggestions(current_input)
                if suggestions:
                    print(f"\nSuggestions ({len(suggestions)}):")
                    for name, spec_id, score in suggestions:
                        print(f"  - {name} ({score}%) [ID: {spec_id}]")
                else:
                    print("\nNo matches found")
            print("\n" + "-" * 50)

if __name__ == "__main__":
    print(f"=== MYSQL SPECIALIZATION SYSTEM - COURSE_ID: {HARDCODED_COURSE_ID} ===")
    
    # Load or train models
    load_models()
    
    # Start auto-retrain daemon
    start_auto_retrain()
    
    print(f"Loaded {len(df)} total specializations")
    print(f"Course {HARDCODED_COURSE_ID} has {len(df[df['course_id'] == HARDCODED_COURSE_ID])} specializations")
    print("[MYSQL-POWERED] Auto-retrains every 5 minutes")
    print("Change HARDCODED_COURSE_ID in the code")
    print("Popular course_ids: 12 (Engineering), 37 (Medical), 46 (Management)")
    print()
    
    realtime_mode()