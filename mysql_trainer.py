import mysql.connector
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import threading
import time

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'password',  # Change this
    'database': 'specializations_db'
}

BASE_DIR = Path("ml_specialization_module")
VECTOR_FILE = BASE_DIR / "spec_vectorizer.pkl"
MODEL_FILE = BASE_DIR / "spec_nn_model.pkl"
LOOKUP_FILE = BASE_DIR / "specializations_lookup.csv"
LAST_UPDATE_FILE = BASE_DIR / "last_update.txt"

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def get_last_db_update():
    """Get the latest update timestamp from database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT MAX(updated_at) FROM specializations")
    result = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return result

def get_last_model_update():
    """Get the last model training timestamp"""
    if LAST_UPDATE_FILE.exists():
        return LAST_UPDATE_FILE.read_text().strip()
    return "1970-01-01 00:00:00"

def load_data_from_db():
    """Load specialization data from MySQL"""
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM specializations", conn)
    conn.close()
    return df

def generate_abbreviation_variants(text):
    """Generate abbreviation variants"""
    variants = [text]
    
    # Generate initials
    words = text.split()
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    # Common abbreviations
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
    """Train ML model from database data"""
    print("Training model from database...")
    df = load_data_from_db()
    
    # Generate expanded training data
    expanded_data = []
    for _, row in df.iterrows():
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
    nn_model = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
    nn_model.fit(X)
    
    # Save artifacts
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn_model, f)
    
    expanded_df.to_csv(LOOKUP_FILE, index=False)
    LAST_UPDATE_FILE.write_text(str(get_last_db_update()))
    
    print(f"Model trained on {len(specializations)} entries (expanded from {len(df)})")

def auto_retrain_daemon():
    """Background daemon to check for updates every 5 minutes"""
    while True:
        try:
            db_update = str(get_last_db_update())
            model_update = get_last_model_update()
            
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

if __name__ == "__main__":
    train_model()
    start_auto_retrain()
    print("Press Ctrl+C to stop...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopped.")