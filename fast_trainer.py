#!/usr/bin/env python3
"""
Fast Pure ML Trainer - Character-level learning without rule-based abbreviations
"""
import mysql.connector
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root',
    'database': 'resume_analyzer'
}

BASE_DIR = Path("ml_specialization_module")
VECTOR_FILE = BASE_DIR / "spec_vectorizer.pkl"
MODEL_FILE = BASE_DIR / "spec_nn_model.pkl"
LOOKUP_FILE = BASE_DIR / "specializations_lookup.csv"

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

def load_data_from_db():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM ups_courses_specialization", conn)
    conn.close()
    return df

def generate_simple_variants(text):
    """Generate minimal variants for fast training"""
    variants = [text]
    text_clean = text.lower().replace(' ', '')
    
    # Add text without spaces
    variants.append(text_clean)
    
    # Add individual words
    words = text.split()
    for word in words:
        if len(word) > 2:
            variants.append(word.lower())
    
    # Add initials
    if len(words) > 1:
        initials = ''.join([w[0] for w in words if w])
        variants.append(initials)
    
    # Add short prefixes
    for length in [2, 3, 4]:
        if len(text_clean) >= length:
            variants.append(text_clean[:length])
    
    return list(set(variants))

def train_fast_model():
    print("=== Fast ML Training ===")
    
    # Load data
    print("Loading data...")
    source_df = load_data_from_db()
    print(f"Loaded {len(source_df)} specializations")
    
    # Generate minimal training data
    print("Generating variants...")
    expanded_data = []
    
    for _, row in source_df.iterrows():
        spec = " ".join(str(row['specilization_name']).lower().replace(".", "").strip().split())
        variants = generate_simple_variants(spec)
        
        for variant in variants:
            new_row = row.copy()
            new_row['spec_normalized'] = variant
            expanded_data.append(new_row)
    
    expanded_df = pd.DataFrame(expanded_data)
    specializations = expanded_df['spec_normalized'].tolist()
    
    print(f"Generated {len(specializations)} training entries")
    
    # Train model
    print("Training model...")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", 
        ngram_range=(1, 4),
        max_features=8000,
        lowercase=True
    )
    X = vectorizer.fit_transform(specializations)
    nn_model = NearestNeighbors(n_neighbors=len(specializations), metric="cosine")
    nn_model.fit(X)
    
    # Save
    print("Saving...")
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(VECTOR_FILE, "wb") as f:
        pickle.dump(vectorizer, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(nn_model, f)
    
    expanded_df.to_csv(LOOKUP_FILE, index=False)
    
    print("SUCCESS: Fast training complete!")
    print(f"Entries: {len(specializations)}")

if __name__ == "__main__":
    train_fast_model()